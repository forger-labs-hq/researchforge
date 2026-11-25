"""Unattended auto-run for repository projects: the Kaggle auto-run, for ``autorun``.

``researchforge autorun`` runs the DAG loop while a terminal stays open, with an
API key. This module runs it with nobody watching, the way ``kaggle auto-run``
runs competition rounds:

* a background **monitor** (``researchforge auto-run monitor``, installed as a
  launchd job / systemd timer / scheduled task by ``auto-run service``) runs a
  cycle every ``poll_minutes``: start a launch when the gates allow, stop one
  that outlived ``max_minutes``, look for new papers once a day, and send a
  morning briefing;
* a **launch** is a detached child (``python -m researchforge.autorun.unattended
  <root>``) that runs up to ``rounds_per_launch`` rounds of the real engine;
* the **driver** is the project's AI — ``claude`` or ``cursor`` through their CLIs
  (every tool off: the engine writes and imports the plans itself), or ``byok``
  with an API key from ``~/.config/researchforge/auto_run.env``;
* **approval** stays the person's. With ``ask`` (the default) every plan the loop
  writes waits for **Approve plan** on the dashboard (or ``experiment approve``)
  and a phone alert says so; the next launch runs it. ``standing`` is the typed
  approval ``autorun`` already asks for — "run this batch and every later round
  unattended" — given once, by the person, and revocable.

Gates, all required: auto-run on, an approved contract and a baseline, the loop
not paused, no question holding it, no launch running, no plan waiting for
approval (``ask``), the driver's usage limit not hit, today's rounds under
``max_rounds_per_day``, and the target not reached. State lives in
``.researchforge/``: ``auto_run.json`` (settings), ``auto_run_state.json`` (the
launches), ``auto_run.log``, ``monitor_state.json`` and ``loop_control.json``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from researchforge.kaggle import auto_run as _kaggle
from researchforge.kaggle.workspace import atomic_write_json, read_json

RF_DIR = ".researchforge"
CONFIG_FILE = "auto_run.json"
LOG_FILE = "auto_run.log"
MONITOR_FILE = "monitor_state.json"
APPROVALS = ("ask", "standing")
LAUNCH_GRACE_MINUTES = 30  # the monitor stops a launch this long past max_minutes


def rf_dir(root: Path) -> Path:
    return root / RF_DIR


@dataclass
class RepoAutoRunConfig:
    enabled: bool = False  # off until the person turns it on
    driver: str = "claude"
    rounds_per_launch: int = 1
    max_rounds_per_day: int = 4
    max_minutes: int = 240  # the monitor stops a launch running longer than this
    model: str = ""  # optional model id for the driver
    cli_path: str = ""  # optional explicit path to the claude / cursor-agent binary
    approval: str = "ask"  # "ask": each plan waits for Approve plan; "standing": runs
    standing_approved_at: str = ""  # when the person typed the standing approval
    merge: bool = True  # combine two independent winners when the graph offers it
    metric_gate: bool = True  # False: record every result (the local metric is a signal)
    ntfy_topic: str = ""  # this project's own alert topic ("" = the machine-wide one)
    poll_minutes: int = 15
    papers_hours: float = 24.0  # new-paper scan interval (0 = off)
    briefing_at: str = "07:00"  # daily summary time ("" = off)

    @property
    def standing(self) -> bool:
        return self.approval == "standing" and bool(self.standing_approved_at)

    def save(self, root: Path) -> None:
        rf_dir(root).mkdir(parents=True, exist_ok=True)
        atomic_write_json(rf_dir(root) / CONFIG_FILE, asdict(self))

    @classmethod
    def load(cls, root: Path) -> RepoAutoRunConfig:
        raw = read_json(rf_dir(root) / CONFIG_FILE, None)
        if not isinstance(raw, dict):
            raw = {"driver": _kaggle.suggested_driver(rf_dir(root))[0]}
        known = set(cls.__dataclass_fields__)
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        if cfg.driver not in _kaggle.DRIVERS:
            cfg.driver = "claude"
        if cfg.approval not in APPROVALS:
            cfg.approval = "ask"
        cfg.rounds_per_launch = max(1, min(int(cfg.rounds_per_launch), 20))
        cfg.max_rounds_per_day = max(1, min(int(cfg.max_rounds_per_day), 100))
        cfg.max_minutes = max(10, min(int(cfg.max_minutes), 24 * 60))
        cfg.poll_minutes = max(5, min(int(cfg.poll_minutes), 24 * 60))
        return cfg


SETTABLE = {
    "enabled": bool,
    "driver": str,
    "rounds_per_launch": int,
    "max_rounds_per_day": int,
    "max_minutes": int,
    "model": str,
    "cli_path": str,
    "merge": bool,
    "metric_gate": bool,
    "ntfy_topic": str,
    "poll_minutes": int,
    "papers_hours": float,
    "briefing_at": str,
}


def update_settings(root: Path, changes: dict[str, Any]) -> RepoAutoRunConfig:
    """Apply setting changes. Approval is not settable here: see set_approval."""
    cfg = RepoAutoRunConfig.load(root)
    for key, value in changes.items():
        if value is None:
            continue
        if key not in SETTABLE:
            raise ValueError(f"unknown setting {key!r}")
        kind = SETTABLE[key]
        if kind is bool and isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on")
        value = kind(value)
        if key == "driver":
            if value not in _kaggle.DRIVERS:
                raise ValueError(f"driver must be one of {', '.join(_kaggle.DRIVERS)}")
            if value != cfg.driver:
                cfg.model = ""  # a model id belongs to one driver
        if key == "briefing_at" and value and not re.fullmatch(r"\d{2}:\d{2}", str(value)):
            raise ValueError("briefing time must be HH:MM, or empty for none")
        setattr(cfg, key, value)
    cfg.save(root)
    return RepoAutoRunConfig.load(root)


def set_approval(root: Path, approval: str, *, typed: str = "") -> RepoAutoRunConfig:
    """Switch between asking for every plan and the person's standing approval.

    ``standing`` needs the person to type "approve" — the same word ``autorun``
    asks for before it runs the first batch and every later round unattended.
    """
    if approval not in APPROVALS:
        raise ValueError("approval must be ask or standing")
    cfg = RepoAutoRunConfig.load(root)
    if approval == "standing":
        if typed.strip().lower() != "approve":
            raise ValueError('type "approve" to let the loop run its plans without asking')
        cfg.standing_approved_at = datetime.now().isoformat(timespec="seconds")
    else:
        cfg.standing_approved_at = ""
    cfg.approval = approval
    cfg.save(root)
    return cfg


# ---------------------------------------------------------------------------
# Driver readiness and the AI it gives the engine
# ---------------------------------------------------------------------------


def _kcfg(cfg: RepoAutoRunConfig) -> _kaggle.AutoRunConfig:
    return _kaggle.AutoRunConfig(driver=cfg.driver, model=cfg.model, cli_path=cfg.cli_path)


def readiness(cfg: RepoAutoRunConfig, *, check_login: bool = True) -> _kaggle.Readiness:
    return _kaggle.readiness(_kcfg(cfg), check_login=check_login)


class UsageLimitHit(RuntimeError):
    """The driver said its usage limit is reached; later calls fail at once."""


class _LimitWatch:
    """Wraps the driver's provider: notes a usage limit, then refuses further calls.

    The engine treats a failed planning call as one bad hypothesis and moves on,
    so without this a limit would burn every remaining call of the launch.
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.hit: tuple[str, datetime] | None = None

    @property
    def name(self) -> str:
        return str(self.provider.name)

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        if self.hit is not None:
            raise UsageLimitHit(self.hit[0])
        try:
            return str(self.provider.generate(system, user, max_tokens=max_tokens))
        except Exception as exc:
            limit = _kaggle.usage_limit_in(str(exc))
            if limit is not None:
                self.hit = limit
                raise UsageLimitHit(limit[0]) from exc
            raise


def driver_provider(root: Path, cfg: RepoAutoRunConfig) -> Any:
    """The AI the engine plans and re-synthesizes with: the project's driver, metered."""
    from researchforge.kaggle.token_log import metered

    if cfg.driver in ("claude", "cursor"):
        ready = readiness(cfg)
        if not ready.ready or not ready.binary:
            raise RuntimeError(f"the {cfg.driver} driver is not ready: {ready.detail}")
        provider: Any = _kaggle.AgentCLIProvider(cfg.driver, ready.binary, cfg.model, timeout=1200)
    else:
        provider = _kaggle.api_provider()
        if provider is None:
            raise RuntimeError("no AI API key reaches auto-run (see Keys & accounts)")
    return metered(provider, rf_dir(root), "repo_auto_run")


# ---------------------------------------------------------------------------
# Plans and rounds
# ---------------------------------------------------------------------------


def _db(root: Path) -> sqlite3.Connection:
    """The project database for a launch, which writes (schema upgraded if needed)."""
    from researchforge.storage.db import open_project_db

    return open_project_db(root)


def _read(root: Path) -> sqlite3.Connection:
    """A read-only connection: status, gates and the dashboard must never write.

    ``open_project_db`` records the schema version on every open, which would make
    an idle project look freshly active each time the dashboard polls it.
    """
    conn = sqlite3.connect(f"file:{rf_dir(root) / 'researchforge.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _loop_facts(conn: sqlite3.Connection) -> tuple[str, Any, float]:
    """(metric, direction, baseline) the engine would run with; RuntimeError if it cannot.

    The checks of the engine's ``_prepare``, without creating its research log.
    """
    from researchforge.storage.baseline_repository import get_latest_baseline
    from researchforge.storage.contract_repository import get_active_contract
    from researchforge.storage.project_repository import get_project

    project = get_project(conn)
    if project is None or not project.objective:
        raise RuntimeError("No project found. Run `researchforge project create` first.")
    baseline = get_latest_baseline(conn)
    if baseline is None or baseline.metrics is None:
        raise RuntimeError("No frozen baseline. Run `researchforge baseline run` first.")
    contract = get_active_contract(conn)
    if contract is None:
        raise RuntimeError("No approved contract. Run `researchforge contract approve` first.")
    metric = contract.spec.objective.primary_metric
    return metric.name, metric.direction, baseline.metrics.primary_metric.value


def plans_waiting(root: Path) -> list[dict[str, Any]]:
    """Plans whose experiments wait for the person's approval, oldest first."""
    from researchforge.domain.experiment import ExperimentStatus, PlanStatus
    from researchforge.storage.experiment_repository import list_experiments, list_plans

    out = []
    with contextlib.closing(_read(root)) as conn:
        for plan in list_plans(conn):
            if plan.status is not PlanStatus.PLANNED:
                continue
            runnable = [
                e
                for e in list_experiments(conn, plan.plan_id)
                if e.status is ExperimentStatus.PLANNED
            ]
            if runnable:
                out.append(
                    {
                        "plan_id": plan.plan_id,
                        "hypothesis_id": plan.hypothesis_id,
                        "approach": plan.approach_summary,
                        "experiments": [f"{e.experiment_id}: {e.title}" for e in runnable],
                    }
                )
    return out


def plans_approved(root: Path) -> list[str]:
    """Plans the person approved that have experiments still to run, oldest first."""
    from researchforge.domain.experiment import ExperimentStatus, PlanStatus
    from researchforge.storage.experiment_repository import list_experiments, list_plans

    with contextlib.closing(_read(root)) as conn:
        return [
            p.plan_id
            for p in list_plans(conn)
            if p.status is PlanStatus.APPROVED
            and any(
                e.status is ExperimentStatus.APPROVED for e in list_experiments(conn, p.plan_id)
            )
        ]


def rounds_today(root: Path) -> int:
    today = datetime.now().date().isoformat()
    state = _kaggle.load_state(rf_dir(root))
    return sum(
        int(h.get("rounds_run") or 0)
        for h in state.get("history", [])
        if isinstance(h, dict) and str(h.get("finished_at") or "").startswith(today)
    )


def _loop_ready(root: Path) -> str:
    """Why the engine cannot run here yet (no contract, no baseline), or ""."""
    try:
        with contextlib.closing(_read(root)) as conn:
            _loop_facts(conn)
    except Exception as exc:  # the engine's own explanation is the useful one
        return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return ""


def gates(root: Path, cfg: RepoAutoRunConfig) -> list[str]:
    """Reasons not to start a launch now; empty means go."""
    from researchforge.kaggle import control

    d = rf_dir(root)
    if not (d / "researchforge.db").is_file():
        return ["not a ResearchForge project — run researchforge init here first"]
    reasons: list[str] = []
    if not cfg.enabled:
        reasons.append("auto-run is off")
    blocked = _loop_ready(root)
    if blocked:
        reasons.append(f"the loop cannot run yet: {blocked}")
    if control.LoopControl.load(d).paused:
        reasons.append("the loop is paused")
    holding = control.holding_questions(d)
    if holding:
        reasons.append(f"{len(holding)} question(s) put the loop on hold until you answer")
    if _kaggle.current_launch(d):
        reasons.append("a launch is already running")
    limited = _kaggle.usage_limited_until(d)
    if limited and limited[1] == cfg.driver:
        reasons.append(
            f"the {limited[1]} driver hit its usage limit; it resets {limited[0]:%a %H:%M} "
            "(or switch driver)"
        )
    if not cfg.standing and not blocked:
        waiting = plans_waiting(root)
        if waiting:
            ids = ", ".join(w["plan_id"] for w in waiting[:3])
            reasons.append(
                f"{ids} wait{'s' if len(waiting) == 1 else ''} for your approval "
                "(Approve plan on the dashboard)"
            )
    done = rounds_today(root)
    if done >= cfg.max_rounds_per_day:
        reasons.append(f"{done} rounds today reaches the daily limit of {cfg.max_rounds_per_day}")
    last = _kaggle.load_state(d).get("last") or {}
    if isinstance(last, dict) and str(last.get("stop_reason") or "").startswith(
        "objective achieved"
    ):
        reasons.append(f"the target is reached ({last['stop_reason']}) — raise it to go on")
    return reasons


# ---------------------------------------------------------------------------
# Launches
# ---------------------------------------------------------------------------


def child_env(root: Path, cfg: RepoAutoRunConfig) -> dict[str, str]:
    env = _kaggle.child_env(rf_dir(root), cfg.driver)
    env.pop("RESEARCHFORGE_OVERNIGHT_DIR", None)  # a repo project, not a competition
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    from researchforge.kaggle import control

    topic = control.notify_topic(rf_dir(root))[0]
    if topic:  # the engine's per-experiment alerts go to this project's topic too
        env["NTFY_TOPIC"] = topic
    return env


def launch(root: Path, cfg: RepoAutoRunConfig) -> dict[str, Any]:
    """Spawn the detached child that runs one launch. Returns its record."""
    d = rf_dir(root)
    log = (d / LOG_FILE).open("a")
    proc = subprocess.Popen(
        [sys.executable, "-m", "researchforge.autorun.unattended", str(root)],
        cwd=str(root),
        env=child_env(root, cfg),
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    record = {
        "pid": proc.pid,
        "driver": cfg.driver,
        "approval": "standing" if cfg.standing else "ask",
        "rounds_requested": cfg.rounds_per_launch,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
    }
    state = _kaggle.load_state(d)
    state["current"] = record
    _kaggle._save_state(d, state)
    return record


def start_if_ready(root: Path) -> tuple[str, str]:
    """One monitor cycle of the auto-run sensor: report, or start a launch."""
    cfg = RepoAutoRunConfig.load(root)
    running = _kaggle.current_launch(rf_dir(root))
    if running:
        return "ok", f"{running['driver']} launch running since {running['started_at']}"
    blockers = gates(root, cfg)
    if blockers:
        return "skipped", "; ".join(blockers)
    ready = readiness(cfg)
    if not ready.ready:
        return "skipped", f"waiting for the {cfg.driver} driver: {ready.detail} — fix: {ready.fix}"
    rec = launch(root, cfg)
    rounds = cfg.rounds_per_launch
    return "ok", f"started {cfg.driver} for up to {rounds} round(s), pid {rec['pid']}"


def stop_launch(root: Path) -> dict[str, Any] | None:
    return _kaggle.stop_launch(rf_dir(root))


def _run_approved(conn: sqlite3.Connection, plan_id: str, metric_gate: bool) -> dict[str, Any]:
    """Run a plan the person approved: screening then full benchmark."""
    from dataclasses import replace

    from researchforge.execution.experiments import execute_run, start_run

    prep, run = start_run(conn, plan_id)
    if not metric_gate:
        prep = replace(prep, settings=prep.settings.model_copy(update={"metric_gate": False}))
    summary = execute_run(conn, prep, run)
    return {"plan_id": plan_id, "counts": dict(summary.counts), "promising": summary.promising}


def _best_line(conn: sqlite3.Connection) -> str:
    from researchforge.autorun.engine import get_current_best

    metric, direction, baseline = _loop_facts(conn)
    best, value = get_current_best(conn, baseline, direction)
    where = best.experiment_id if best is not None else "baseline"
    return f"best {metric} {value:.4f} ({where}; baseline {baseline:.4f})"


def run_launch(root: Path) -> dict[str, Any]:
    """The launch itself: run approved plans, then rounds of the engine. Returns its outcome.

    With ``ask``, every plan the engine writes is declined at its gate, so the
    launch stops there and the plan waits for the person; with ``standing`` the
    engine runs its plans like ``autorun --yes``.
    """
    from researchforge.ai.service import provider_override
    from researchforge.autorun.engine import AutorunConfig, run_autorun
    from researchforge.execution.experiments import ExperimentBlockedError
    from researchforge.kaggle import control
    from researchforge.storage.contract_repository import get_active_contract

    cfg = RepoAutoRunConfig.load(root)
    d = rf_dir(root)
    said: list[str] = []

    def progress(message: str) -> None:
        print(message, flush=True)
        said.append(message)

    rounds = 0
    ran: list[dict[str, Any]] = []
    with contextlib.closing(_db(root)) as conn:
        for plan_id in plans_approved(root):
            if rounds >= cfg.rounds_per_launch:
                break
            progress(f"Running approved {plan_id}…")
            try:
                ran.append(_run_approved(conn, plan_id, cfg.metric_gate))
            except ExperimentBlockedError as exc:
                progress(f"  ✗ {plan_id} blocked: {exc}")
                control.notify(d, "alert", f"Auto-run could not run {plan_id}: {exc}"[:300])
                continue
            rounds += 1
        for r in ran:
            counts = ", ".join(f"{v} {k}" for k, v in sorted(r["counts"].items()))
            control.notify(d, "result", f"{r['plan_id']} ran: {counts}. {_best_line(conn)}")

        outcome: dict[str, Any] = {"plans_run": [r["plan_id"] for r in ran]}
        left = cfg.rounds_per_launch - rounds
        if left > 0:
            before = {w["plan_id"] for w in plans_waiting(root)}
            watch = _LimitWatch(driver_provider(root, cfg))
            contract = get_active_contract(conn)
            target = contract.spec.objective.primary_metric.target_value if contract else None
            config = AutorunConfig(
                max_rounds=left,
                max_hours=max(0.1, cfg.max_minutes / 60),
                target_value=target,
                merge=cfg.merge,
                yes=cfg.standing,
                ask_every_plan=not cfg.standing,
                no_metric_gate=not cfg.metric_gate,
            )
            with provider_override(watch):
                result = run_autorun(
                    conn, config, on_progress=progress, gate=None if cfg.standing else _decline
                )
            rounds += sum(1 for r in result.rounds if r.experiments_run)
            outcome["stop_reason"] = result.stop_reason
            if watch.hit is not None:
                message, until = watch.hit
                outcome["usage_limit"] = {"message": message, "until": until}
            new = [w for w in plans_waiting(root) if w["plan_id"] not in before]
            for w in new:
                control.notify(
                    d,
                    "interaction",
                    f"{w['plan_id']} waits for your approval: {w['approach'][:160]} "
                    f"({len(w['experiments'])} experiment(s)). Approve plan on the dashboard, "
                    f"or researchforge experiment approve {w['plan_id']}.",
                    priority="high",
                )
            outcome["plans_waiting"] = [w["plan_id"] for w in new]
        outcome["rounds_run"] = rounds
        outcome["best"] = _best_line(conn)
    return outcome


def _decline(preview: Any) -> bool:
    """The gate in ``ask`` mode: the person approves this plan, not the loop."""
    return False


def _finish(root: Path, outcome: dict[str, Any]) -> None:
    from researchforge.kaggle import control

    d = rf_dir(root)
    state = _kaggle.load_state(d)
    record = {**(state.get("current") or {}), **outcome}
    history = [h for h in state.get("history", []) if isinstance(h, dict)][-19:]
    state["history"] = [*history, record]
    state["current"] = None
    state["last"] = record
    _kaggle._save_state(d, state)
    status = record.get("status")
    if status in ("finished", "stopped"):
        how = "" if status == "finished" else ", then stopped on request"
        waiting = record.get("plans_waiting") or []
        ask = f" {', '.join(waiting)} wait(s) for your approval." if waiting else ""
        msg = (
            f"Auto-run ({record.get('driver')}): {record.get('rounds_run', 0)} round(s) "
            f"ran{how}. {record.get('best', '')}.{ask}"
        )
        control.notify(d, "briefing", msg[:300])
    elif status == "usage_limit":
        control.notify(d, "alert", f"Auto-run paused: {record.get('error', '')}"[:300])
    else:
        control.notify(
            d,
            "alert",
            f"Auto-run ({record.get('driver')}) {status}: {record.get('error', '')}"[:300],
            priority="high",
        )


def run_child(root: Path) -> int:
    """Entry point of the launch child."""
    os.chdir(root)
    cfg = RepoAutoRunConfig.load(root)
    started = datetime.now().isoformat(timespec="seconds")
    print(
        f"[{started}] repo auto-run {cfg.driver} ({'standing approval' if cfg.standing else 'ask'})"
        f": up to {cfg.rounds_per_launch} round(s), limit {cfg.max_minutes} min",
        flush=True,
    )
    signal.signal(signal.SIGTERM, _kaggle._on_sigterm)
    outcome: dict[str, Any]
    try:
        outcome = {"status": "finished", **run_launch(root)}
    except _kaggle._StopRequested:
        outcome = {"status": "stopped", "error": "stopped on request"}
    except Exception as exc:  # recorded and alerted, never silent
        outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:400]}
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    hit = outcome.pop("usage_limit", None)
    if hit is not None:
        until = hit["until"]
        outcome["status"] = "usage_limit"
        outcome["error"] = f"{hit['message']} — auto-run waits until {until:%a %H:%M}"
        state = _kaggle.load_state(rf_dir(root))
        state["usage_limit"] = {
            "driver": cfg.driver,
            "message": hit["message"],
            "until": until.isoformat(timespec="minutes"),
        }
        _kaggle._save_state(rf_dir(root), state)
    outcome.setdefault("rounds_run", 0)
    outcome["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _finish(root, outcome)
    print(f"[{outcome['finished_at']}] repo auto-run {outcome['status']}", flush=True)
    return 0 if outcome["status"] == "finished" else 1


# ---------------------------------------------------------------------------
# The monitor cycle
# ---------------------------------------------------------------------------

SENSORS = (
    ("auto_run", "Starts a launch when the gates allow; plans wait for your approval"),
    ("launch_watch", "Stops a launch that outlives its time limit"),
    ("papers", "Looks for new arXiv papers on the objective once a day"),
    ("briefing", "A daily summary to your phone"),
)


def _record(state: dict[str, Any], name: str, status: str, detail: str = "") -> None:
    state.setdefault("sensors", {})[name] = {
        "status": status,
        "detail": detail[:300],
        "last_run_at": datetime.now().isoformat(timespec="seconds"),
    }


def _watch_launch(root: Path, cfg: RepoAutoRunConfig) -> tuple[str, str]:
    running = _kaggle.current_launch(rf_dir(root))
    if not running:
        return "ok", "no launch running"
    started = datetime.fromisoformat(str(running["started_at"]))
    minutes = (datetime.now() - started).total_seconds() / 60
    if minutes > cfg.max_minutes + LAUNCH_GRACE_MINUTES:
        stop_launch(root)
        from researchforge.kaggle import control

        control.notify(
            rf_dir(root),
            "alert",
            f"Auto-run launch ran {minutes:.0f} min (limit {cfg.max_minutes}); stopped it.",
            priority="high",
        )
        return "error", f"stopped after {minutes:.0f} min (limit {cfg.max_minutes})"
    return "ok", f"launch running {minutes:.0f} min of {cfg.max_minutes}"


def _papers(root: Path, cfg: RepoAutoRunConfig, state: dict[str, Any]) -> tuple[str, str]:
    if cfg.papers_hours <= 0:
        return "skipped", "off"
    last = str(state.get("last_papers_at") or "")
    if last and datetime.now() - datetime.fromisoformat(last) < timedelta(hours=cfg.papers_hours):
        return "skipped", f"not due — every {cfg.papers_hours:g}h"
    since = (datetime.fromisoformat(last) if last else datetime.now() - timedelta(days=30)).date()
    with contextlib.closing(_read(root)) as conn:
        before = _paper_count(conn)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchforge",
            "research",
            "search",
            "--since",
            str(since),
            "--json",
        ],
        cwd=str(root),
        env=child_env(root, cfg),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    state["last_papers_at"] = datetime.now().isoformat(timespec="seconds")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"research search exited {proc.returncode}")
    with contextlib.closing(_read(root)) as conn:
        added = _paper_count(conn) - before
    if added > 0:
        from researchforge.kaggle import control

        control.notify(
            rf_dir(root),
            "ideas",
            f"{added} new paper(s) on the objective; the next re-synthesis draws on them.",
        )
    return "ok", f"{max(0, added)} new paper(s) since {since}"


def _paper_count(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row else 0


def _briefing(root: Path, cfg: RepoAutoRunConfig, state: dict[str, Any]) -> tuple[str, str]:
    if not cfg.briefing_at:
        return "skipped", "off"
    today = datetime.now().date().isoformat()
    if state.get("last_briefing_date") == today:
        return "skipped", f"sent today; daily at {cfg.briefing_at}"
    if datetime.now().strftime("%H:%M") < cfg.briefing_at:
        return "skipped", f"not due — daily at {cfg.briefing_at}"
    from researchforge.kaggle import control

    with contextlib.closing(_read(root)) as conn:
        best = _best_line(conn) if not _loop_ready(root) else "the loop is not set up yet"
    waiting = plans_waiting(root)
    hist = _kaggle.load_state(rf_dir(root)).get("history", [])
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    ran = sum(
        int(h.get("rounds_run") or 0)
        for h in hist
        if isinstance(h, dict) and str(h.get("finished_at") or "")[:10] in (yesterday, today)
    )
    parts = [f"{root.name}: {best}.", f"{ran} round(s) since yesterday."]
    if waiting:
        parts.append(f"{', '.join(w['plan_id'] for w in waiting)} wait for your approval.")
    gate = gates(root, cfg)
    if gate:
        parts.append("Waiting: " + "; ".join(gate)[:140])
    control.notify(rf_dir(root), "briefing", " ".join(parts)[:400])
    state["last_briefing_date"] = today
    return "ok", "briefing sent"


def monitor_cycle(root: Path) -> dict[str, Any]:
    """Run every sensor once. Returns the monitor state written."""
    d = rf_dir(root)
    cfg = RepoAutoRunConfig.load(root)
    state: dict[str, Any] = read_json(d / MONITOR_FILE, {}) or {}
    sensors: tuple[tuple[str, Callable[[], tuple[str, str]]], ...] = (
        ("launch_watch", lambda: _watch_launch(root, cfg)),
        ("auto_run", lambda: start_if_ready(root)),
        ("papers", lambda: _papers(root, cfg, state)),
        ("briefing", lambda: _briefing(root, cfg, state)),
    )
    for name, fn in sensors:
        try:
            status, detail = fn()
        except Exception as exc:
            status, detail = "error", f"{type(exc).__name__}: {exc}"
        _record(state, name, status, detail)
    from researchforge.plugins import MonitorEvent, run_monitor_hooks

    for name, status, detail in run_monitor_hooks(MonitorEvent("repo", root, d)):
        _record(state, name, status, detail)
    now = datetime.now()
    state["last_cycle_at"] = now.isoformat(timespec="seconds")
    state["next_cycle_at"] = (now + timedelta(minutes=cfg.poll_minutes)).isoformat(
        timespec="seconds"
    )
    atomic_write_json(d / MONITOR_FILE, state)
    return state


# ---------------------------------------------------------------------------
# The background service (launchd / systemd timer / Task Scheduler)
# ---------------------------------------------------------------------------


def service_key(root: Path) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-") or "project"
    return f"{name[:40]}-{hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:6]}"


def service_label(root: Path) -> str:
    return f"com.researchforge.repo.{service_key(root)}"


def _share() -> Path:
    return Path.home() / ".local" / "share" / "researchforge"


def _plist(root: Path) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{service_label(root)}.plist"


def service_log(root: Path) -> Path:
    return _share() / "logs" / f"repo_{service_key(root)}.log"


def _write_launcher(root: Path) -> Path:
    """A launcher outside ~/Desktop: launchd may not read a venv or a log there (TCC)."""
    import sysconfig

    import researchforge as _pkg

    launchers = _share() / "launchers"
    launchers.mkdir(parents=True, exist_ok=True)
    path = launchers / f"repo_monitor_{service_key(root)}.py"
    redirect = (
        f"_log = open({str(service_log(root))!r}, 'a', encoding='utf-8', buffering=1)\n"
        "sys.stdout = sys.stderr = _log\n"
        if sys.platform.startswith("win")
        else ""
    )
    path.write_text(
        f"# Auto-generated launcher: researchforge auto-run monitor for {root}\n"
        "import sys, os\n"
        f"sys.path.insert(0, {sysconfig.get_paths()['purelib']!r})\n"
        f"sys.path.insert(0, {str(Path(_pkg.__file__).resolve().parent.parent)!r})\n"
        f"os.chdir({str(root)!r})\n"
        f"{redirect}"
        f"sys.argv = ['researchforge', 'auto-run', 'monitor', '--once', '--path', {str(root)!r}]\n"
        "from researchforge.cli import app\n"
        "app(standalone_mode=True)\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def install_service(root: Path) -> str:
    """Install the background service that runs a monitor cycle every poll_minutes."""
    cfg = RepoAutoRunConfig.load(root)
    log = service_log(root)
    log.parent.mkdir(parents=True, exist_ok=True)
    launcher = _write_launcher(root)
    # The base interpreter: a venv python under ~/Desktop reads pyvenv.cfg, which TCC blocks.
    python_exe = getattr(sys, "_base_executable", "") or sys.executable
    if sys.platform != "darwin":
        from researchforge.kaggle import service_install

        ok, detail = service_install.install(
            sys.platform,
            f"repo-{service_key(root)}",
            python_exe,
            launcher,
            cfg.poll_minutes / 60,
            log,
        )
        if not ok:
            raise RuntimeError(detail)
        return detail
    path_env = str(Path(python_exe).parent) + ":/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{service_label(root)}</string>
    <key>ProgramArguments</key>
    <array><string>{python_exe}</string><string>{launcher}</string></array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>{path_env}</string>
        <key>HOME</key><string>{Path.home()}</string>
    </dict>
    <key>StartInterval</key><integer>{cfg.poll_minutes * 60}</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>{log}</string>
    <key>StandardErrorPath</key><string>{log}</string>
    <key>WorkingDirectory</key><string>{_share()}</string>
</dict>
</plist>
"""
    target = _plist(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        subprocess.run(["launchctl", "unload", "-w", str(target)], capture_output=True, check=False)
    target.write_text(plist, encoding="utf-8")
    res = subprocess.run(
        ["launchctl", "load", "-w", str(target)], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        raise RuntimeError(f"launchctl load failed: {res.stderr.strip()}")
    return f"launchd job {service_label(root)} (every {cfg.poll_minutes} min)"


def uninstall_service(root: Path) -> str:
    if sys.platform != "darwin":
        from researchforge.kaggle import service_install

        removed, detail = service_install.uninstall(sys.platform, f"repo-{service_key(root)}")
        return detail if removed else f"no service installed ({detail})"
    target = _plist(root)
    if not target.exists():
        return "no service installed"
    subprocess.run(["launchctl", "unload", "-w", str(target)], capture_output=True, check=False)
    target.unlink()
    return f"removed {service_label(root)}"


def service_status(root: Path) -> dict[str, Any]:
    """Whether the service is installed and loaded, with its log's tail."""
    out: dict[str, Any] = {"label": service_label(root), "installed": False, "loaded": False}
    if sys.platform == "darwin":
        out["installed"] = _plist(root).exists()
        if out["installed"]:
            res = subprocess.run(
                ["launchctl", "list", service_label(root)],
                capture_output=True,
                text=True,
                check=False,
            )
            out["loaded"] = res.returncode == 0
            m = re.search(r'"LastExitStatus"\s*=\s*(-?\d+)', res.stdout)
            out["last_exit_status"] = int(m.group(1)) if m else None
    else:
        out["installed"] = out["loaded"] = (
            (_share() / "launchers").joinpath(f"repo_monitor_{service_key(root)}.py").exists()
        )
    log = service_log(root)
    out["log_path"] = str(log)
    if log.exists():
        with contextlib.suppress(OSError):
            out["log_tail"] = log.read_text(errors="replace").splitlines()[-15:]
    return out


def status(root: Path, *, check_login: bool = False) -> dict[str, Any]:
    """Everything the dashboard and ``auto-run status`` show for one project."""
    cfg = RepoAutoRunConfig.load(root)
    d = rf_dir(root)
    state = _kaggle.load_state(d)
    ready = readiness(cfg, check_login=check_login)
    drivers = {}
    for name in _kaggle.DRIVERS:
        r = _kaggle.readiness(_kaggle.AutoRunConfig(driver=name), check_login=False)
        drivers[name] = {"ready": r.ready, "detail": r.detail, "steps": _kaggle.setup_steps(name)}
    monitor = read_json(d / MONITOR_FILE, {}) or {}
    from researchforge.kaggle import control

    ctl = control.LoopControl.load(d)
    return {
        "config": asdict(cfg),
        "enabled": cfg.enabled,
        "standing": cfg.standing,
        "readiness": {"ready": ready.ready, "detail": ready.detail, "fix": ready.fix},
        "drivers": drivers,
        "gates": gates(root, cfg),
        "running": _kaggle.current_launch(d),
        "last": state.get("last"),
        "history": [h for h in state.get("history", []) if isinstance(h, dict)][-10:],
        "rounds_today": rounds_today(root),
        "plans_waiting": plans_waiting(root) if (d / "researchforge.db").is_file() else [],
        "paused": ctl.paused,
        "pause_reason": ctl.pause_reason,
        "notify_topic": control.notify_topic(d)[0],
        "notify_topic_source": control.notify_topic(d)[1],
        "monitor": {
            "last_cycle_at": monitor.get("last_cycle_at"),
            "next_cycle_at": monitor.get("next_cycle_at"),
            "sensors": [
                {"name": n, "description": desc, **(monitor.get("sensors", {}).get(n) or {})}
                for n, desc in SENSORS
            ],
        },
        "service": service_status(root),
        "log": str(d / LOG_FILE),
        "env_file": str(_kaggle.env_file()),
    }


if __name__ == "__main__":  # the launch child
    sys.exit(run_child(Path(sys.argv[1]).resolve()))
