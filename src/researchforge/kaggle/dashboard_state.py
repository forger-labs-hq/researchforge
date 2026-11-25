"""Canonical live state for the dashboard — one versioned payload every view consumes.

``build_state()`` turns what is on disk *right now* into a single JSON-able
dict. Every view (Home, Automation, Runs, DAG, Hypotheses, Submissions,
Timeline, Insights) renders from this payload and nothing else, so views can
never disagree and a refresh can never drift.

Sources, per project root (see :mod:`researchforge.kaggle.workspace`):

* ``.overnight_<slug>/`` — overnight_setup.json, competition_metadata.json,
  experiment_log.db, inflight_rounds.json / execute_round.lock,
  proposal.json, pending_hypotheses.json, watcher_config.json,
  scheduler_config.json + monitor_state.json (sensors), submissions.json,
  token_usage.jsonl, the launchd plist/launcher/log for the slug
* ``.researchforge/`` — the project DB (read-only), autorun.json,
  synthesis/hypotheses.yaml drafts, research-log.md
* ``~/.config/researchforge/notification_log.jsonl``

Nothing here writes. Read failures are reported in ``project.issues`` instead
of being swallowed into an empty list — an empty panel always says why.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import plistlib
import re
import sqlite3
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle import sensors as _sensors
from researchforge.kaggle.workspace import (
    ProjectLocation,
    config_dir,
    discover_locations,
    has_setup,
    location_role,
    read_json,
    slug_of,
)

SCHEMA_VERSION = 1
ACTIVE_WINDOW_HOURS = 24
STALE_RUN_HOURS = 6
PATCH_CHARS = 12000  # patch text shipped per DAG node; the rest stays in the DB


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds") if ts else None


def _norm_ts(value: Any) -> str | None:
    """Normalise any timestamp to local-naive ISO so the UI can sort strings."""
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00").replace(" ", "T"))
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt.isoformat(timespec="seconds")


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ValueError, TypeError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def _issue(issues: list[Any], level: str, message: str, hint: str = "") -> None:
    issues.append({"level": level, "message": message, "hint": hint})


def project_id(root: Path) -> str:
    return hashlib.sha1(str(root).encode()).hexdigest()[:10]


def _ro_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# launchd (cached: launchctl is a subprocess)
# ---------------------------------------------------------------------------

_LAUNCHD_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def scheduler_label(slug: str) -> str:
    return f"com.researchforge.watcher.{slug}"


def _launcher_target(launcher: Path, slug: str) -> Path | None:
    """Which artifacts dir a launcher script's monitor will write to."""
    try:
        code = launcher.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"RESEARCHFORGE_OVERNIGHT_DIR'\]\s*=\s*'([^']+)'", code)
    if m:
        return Path(m.group(1))
    m = re.search(r"os\.chdir\('([^']+)'\)", code)
    if m:
        from researchforge.kaggle.workspace import resolve_artifacts_dir

        return resolve_artifacts_dir(slug, Path(m.group(1)), register=False)
    return None


def launchd_status(slug: str, ttl: float = 10.0) -> dict[str, Any]:
    """Real scheduler state for a slug: plist + `launchctl list` + log tail."""
    label = scheduler_label(slug)
    cached = _LAUNCHD_CACHE.get(label)
    if cached and time.time() - cached[0] < ttl:
        return cached[1]

    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    info: dict[str, Any] = {
        "label": label,
        "plist": str(plist_path),
        "installed": plist_path.exists(),
        "loaded": False,
        "pid": None,
        "last_exit_status": None,
        "state": "not_installed",
        "interval_hours": None,
        "log_path": None,
        "log_tail": [],
        "last_error": "",
        "target_dir": None,
        "program": None,
    }
    if plist_path.exists():
        try:
            with plist_path.open("rb") as fh:
                pl = plistlib.load(fh)
            info["interval_hours"] = (pl.get("StartInterval") or 0) / 3600 or None
            info["log_path"] = pl.get("StandardOutPath")
            args = pl.get("ProgramArguments") or []
            info["program"] = args[0] if args else None
            if len(args) > 1 and args[1].endswith(".py"):
                tgt = _launcher_target(Path(args[1]), slug)
                info["target_dir"] = str(tgt) if tgt else None
            if not info["target_dir"] and info["log_path"]:
                info["target_dir"] = str(Path(info["log_path"]).parent)
        except Exception as exc:  # malformed plist is itself a real state
            info["last_error"] = f"cannot parse plist: {exc}"
        try:
            res = subprocess.run(
                ["launchctl", "list", label], capture_output=True, text=True, timeout=5, check=False
            )
            if res.returncode == 0:
                info["loaded"] = True
                pid = re.search(r'"PID"\s*=\s*(\d+)', res.stdout)
                code = re.search(r'"LastExitStatus"\s*=\s*(-?\d+)', res.stdout)
                info["pid"] = int(pid.group(1)) if pid else None
                if code:
                    raw = int(code.group(1))
                    # launchctl reports a wait(2) status; the exit code is the high byte.
                    info["last_exit_status"] = raw >> 8 if raw > 255 else raw
        except (OSError, subprocess.TimeoutExpired) as exc:
            info["last_error"] = f"launchctl unavailable: {exc}"

        if info["log_path"]:
            try:
                lines = (
                    Path(info["log_path"])
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()
                )
                info["log_tail"] = [ln for ln in lines[-40:] if ln.strip()][-12:]
                for line in reversed(lines[-200:]):
                    if re.search(
                        r"(Error|error:|Traceback|Permission|No module named|Fatal)", line
                    ):
                        info["last_error"] = line.strip()[:300]
                        break
            except OSError:
                pass

        if not info["loaded"]:
            info["state"] = "installed_not_loaded"
        elif info["pid"]:
            info["state"] = "running"
        elif info["last_exit_status"] not in (None, 0):
            info["state"] = "failing"
        else:
            info["state"] = "idle"
    _LAUNCHD_CACHE[label] = (time.time(), info)
    return info


def invalidate_launchd(slug: str) -> None:
    _LAUNCHD_CACHE.pop(scheduler_label(slug), None)


# ---------------------------------------------------------------------------
# Overnight artifacts
# ---------------------------------------------------------------------------

_OVERNIGHT_EXPERIMENT_COLS = [
    "round_num",
    "parent_round_num",
    "branch_tag",
    "source_credit",
    "patch_description",
    "patch_code",
    "screening_passed",
    "screening_metric",
    "screening_duration",
    "full_passed",
    "full_metric",
    "full_improved",
    "full_duration",
    "rejected_reason",
    "started_at",
    "completed_at",
]


def _read_overnight_experiments(
    db: Path, issues: list[Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not db.exists():
        return [], {}
    try:
        with _ro_connect(db) as conn:
            present = {r["name"] for r in conn.execute("PRAGMA table_info(experiments)")}
            if not present:
                _issue(issues, "warn", f"{db.name} has no experiments table yet")
                return [], {}
            cols = [c for c in _OVERNIGHT_EXPERIMENT_COLS if c in present]
            rows = [
                dict(r)
                for r in conn.execute(
                    f"SELECT {', '.join(cols)} FROM experiments ORDER BY round_num"
                )
            ]
            meta: dict[str, Any] = {}
            with contextlib.suppress(sqlite3.Error):
                meta = {
                    r["key"]: r["value"]
                    for r in conn.execute("SELECT key, value FROM run_metadata")
                }
        missing = {"parent_round_num", "branch_tag"} - present
        if missing:
            _issue(
                issues,
                "info",
                f"{db.name} uses an older schema (no {', '.join(sorted(missing))}); "
                "rounds are drawn as branching from the baseline",
            )
        return rows, meta
    except sqlite3.Error as exc:
        _issue(issues, "error", f"cannot read {db}: {exc}")
        return [], {}


def _daily_allowance(d: Path) -> dict[str, Any] | None:
    from researchforge.kaggle.submission import daily_allowance

    try:
        return daily_allowance(d)
    except OSError:
        return None


KERNEL_BOARD_ROWS = 60


def _kernel_board_view(d: Path) -> dict[str, Any] | None:
    """The public-notebook board the kernel watcher keeps (kernel_board), trimmed."""
    from researchforge.kaggle.kernel_board import load_board

    board = load_board(d)
    if not board.get("rows"):
        return None
    return {**board, "rows": board["rows"][:KERNEL_BOARD_ROWS], "total": len(board["rows"])}


def _beats(metric: Any, best: float | None, direction: str) -> bool:
    if not isinstance(metric, (int, float)) or best is None:
        return False
    return metric < best if direction.startswith("min") else metric > best


def _overnight_experiment(
    row: dict[str, Any],
    baseline: float | None,
    direction: str,
    prior_best: float | None = None,
    kept_output: bool = True,
) -> dict[str, Any]:
    """One round. ``prior_best``: the best score of the rounds before it.

    With no baseline score recorded, the loop counts a round as improved when it beat
    every earlier round; that is checked here against ``prior_best``. A round with no
    score is "unscored" only when it kept something to submit (``kept_output``).
    """
    from researchforge.kaggle.benchmark_runner import UNSCORED_PATTERN

    # A run that failed has no score: the log stores 0.0 there as a placeholder.
    if row.get("full_passed") == 0:
        metric = None
    else:
        metric = row.get("full_metric")
        if metric is None and row.get("screening_passed") != 0:
            metric = row.get("screening_metric")
    rejected = row.get("rejected_reason")
    unverified = (
        bool(row.get("full_improved"))
        and baseline is None
        and not _beats(metric, prior_best, direction)
    )
    if row.get("full_improved") and not unverified:
        status = "improved"
    elif (
        row.get("full_passed") == 0 and kept_output and UNSCORED_PATTERN.search(str(rejected or ""))
    ):
        status = "unscored"
    elif rejected or row.get("screening_passed") == 0 or row.get("full_passed") == 0:
        status = "failed"
    elif row.get("completed_at"):
        status = "complete"
    else:
        status = "running"
    parent = row.get("parent_round_num")
    return {
        "id": f"R{row['round_num']}",
        "num": row["round_num"],
        "source": "overnight",
        "title": row.get("patch_description") or f"Round {row['round_num']}",
        "parents": [f"R{parent}"] if parent else [],
        "status": status,
        "stage": "full"
        if row.get("full_metric") is not None
        else ("screening" if row.get("screening_metric") is not None else None),
        "metric": metric,
        "metric_direction": direction,
        "baseline": baseline,
        "branch": row.get("branch_tag") or "",
        "credit": row.get("source_credit") or "",
        "reason": rejected
        or (
            "experiment_log marks this improved, but it did not beat an earlier round "
            "and no baseline metric was recorded in overnight_setup.json"
            if unverified
            else ""
        ),
        "started_at": _norm_ts(row.get("started_at")),
        "completed_at": _norm_ts(row.get("completed_at")),
        "duration_s": (row.get("screening_duration") or 0) + (row.get("full_duration") or 0)
        or None,
        "hypothesis_ids": [],
        "patch": (row.get("patch_code") or "")[:PATCH_CHARS],
        "patch_truncated": len(row.get("patch_code") or "") > PATCH_CHARS,
        "proposal": None,
    }


def _imported_experiment(r: dict[str, Any], direction: str) -> dict[str, Any]:
    """A repo-engine experiment folded into the Kaggle loop's history (never compared)."""
    name = r.get("metric_name") or "its own metric"
    bl = r.get("baseline")
    verdict = {"improved": "improved on its baseline", "failed": "failed"}.get(
        str(r.get("status")), ""
    )
    note = (
        f"Imported from the repo engine, which measured {name} with its own benchmark on this "
        "machine"
        + (f" (its baseline {bl:.4g})" if isinstance(bl, (int, float)) else "")
        + (f"; there it {verdict}" if verdict else "")
        + ". Not comparable with round scores, so it is never best or a parent."
    )
    # "improved" in this history means improved by the Kaggle loop's own measure.
    status = "failed" if r.get("status") == "failed" else "complete"
    return {
        "id": r["repo_id"],
        "num": r.get("num") or 0,
        "source": "overnight",
        "imported": True,
        "title": r.get("title") or r["repo_id"],
        "parents": r.get("parents") or [],
        "status": status,
        "stage": r.get("stage") or None,
        "metric": r.get("metric"),
        "metric_direction": r.get("direction") or direction,
        "baseline": None,
        "branch": "repo engine",
        "credit": "",
        "reason": " ".join(x for x in (r.get("reason") or "", note) if x),
        "started_at": _norm_ts(r.get("started_at")),
        "completed_at": _norm_ts(r.get("completed_at")),
        "duration_s": None,
        "hypothesis_ids": [],
        "patch": r.get("patch") or "",
        "patch_truncated": False,
        "proposal": {"description": r["change"]} if r.get("change") else None,
        "driver": "",
        "auto_run": False,
    }


def _overnight_merge_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The merge the engine would suggest in prepare-round, from the same rule."""
    from types import SimpleNamespace as NS

    from researchforge.kaggle.cli import _detect_merge_opportunity

    rounds = [
        NS(
            round_num=r["round_num"],
            branch_tag=r.get("branch_tag") or "",
            full_result=NS(
                metric_improved=bool(r.get("full_improved")),
                metric_value=r.get("full_metric") or 0.0,
            )
            if r.get("full_passed") is not None
            else None,
        )
        for r in rows
    ]
    m = _detect_merge_opportunity(rounds)
    if not m:
        return None
    return {
        "parents": [f"R{m['parent_a']}", f"R{m['parent_b']}"],
        "metrics": [m.get("metric_a"), m.get("metric_b")],
        "reason": "best improved rounds on two different branch tags (prepare-round's merge rule)",
    }


def _engine_next(d: Path, direction: str, limit: int = 5) -> list[dict[str, Any]]:
    """The engine's own next ideas after the pinned ones, for the Up next queue."""
    from researchforge.kaggle.hypothesis_pool import pin_id, rank_untested

    try:
        ranked = rank_untested(d, direction)
    except Exception:  # a half-written pool must not break the whole view
        return []
    return [
        {"pin_id": pin_id(str(h["idea"])), "text": str(h["idea"]), "why": _why_next(h)}
        for h in ranked
        if h.get("pin") is None
    ][:limit]


def _why_next(h: dict[str, Any]) -> str:
    gap = h.get("claim_vs_our_best")
    if isinstance(gap, (int, float)) and gap > 0:
        return f"notebook claims +{gap:.4g} over our best"
    if h.get("source") == "user":
        return "added by you"
    return f"{h.get('metric_confidence') or 'unknown'} confidence"


def _hypothesis_from_pool(i: int, h: dict[str, Any]) -> dict[str, Any]:
    src = str(h.get("source") or "unknown")
    return {
        "id": f"pool-{i}",
        "source": "overnight",
        "origin": src,
        "title": str(h.get("idea") or "").strip() or "(empty idea)",
        "detail": (
            f"Skipped: {h.get('skip_reason') or 'no reason given'}. "
            if h.get("skipped") and not h.get("tested")
            else ""
        )
        + (h.get("rationale") or ""),
        "status": "tested" if h.get("tested") else "rejected" if h.get("skipped") else "untested",
        "tested": bool(h.get("tested")),
        "priority": h.get("priority"),
        "confidence": h.get("metric_confidence") or "",
        "created_at": _norm_ts(h.get("proposed_at") or h.get("injected_at")),
        "refs": h.get("source_refs") or [],
        "tested_by": [f"R{h['tested_in_round']}"] if h.get("tested_in_round") else [],
        "outcome": h.get("tested_outcome") or "",
        **_kernel_claim(src, h),
    }


def _kernel_claim(origin: str, h: dict[str, Any]) -> dict[str, Any]:
    """The notebook a kernel idea came from and the LB score its title claims."""
    if origin != "kernel_watcher":
        return {}
    from researchforge.kaggle.notebook_baseline import claimed_lb_score

    ref = h.get("kernel_ref")
    claimed = h.get("claimed_score")
    if claimed is None and "claimed_score" not in h:
        # Ideas from before the watcher recorded it: a title-derived idea *is*
        # the notebook title, so the claim can still be read from it.
        claimed = claimed_lb_score(str(h.get("idea") or ""))
    return {
        "kernel": {
            "ref": ref,
            "url": f"https://www.kaggle.com/code/{ref}" if ref else None,
            "votes": h.get("kernel_votes"),
            "claimed_score": claimed,
            "claim_source": h.get("claim_source") or ("title" if claimed is not None else None),
            "claim_snippet": h.get("claim_snippet") or "",
        }
    }


def _our_best_lb(subs: dict[str, Any], direction: str) -> dict[str, Any] | None:
    """Our best scored public-LB submission — the like-for-like yardstick for claims."""
    scored = []
    for s in subs.get("items") or []:
        try:
            scored.append((float(s.get("public") or ""), s))
        except ValueError:
            continue
    if not scored:
        return None
    pick = max if direction.startswith("max") else min
    value, s = pick(scored, key=lambda x: x[0])
    return {"value": value, "date": s.get("date"), "ref": s.get("ref")}


_READY_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def _run_config_view(d: Path) -> dict[str, Any]:
    from researchforge.kaggle.run_config import config_view

    try:
        return config_view(d)
    except Exception as exc:  # a hand-edited file must not blank the dashboard
        return {"error": f"{type(exc).__name__}: {exc}"}


def _auto_run_view(d: Path, slug: str) -> dict[str, Any]:
    """The project's driver (with setup steps for each choice) and auto-run state."""
    from researchforge.kaggle import auto_run as ar

    try:
        cfg = ar.AutoRunConfig.load(d)
        drivers: dict[str, Any] = {}
        for name in ar.DRIVERS:
            key = (name, cfg.cli_path if name == cfg.driver else "")
            hit = _READY_CACHE.get(key)
            if hit is None or time.time() - hit[0] > 300:  # sign-in checks spawn a process
                r = ar.readiness(ar.AutoRunConfig(driver=name, cli_path=key[1]))
                info = {"ready": r.ready, "detail": r.detail, "fix": r.fix}
                hit = (time.time(), {**info, "steps": ar.setup_steps(name)})
                _READY_CACHE[key] = hit
            drivers[name] = hit[1]
        state = ar.load_state(d)
        # Only what setup recorded: this dashboard's own process is not "you".
        suggested = ar.suggested_driver(d, env={})
        return {
            "enabled": _sensors.is_enabled(_sensors.load_config(d, slug), ar.SENSOR),
            "config": asdict(cfg),
            "drivers": drivers,
            "readiness": drivers[cfg.driver],
            "suggested": {"driver": suggested[0], "reason": suggested[1]},
            "gates": ar.gates(d, cfg),
            "rounds_today": ar.rounds_today(d),
            "running": ar.current_launch(d),
            "last": state.get("last"),
            "env_file": str(ar.env_file()),
            "log": str(d / ar.LOG_FILE),
        }
    except Exception as exc:  # the view must never take the dashboard down
        return {"error": f"{type(exc).__name__}: {exc}"}


def _sensor_rows(
    d: Path, slug: str, watcher: dict[str, Any], monitor_state: dict[str, Any]
) -> dict[str, Any]:
    try:
        cfg = _sensors.load_config(d, slug)
        cfg_error = ""
    except Exception as exc:
        cfg, cfg_error = None, str(exc)
    runtime = (
        monitor_state.get("sensors") if isinstance(monitor_state.get("sensors"), dict) else None
    )
    legacy = runtime is None and bool(monitor_state)
    rows = []
    for b in _sensors.BUILTIN_SENSORS:
        name = b["name"]
        enabled = _sensors.is_enabled(cfg, name) if cfg else name not in _sensors.DEFAULT_OFF
        rt = (runtime or {}).get(name, {})
        if legacy:
            # Pre-sensor monitor_state: only timestamps exist, never a status.
            last = {
                "kernel_watcher_sensor": watcher.get("last_checked_at"),
                "arxiv_scanner_sensor": monitor_state.get("last_arxiv_at"),
                "morning_briefing_sensor": monitor_state.get("last_briefing_date"),
                "hypothesis_pool_sensor": None,
            }.get(name)
            rt = {
                "last_run_at": last,
                "status": "unknown" if last else "never",
                "detail": "recorded by an older monitor — no per-sensor status" if last else "",
            }
        if enabled and rt.get("status") == "disabled":
            # Turned on since the monitor last looked; the old "disabled" is stale.
            rt = {**rt, "status": "pending", "detail": "turned on — runs on the next monitor cycle"}
        rows.append(
            {
                "name": name,
                "kind": b["kind"],
                "description": b["description"],
                "custom": False,
                "enabled": enabled,
                "status": rt.get("status") or "never",
                "last_run_at": _norm_ts(rt.get("last_run_at")),
                "last_ok_at": _norm_ts(rt.get("last_ok_at")),
                "detail": rt.get("detail", ""),
                "error": rt.get("error", ""),
            }
        )
    for c in cfg.custom_sensors if cfg else []:
        rt = (runtime or {}).get(c.get("name"), {})
        rows.append(
            {
                "name": c.get("name"),
                "kind": "Custom sensor",
                "description": c.get("command", ""),
                "custom": True,
                "enabled": c.get("enabled", True),
                "status": rt.get("status") or "never",
                "last_run_at": _norm_ts(rt.get("last_run_at")),
                "last_ok_at": _norm_ts(rt.get("last_ok_at")),
                "detail": rt.get("detail", ""),
                "error": rt.get("error", ""),
            }
        )
    pid = monitor_state.get("monitor_pid")
    return {
        "has_state": bool(monitor_state),
        "legacy_state": legacy,
        "config_file": (d / "scheduler_config.json").exists(),
        "config_error": cfg_error,
        "last_cycle_at": _norm_ts(monitor_state.get("last_cycle_at")),
        "next_cycle_at": _norm_ts(monitor_state.get("next_cycle_at")),
        "poll_hours": monitor_state.get("poll_hours"),
        "monitor_pid": pid,
        "monitor_alive": bool(pid) and _pid_alive(pid),
        "items": rows,
    }


def _read_tokens(d: Path, issues: list[Any]) -> list[dict[str, Any]]:
    path = d / "token_usage.jsonl"
    if not path.exists():
        return []
    calls = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                calls.append(json.loads(line))
            except ValueError:
                continue
    except OSError as exc:
        _issue(issues, "warn", f"cannot read token_usage.jsonl: {exc}")
    return calls


def _read_overnight(d: Path, slug: str, sibling_setups: list[str]) -> dict[str, Any]:
    issues: list[Any] = []
    role = location_role(d)
    setup = read_json(d / "overnight_setup.json", None)
    if setup is None and (d / "overnight_setup.json").exists():
        _issue(issues, "error", "overnight_setup.json is unreadable")
    setup = setup if isinstance(setup, dict) else {}
    meta = read_json(d / "competition_metadata.json", None)
    if not isinstance(meta, dict):
        meta = (
            setup.get("competition_metadata")
            if isinstance(setup.get("competition_metadata"), dict)
            else {}
        )
    raw_bl = setup.get("baseline_result")
    bl: dict[str, Any] = raw_bl if isinstance(raw_bl, dict) else {}
    baseline = bl.get("cv_score")
    baseline = baseline if isinstance(baseline, (int, float)) and baseline not in (0, 0.0) else None
    # The loop judges rounds by the setup contract; show the same direction.
    contract_metric = ((setup.get("contract") or {}).get("objective") or {}).get(
        "primary_metric"
    ) or {}
    direction = str(
        contract_metric.get("direction") or meta.get("metric_direction") or "maximize"
    ).lower()

    rows, run_meta = _read_overnight_experiments(d / "experiment_log.db", issues)
    raw_outputs = read_json(d / "round_outputs.json", {}) or {}
    kept = {
        str(k)
        for k, v in (raw_outputs.items() if isinstance(raw_outputs, dict) else [])
        if isinstance(v, dict) and v.get("submission_file")  # as kaggle.submission.submittable
    }
    experiments = []
    prior_best: float | None = None
    for r in rows:  # in round order
        e = _overnight_experiment(
            r, baseline, direction, prior_best, kept_output=str(r["round_num"]) in kept
        )
        experiments.append(e)
        m = e["metric"]
        if isinstance(m, (int, float)) and (prior_best is None or _beats(m, prior_best, direction)):
            prior_best = float(m)
    from researchforge.kaggle.auto_run import round_drivers

    drivers = round_drivers(d)
    from researchforge.kaggle.submission import (
        leaderboard_by_round,
        submittable,
        submitted_rounds,
    )

    sent, lb = submitted_rounds(d), leaderboard_by_round(d)
    for e in experiments:
        ran = drivers.get(str(e["num"]), {})
        e["driver"] = ran.get("driver", "")
        e["auto_run"] = bool(ran.get("auto_run"))
        # Whether the Submit button can send it, and whether it already went.
        e["submittable"] = submittable(d, int(e["num"]))
        e["submitted"] = int(e["num"]) in sent or int(e["num"]) in lb
        e["lb"] = lb.get(int(e["num"]))  # what Kaggle scored it, if it was submitted
    # Archived rounds leave every view, the DAG and "best"; Runs lists them apart.
    from researchforge.kaggle.experiment_loop import archived_rounds

    hidden = archived_rounds(d)
    archived = [e for e in experiments if e["num"] in hidden]
    experiments = [e for e in experiments if e["num"] not in hidden]
    # The repo engine's experiments, folded into this one history (see history_import).
    from researchforge.kaggle.history_import import imported_experiments

    experiments += [_imported_experiment(r, direction) for r in imported_experiments(d)]

    # In-flight rounds (execute-round / run-parallel publish these while running)
    inflight = read_json(d / "inflight_rounds.json", {}) or {}
    lock = read_json(d / "execute_round.lock", None)
    if isinstance(lock, dict) and lock.get("round_num") and str(lock["round_num"]) not in inflight:
        inflight[str(lock["round_num"])] = lock
    elif isinstance(lock, dict) and not lock.get("round_num"):
        inflight.setdefault("lock", {**lock, "title": "execute-round starting"})
    known = {e["num"] for e in experiments}
    for _key, info in inflight.items():
        if not isinstance(info, dict):
            continue
        num = info.get("round_num")
        if num in known:
            continue
        pid = info.get("pid")
        alive = _pid_alive(pid) if pid else None
        parent = info.get("parent_round_num")
        experiments.append(
            {
                "id": f"R{num}" if num else "R?",
                "num": num or 10**6,
                "source": "overnight",
                "title": info.get("title") or "(round starting)",
                "parents": [f"R{parent}"] if parent else [],
                "status": "running" if alive is not False else "stale",
                "stage": info.get("stage"),
                "metric": None,
                "metric_direction": direction,
                "baseline": baseline,
                "branch": info.get("branch_tag") or "",
                "credit": "",
                "hypothesis_ids": [],
                "reason": ""
                if alive is not False
                else f"process {pid} is gone but its lock remains",
                "started_at": _norm_ts(info.get("started_at")),
                "completed_at": None,
                "duration_s": None,
            }
        )
        if alive is False:
            _issue(
                issues,
                "warn",
                f"round {num or '?'} lock is held by dead PID {pid}",
                f"delete {d / 'execute_round.lock'} if that run crashed",
            )

    # execute-round archives each proposal as proposal_round_<n>.json: it holds
    # the description, rationale and expected impact behind the patch.
    for e in experiments:
        prop = read_json(d / f"proposal_round_{e['num']}.json", None)
        if isinstance(prop, dict):
            e["proposal"] = {
                k: str(prop.get(k) or "")
                for k in ("description", "rationale", "expected_impact", "hypothesis")
            }
            if not e.get("patch") and prop.get("code_changes"):
                code = str(prop["code_changes"])
                e["patch"], e["patch_truncated"] = code[:PATCH_CHARS], len(code) > PATCH_CHARS

    try:
        merge_candidate = _overnight_merge_candidate(rows)
    except Exception as exc:  # the dashboard must still render
        merge_candidate = None
        _issue(issues, "warn", f"merge detection failed: {type(exc).__name__}: {exc}")

    # The autonomous loop writes questions here when it needs a human answer
    # mid-run; it counts as waiting until user_answers.json is saved after it.
    waiting_input = None
    q_path = d / "mid_loop_questions.json"
    q_m, a_m = _mtime(q_path), _mtime(d / "user_answers.json")
    if q_m and (not a_m or a_m < q_m):
        qs = read_json(q_path, []) or []
        waiting_input = {
            "written_at": _iso(q_m),
            "questions": [str(q.get("text") or "") for q in qs if isinstance(q, dict)],
        }
    # Non-blocking interactions (kaggle respond answers them); the loop keeps running.
    from researchforge.kaggle.control import LoopControl, open_interactions

    try:
        control = asdict(LoopControl.load(d))
        open_qs = open_interactions(d)
    except Exception as exc:  # a hand-edited file must not blank the dashboard
        control, open_qs = None, []
        _issue(issues, "warn", f"loop control unreadable: {type(exc).__name__}: {exc}")
    if open_qs:
        waiting_input = {
            "written_at": open_qs[0].created_at,
            "questions": [q.question for q in open_qs],
            "ids": [q.id for q in open_qs],
        }

    pool_raw = read_json(d / "pending_hypotheses.json", None)
    if pool_raw is None and (d / "pending_hypotheses.json").exists():
        _issue(issues, "error", "pending_hypotheses.json is unreadable (partial write?)")
    pool = pool_raw if isinstance(pool_raw, list) else []
    hypotheses = [_hypothesis_from_pool(i, h) for i, h in enumerate(pool) if isinstance(h, dict)]
    from researchforge.kaggle.hypothesis_pool import idea_key, pin_id, pinned

    pins = {idea_key(str(h["idea"])): h["pin"] for h in pinned(d)}
    from researchforge.kaggle.hypothesis_pool import below_our_lb

    below = below_our_lb(d, direction)
    for h, raw in zip(hypotheses, (x for x in pool if isinstance(x, dict)), strict=True):
        idea = str(raw.get("idea") or "")
        h["pin_id"] = pin_id(idea) if idea else ""
        h["pin"] = pins.get(idea_key(idea))
        # Left out of the queue: a notebook claiming less than our best LB (still pinnable).
        h["below_lb"] = idea in below and h["pin"] is None
    by_round = {e["id"]: e for e in experiments}
    for h in hypotheses:
        for rid in h["tested_by"]:
            if rid in by_round:
                by_round[rid]["hypothesis_ids"].append(h["id"])
    # What your directions and pinned ideas led to: the rounds and the ideas they tested.
    from researchforge.kaggle.steering import steered_rounds

    steered = steered_rounds(d)
    for e in experiments:
        e["steered"] = steered.get(int(e.get("num") or 0), [])
    for h in hypotheses:
        h["steered"] = [
            g
            for rid in h["tested_by"]
            for g in by_round.get(rid, {}).get("steered", [])
            if g.get("how") == "followed"
        ]

    proposal = read_json(d / "proposal.json", None)
    pending_proposal = None
    if isinstance(proposal, dict):
        pending_proposal = {
            "title": proposal.get("title", ""),
            "branch_tag": proposal.get("branch_tag", ""),
            "parent_round_num": proposal.get("parent_round_num"),
            "hypothesis": proposal.get("hypothesis", ""),
            "written_at": _iso(_mtime(d / "proposal.json")),
        }

    ctx = read_json(d / "pending_round_context.json", {}) or {}
    watcher = read_json(d / "watcher_config.json", {}) or {}
    monitor_state = read_json(d / "monitor_state.json", {}) or {}

    from researchforge.kaggle import steering
    from researchforge.kaggle.new_project import load_draft

    draft = load_draft(d)
    if not setup and draft:
        _issue(
            issues,
            "info",
            "setup is not finished yet",
            "Finish it under New competition on this dashboard, or with /overnight "
            f"{slug} in Claude Code or Cursor",
        )
    elif not setup:
        msg = "no overnight_setup.json — this directory only holds watcher/monitor state"
        hint = ""
        if sibling_setups:
            hint = f"The set-up project for {slug} is at {sibling_setups[0]}"
        _issue(issues, "warn" if sibling_setups else "info", msg, hint)
    if not pool and (d / "pending_hypotheses.json").exists() is False:
        _issue(
            issues,
            "info",
            "no pending_hypotheses.json yet",
            f"ideas arrive from the monitor's kernel watcher, arXiv scans, or "
            f"researchforge kaggle propose-hypothesis {slug}",
        )

    files = {}
    for name in (
        "overnight_setup.json",
        "competition_metadata.json",
        "experiment_log.db",
        "pending_hypotheses.json",
        "watcher_config.json",
        "scheduler_config.json",
        "monitor_state.json",
        "idea_sources.json",
        "submissions.json",
        "token_usage.jsonl",
        "inflight_rounds.json",
        "execute_round.lock",
        "proposal.json",
        "pending_round_context.json",
        "scheduler_launchd.log",
        "mid_loop_questions.json",
    ):
        m = _mtime(d / name)
        files[name] = _iso(m)

    activity = (
        [m for m in (_mtime(p) for p in d.iterdir() if p.is_file()) if m] if d.is_dir() else []
    )
    intel = d / "intelligence"
    if intel.is_dir():
        activity += [m for m in (_mtime(p) for p in intel.iterdir()) if m]

    return {
        "artifacts_dir": str(d),
        "slug": slug,
        "role": role,
        "has_setup": bool(setup),
        "competition": {
            "title": meta.get("title") or slug,
            "metric": meta.get("metric_name") or "",
            "direction": direction,
            "direction_source": meta.get("metric_direction_source") or "",
            "deadline": _norm_ts(meta.get("deadline")),
            "is_code_competition": bool(meta.get("is_code_competition")),
            "url": f"https://www.kaggle.com/competitions/{slug}",
        },
        "setup": {
            "baseline_cv": baseline,
            "baseline_kernel": bl.get("kernel_ref") or setup.get("baseline_kernel_ref") or "",
            "gpu": setup.get("selected_gpu") or "",
            "time_budget_hours": setup.get("time_budget_hours"),
            "max_experiments": setup.get("max_experiments"),
            "ai_provider_mode": setup.get("ai_provider_mode") or "",
            "ai_model": setup.get("ai_model") or "",
            "set_up_at": files["overnight_setup.json"],
        },
        "run_metadata": run_meta,
        "experiments": experiments,
        "archived_experiments": archived,
        "hypotheses": hypotheses,
        "kernel_board": _kernel_board_view(d),
        "pending_proposal": pending_proposal,
        "merge_candidate": merge_candidate,
        "waiting_input": waiting_input,
        "control": control,
        "round_context": {
            "round_num": ctx.get("round_num"),
            "recommended_next": ctx.get("recommended_next"),
            "experiments_remaining": ctx.get("experiments_remaining"),
            "resynthesize_due": ctx.get("resynthesize_due"),
            "written_at": files["pending_round_context.json"],
        }
        if ctx
        else None,
        "watcher": {
            "configured": bool(watcher),
            "enabled": watcher.get("enabled"),
            "interval_hours": watcher.get("interval_hours"),
            "last_checked_at": _norm_ts(watcher.get("last_checked_at")),
            "seen_kernels": len(watcher.get("seen_kernel_refs", []) or []),
            "seen_submissions": len(watcher.get("seen_submission_ids", []) or []),
        },
        "sensors": _sensor_rows(d, slug, watcher, monitor_state),
        "draft": draft,
        "steering": [asdict(x) for x in steering.load(d)],
        "up_next": steering.up_next(d) if setup else [],
        "engine_next": _engine_next(d, direction, limit=10) if setup else [],
        "auto_run": _auto_run_view(d, slug),
        "run_config": _run_config_view(d),
        "submissions_snapshot": read_json(d / "submissions.json", None),
        "submissions_daily": _daily_allowance(d),
        "token_calls": _read_tokens(d, issues),
        "files": files,
        "last_activity_ts": max(activity) if activity else None,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Repo project (.researchforge/)
# ---------------------------------------------------------------------------


def _metric_of(execution: dict[str, Any]) -> float | None:
    pm = (execution.get("metrics") or {}).get("primary_metric") or {}
    v = pm.get("value")
    return v if isinstance(v, (int, float)) else None


def _read_repo(root: Path, repo_dir: Path) -> dict[str, Any]:
    issues: list[Any] = []
    out: dict[str, Any] = {
        "db_path": str(repo_dir / "researchforge.db"),
        "experiments": [],
        "hypotheses": [],
        "runs": [],
        "baselines": [],
        "deliverables": [],
        "ai_calls": [],
        "issues": issues,
        "merge_candidates": [],
        "project": None,
        "next_action": "",
        "run_in_progress": False,
        "autorun": None,
        "last_activity_ts": None,
        "research_log": None,
    }
    act = [
        m
        for m in (
            _mtime(repo_dir / n)
            for n in ("researchforge.db", "researchforge.db-wal", "autorun.json", "research-log.md")
        )
        if m
    ]
    for sub in ("synthesis", "experiments"):
        if (repo_dir / sub).is_dir():
            act += [m for m in (_mtime(p) for p in (repo_dir / sub).iterdir()) if m]
    out["last_activity_ts"] = max(act) if act else None

    auto = read_json(repo_dir / "autorun.json", None)
    if isinstance(auto, dict):
        out["autorun"] = {
            k: auto.get(k)
            for k in (
                "status",
                "started_at",
                "updated_at",
                "rounds_completed",
                "total_experiments",
                "best_experiment_id",
                "best_metric_value",
                "stop_reason",
                "objective_achieved",
            )
        }
        out["autorun"]["updated_at"] = _norm_ts(auto.get("updated_at"))
    # Unattended auto-run (researchforge auto-run): settings, gates, launches, service.
    try:
        from researchforge.autorun.unattended import status as _unattended_status

        out["auto_run"] = _unattended_status(root)
    except Exception as exc:  # shown on the card, never fatal to the dashboard
        out["auto_run"] = {"error": f"{type(exc).__name__}: {exc}"}
    log = repo_dir / "research-log.md"
    if log.exists():
        try:
            heads = [
                ln[3:].strip()
                for ln in log.read_text(encoding="utf-8").splitlines()
                if ln.startswith("## ")
            ]
            out["research_log"] = {
                "path": str(log),
                "updated_at": _iso(_mtime(log)),
                "last_heading": heads[-1] if heads else "",
            }
        except OSError:
            pass

    if not (repo_dir / "researchforge.db").exists():
        _issue(
            issues,
            "warn",
            ".researchforge/ exists but has no researchforge.db",
            "run `researchforge init` in this folder",
        )
        return out
    try:
        from researchforge.server.data import read_state

        st = read_state(root)
    except Exception as exc:
        _issue(issues, "error", f"cannot read project DB: {type(exc).__name__}: {exc}")
        return out

    def dump(m: Any) -> dict[str, Any]:
        data: dict[str, Any] = m.model_dump(mode="json")
        return data

    project = dump(st.project)
    out["project"] = {
        "name": project.get("name"),
        "mode": project.get("mode"),
        "status": project.get("status"),
        "objective": project.get("objective"),
        "created_at": _norm_ts(project.get("created_at")),
        "updated_at": _norm_ts(project.get("updated_at")),
    }
    out["next_action"] = st.next_action
    out["run_in_progress"] = st.run_in_progress

    baseline = None
    if st.baseline is not None:
        b = dump(st.baseline)
        baseline = _metric_of(b)
    contract = dump(st.contract) if st.contract else {}
    obj = ((contract.get("spec") or {}).get("objective") or {}) if contract else {}
    direction = str(
        (obj.get("primary_metric") or {}).get("direction") or obj.get("direction") or "maximize"
    )

    execs: dict[str, list[dict[str, Any]]] = {}
    for ex in st.executions:
        e = dump(ex)
        execs.setdefault(e["experiment_id"], []).append(e)
    rank = {"validation": 3, "full": 2, "screening": 1}
    for exp in st.experiments:
        e = dump(exp)
        runs = sorted(
            execs.get(e["experiment_id"], []),
            key=lambda x: (rank.get(str(x.get("benchmark_stage")), 0), x.get("started_at") or ""),
        )
        succeeded = [
            x for x in runs if x.get("status") == "succeeded" and _metric_of(x) is not None
        ]
        best_exec = succeeded[-1] if succeeded else (runs[-1] if runs else None)
        status = e.get("status") or "planned"
        ui = {
            "promising": "improved",
            "validated": "improved",
            "implementation_ready": "improved",
            "rejected": "failed",
            "failed_setup": "failed",
            "failed_execution": "failed",
            "running": "running",
            "preparing": "running",
            "validating": "running",
            "planned": "planned",
            "approved": "planned",
            "cancelled": "cancelled",
        }.get(status, status)
        decision = e.get("decision") or {}
        parents = list(e.get("parent_experiment_ids") or [])
        if not parents and e.get("parent_experiment_id"):
            parents = [e["parent_experiment_id"]]
        out["experiments"].append(
            {
                "id": e["experiment_id"],
                "num": int(re.sub(r"\D", "", e["experiment_id"]) or 0),
                "source": "repo",
                "title": e.get("title") or e["experiment_id"],
                "parents": parents,
                "status": ui,
                "raw_status": status,
                "stage": best_exec.get("benchmark_stage") if best_exec else None,
                "metric": _metric_of(best_exec) if best_exec else None,
                "metric_direction": direction,
                "baseline": baseline,
                "branch": e.get("plan_id") or "",
                "credit": "",
                "is_merge": bool(e.get("is_merge")),
                "reason": decision.get("reason") or (best_exec or {}).get("failure_reason") or "",
                "started_at": _norm_ts(
                    (runs[0] if runs else {}).get("started_at") or e.get("created_at")
                ),
                "completed_at": _norm_ts((best_exec or {}).get("completed_at")),
                "duration_s": sum((x.get("duration_seconds") or 0) for x in runs) or None,
                "hypothesis_ids": [e["hypothesis_id"]] if e.get("hypothesis_id") else [],
                "executions": [
                    {
                        "stage": x.get("benchmark_stage"),
                        "status": x.get("status"),
                        "metric": _metric_of(x),
                        "started_at": _norm_ts(x.get("started_at")),
                        "completed_at": _norm_ts(x.get("completed_at")),
                    }
                    for x in runs
                ],
                "patch": (e.get("patch_text") or "")[:PATCH_CHARS],
                "patch_truncated": len(e.get("patch_text") or "") > PATCH_CHARS,
                "proposal": {
                    "description": e.get("change_summary") or "",
                    "changed_files": e.get("changed_files") or [],
                    "observation": e.get("observation") or "",
                },
            }
        )

    from researchforge.autorun.merges import find_merge_opportunities

    out["merge_candidates"] = [
        {
            "parents": [m["parent_a"], m["parent_b"]],
            "command": m["command"],
            "reason": "independent improvements: different hypotheses, disjoint files and lineages",
        }
        for m in find_merge_opportunities([dump(x) for x in st.experiments])
    ]

    imported_at: dict[str, str] = {}
    try:  # the domain model drops imported_at; read it directly (read-only)
        with _ro_connect(repo_dir / "researchforge.db") as conn:
            imported_at = {
                r["hypothesis_id"]: r["imported_at"]
                for r in conn.execute("SELECT hypothesis_id, imported_at FROM hypotheses")
            }
    except sqlite3.Error:
        pass

    tested: dict[str, list[str]] = {}
    for e in out["experiments"]:
        for hid in e["hypothesis_ids"]:
            tested.setdefault(hid, []).append(e["id"])
    db_ids = set()
    for hyp in st.hypotheses:
        h = dump(hyp)
        db_ids.add(h["hypothesis_id"])
        out["hypotheses"].append(
            {
                "id": h["hypothesis_id"],
                "source": "repo",
                "origin": "imported",
                "title": h.get("title") or h["hypothesis_id"],
                "detail": h.get("claim") or h.get("rationale") or "",
                "status": h.get("status") or "",
                "tested": bool(tested.get(h["hypothesis_id"])),
                "priority": None,
                "confidence": h.get("novelty_confidence") or "",
                "feasibility": h.get("feasibility") or "",
                "evidence": h.get("evidence_status") or "",
                "created_at": _norm_ts(imported_at.get(h["hypothesis_id"])),
                "refs": [],
                "tested_by": tested.get(h["hypothesis_id"], []),
                "outcome": "",
            }
        )
    # Drafts Claude wrote but has not imported yet — real files, clearly labelled.
    draft_path = repo_dir / "synthesis" / "hypotheses.yaml"
    if draft_path.exists():
        try:
            import yaml

            raw = yaml.safe_load(draft_path.read_text(encoding="utf-8")) or {}
            items = raw.get("hypotheses", []) if isinstance(raw, dict) else raw
            for h in items or []:
                if isinstance(h, dict) and h.get("hypothesis_id") not in db_ids:
                    out["hypotheses"].append(
                        {
                            "id": h.get("hypothesis_id") or "draft",
                            "source": "repo",
                            "origin": "draft (not imported)",
                            "title": h.get("title") or "(untitled draft)",
                            "detail": h.get("claim") or "",
                            "status": "draft",
                            "tested": False,
                            "priority": None,
                            "confidence": "",
                            "created_at": _iso(_mtime(draft_path)),
                            "refs": [],
                            "tested_by": [],
                            "outcome": "",
                        }
                    )
        except Exception as exc:
            _issue(issues, "warn", f"cannot parse synthesis/hypotheses.yaml: {exc}")

    out["runs"] = [
        {
            "id": r["run_id"],
            "plan_id": r.get("plan_id"),
            "status": r.get("status"),
            "started_at": _norm_ts(r.get("started_at")),
            "completed_at": _norm_ts(r.get("completed_at")),
        }
        for r in (dump(x) for x in st.runs)
    ]
    out["baselines"] = (
        [{"metric": baseline, "at": _norm_ts(dump(st.baseline).get("completed_at"))}]
        if st.baseline
        else []
    )
    out["deliverables"] = [
        {
            "kind": d.get("kind"),
            "at": _norm_ts(d.get("created_at")),
            "path": d.get("path") or d.get("output_path") or "",
        }
        for d in (dump(x) for x in st.deliverables)
    ]
    out["ai_calls"] = [
        {
            "ts": _norm_ts(c.get("created_at")),
            "purpose": c.get("purpose"),
            "provider": c.get("provider"),
            "model": c.get("model"),
            "input_tokens": (c.get("usage") or {}).get("input_tokens", c.get("input_tokens", 0)),
            "output_tokens": (c.get("usage") or {}).get("output_tokens", c.get("output_tokens", 0)),
            "estimated": c.get("estimated", False),
        }
        for c in (_dump_any(x) for x in st.ai_calls)
    ]
    if not out["experiments"] and not out["hypotheses"]:
        _issue(issues, "info", "project has no hypotheses or experiments yet", st.next_action)
    return out


def _dump_any(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        data: dict[str, Any] = obj.model_dump(mode="json")
        return data
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        d = asdict(obj)
        d["created_at"] = getattr(obj, "created_at", None)
        return d
    return dict(obj)


# ---------------------------------------------------------------------------
# Submissions, tokens, timeline
# ---------------------------------------------------------------------------


def _submissions(slug: str, snapshot: Any, live: dict[str, Any] | None) -> dict[str, Any]:
    snap_at = _norm_ts(snapshot.get("fetched_at")) if isinstance(snapshot, dict) else None
    live_at = live.get("fetched_at") if live else None
    use_live = live is not None and (not snap_at or (live_at or "") >= snap_at)
    src = live if use_live else (snapshot if isinstance(snapshot, dict) else None)
    if src is None:
        return {"status": "pending", "source": None, "fetched_at": None, "error": "", "items": []}
    items = []
    for s in src.get("submissions", []) or []:
        items.append(
            {
                "ref": s.get("ref", ""),
                "file": s.get("filename") or s.get("filename", ""),
                "description": s.get("description", ""),
                "date": _norm_ts(s.get("date")),
                "status": (s.get("status") or "").replace("SubmissionStatus.", "").lower(),
                "public": s.get("publicscore") or s.get("public_score") or "",
                "private": s.get("privatescore") or s.get("private_score") or "",
                "error": s.get("errordescription") or s.get("error_description") or "",
            }
        )
    return {
        "status": "error" if src.get("error") else "ok",
        "source": "dashboard" if use_live else "monitor snapshot",
        "fetched_at": _norm_ts(src.get("fetched_at")),
        "error": src.get("error", ""),
        "items": items,
    }


# Public list prices (USD / 1M tokens). Shown in the UI as an estimate, editable
# via ~/.config/researchforge/dashboard.json {"token_prices": {"<model>": [in, out]}}.
_DEFAULT_PRICES = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.8, 4.0),
    "gemini": (1.25, 10.0),
    "gpt-4o": (2.5, 10.0),
    "gpt": (2.5, 10.0),
}


def _tokens(calls: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    cfg = read_json(config_dir() / "dashboard.json", {}) or {}
    prices = {**{k: list(v) for k, v in _DEFAULT_PRICES.items()}, **(cfg.get("token_prices") or {})}

    def price(model: str) -> Any:
        m = (model or "").lower()
        best = max((k for k in prices if k in m), key=len, default=None)
        return prices[best] if best else None

    by_model: dict[str, dict[str, Any]] = {}
    by_purpose: dict[str, int] = {}
    by_round: dict[str, int] = {}  # proposal tokens per overnight round (autonomous mode)
    tin = tout = cached = 0
    cost = reported = 0.0
    unpriced = False
    for c in calls:
        i, o = int(c.get("input_tokens") or 0), int(c.get("output_tokens") or 0)
        cr = int(c.get("cache_read_tokens") or 0)
        tin, tout, cached = tin + i, tout + o, cached + cr
        m = by_model.setdefault(
            c.get("model") or "unknown",
            {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cost_usd": 0.0},
        )
        m["calls"] += 1
        m["input"] += i
        m["output"] += o
        m["cache_read"] += cr
        by_purpose[c.get("purpose") or "other"] = (
            by_purpose.get(c.get("purpose") or "other", 0) + i + o
        )
        rm = re.fullmatch(r"proposal_round_(\d+)", c.get("purpose") or "")
        if rm:
            by_round[f"R{rm.group(1)}"] = by_round.get(f"R{rm.group(1)}", 0) + i + o
        # A cost the provider reported (Claude Code does) beats list prices, which
        # know nothing of cache discounts. Local Ollama models cost compute, not tokens.
        if isinstance(c.get("cost_usd"), (int, float)):
            one = float(c["cost_usd"])
            reported += one
        else:
            p = [0.0, 0.0] if c.get("provider") == "ollama" else price(c.get("model") or "")
            one = i / 1e6 * p[0] + o / 1e6 * p[1] if p is not None else 0.0
            unpriced = unpriced or (p is None and bool(i or o))
        cost += one
        m["cost_usd"] = round(m["cost_usd"] + one, 4)
    return {
        "calls": len(calls),
        "input_tokens": tin,
        "output_tokens": tout,
        "cache_read_tokens": cached,
        "estimated_cost_usd": round(cost, 4) if calls else None,
        "reported_cost_usd": round(reported, 4),
        "has_unpriced": unpriced,
        "by_model": by_model,
        "by_purpose": by_purpose,
        "by_round": by_round,
        "unobservable": mode == "claude_code",
        "note": (
            "Auto-run launches and sensor calls are metered. Rounds you drive from your own "
            "Claude Code chat are not: those tokens never pass through ResearchForge."
            if mode == "claude_code"
            else ""
        ),
    }


def _timeline(p: dict[str, Any], notifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ev: list[dict[str, Any]] = []

    def add(
        ts: Any, kind: Any, title: Any, detail: Any = "", priority: Any = "normal", ref: Any = None
    ) -> Any:
        ts = _norm_ts(ts)
        if ts:
            ev.append(
                {
                    "ts": ts,
                    "kind": kind,
                    "title": title,
                    "detail": detail,
                    "priority": priority,
                    "ref": ref,
                }
            )

    o = p.get("overnight")
    if o:
        add(
            o["setup"]["set_up_at"],
            "setup",
            f"Overnight set up: {o['slug']}",
            f"baseline CV {o['setup']['baseline_cv']}" if o["setup"]["baseline_cv"] else "",
        )
        if o["round_context"]:
            add(
                o["round_context"]["written_at"],
                "claude",
                f"prepare-round → round {o['round_context']['round_num']}",
                f"recommended: {o['round_context']['recommended_next'] or '—'}",
            )
        if o["pending_proposal"]:
            add(
                o["pending_proposal"]["written_at"],
                "claude",
                f"Proposal written: {o['pending_proposal']['title']}",
                "waiting for execute-round",
                "high",
            )
        w = o["watcher"]
        if w["last_checked_at"]:
            add(
                w["last_checked_at"],
                "sensor",
                "Kernel watcher checked",
                f"{w['seen_kernels']} kernels seen",
            )
        for s in o["sensors"]["items"]:
            if s["last_run_at"] and s["status"] in ("ok", "error"):
                add(
                    s["last_run_at"],
                    "sensor",
                    f"{s['name']}: {s['status']}",
                    s["error"] or s["detail"],
                    "high" if s["status"] == "error" else "normal",
                )
        for h in p["hypotheses"]:
            if h["source"] == "overnight":
                add(
                    h["created_at"],
                    "hypothesis",
                    f"Hypothesis added ({h['origin']})",
                    h["title"][:120],
                    ref=h["id"],
                )
    for e in p["experiments"]:
        res = e["status"]
        metric = f"metric {e['metric']:.4f}" if isinstance(e["metric"], (int, float)) else ""
        add(
            e["started_at"],
            "experiment",
            f"{e['id']} started: {e['title'][:70]}",
            e["branch"],
            ref=e["id"],
        )
        if e["completed_at"]:
            add(
                e["completed_at"],
                "experiment",
                f"{e['id']} {res}",
                metric or e["reason"][:120],
                "high" if res == "improved" else "normal",
                ref=e["id"],
            )
    for s in p["submissions"]["items"]:
        done = s["status"] == "complete"
        add(
            s["date"],
            "submission",
            f"Submission {s['ref']}: {s['file']}",
            f"public {s['public']}" if s["public"] else s["status"],
            "high" if done else "normal",
        )
    r = p.get("repo")
    if r:
        if r["project"]:
            add(
                r["project"]["created_at"],
                "setup",
                f"Project created: {r['project']['name']}",
                r["project"]["mode"] or "",
            )
        for b in r["baselines"]:
            add(
                b["at"],
                "baseline",
                "Baseline measured",
                f"metric {b['metric']}" if b["metric"] is not None else "",
            )
        for run in r["runs"]:
            add(run["started_at"], "run", f"{run['id']} started ({run['plan_id']})", run["status"])
        for d in r["deliverables"]:
            add(d["at"], "ship", f"Deliverable: {d['kind']}", d["path"])
        if r["research_log"]:
            add(
                r["research_log"]["updated_at"],
                "claude",
                "Research log updated",
                r["research_log"]["last_heading"],
            )
        if r["autorun"]:
            add(
                r["autorun"]["updated_at"],
                "run",
                f"Autorun {r['autorun']['status']}",
                r["autorun"].get("stop_reason") or "",
            )
        for h in p["hypotheses"]:
            if h["source"] == "repo":
                add(
                    h["created_at"],
                    "hypothesis",
                    "Draft hypothesis written" if h["status"] == "draft" else "Hypothesis imported",
                    h["title"][:120],
                    ref=h["id"],
                )
        for e in p["experiments"]:
            for x in e.get("executions") or []:
                if x["stage"] == "validation":
                    add(
                        x["completed_at"] or x["started_at"],
                        "validation",
                        f"{e['id']} validation {x['status']}",
                        f"metric {x['metric']:.4f}"
                        if isinstance(x["metric"], (int, float))
                        else "",
                        ref=e["id"],
                    )
    for n in notifications:
        add(
            n.get("timestamp"),
            "notification",
            n.get("title", "Notification"),
            n.get("message", "")[:160],
            "high" if n.get("priority") == "high" else "normal",
        )
    ev.sort(key=lambda x: x["ts"], reverse=True)
    return ev[:300]


def read_notifications(limit: int = 500) -> list[dict[str, Any]]:
    path = config_dir() / "notification_log.jsonl"
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            with contextlib.suppress(ValueError):
                out.append(json.loads(line))
    except OSError:
        pass
    return list(reversed(out))


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------


def _best(experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [
        e
        for e in experiments
        if isinstance(e.get("metric"), (int, float))
        and e["status"] not in ("failed", "stale")
        and not e.get("imported")  # measured by another benchmark
    ]
    if not scored:
        return None
    maximize = (scored[0].get("metric_direction") or "maximize").startswith("max")
    return (max if maximize else min)(scored, key=lambda e: e["metric"])


def _plans_awaiting_approval(repo_dir: Path) -> set[str]:
    """Plans still waiting for a typed approval.

    A ``planned`` plan older than the newest plan that went ahead was superseded
    (re-planned and run), so it is not waiting for anyone.
    """
    db = repo_dir / "researchforge.db"
    if not db.is_file():
        return set()
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT plan_id, status, created_at FROM experiment_plans"
            ).fetchall()
    except sqlite3.Error:
        return set()
    went_ahead = max((str(c) for _p, st, c in rows if st != "planned" and c), default="")
    return {str(p) for p, st, c in rows if st == "planned" and str(c or "") > went_ahead}


def _plan_summary(repo_dir: Path, plan_id: str) -> str:
    try:
        with sqlite3.connect(f"file:{repo_dir / 'researchforge.db'}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT record FROM experiment_plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
        record = json.loads(row[0]) if row else {}
    except (sqlite3.Error, ValueError, TypeError):
        return ""
    return str(record.get("approach_summary") or "")


def _questions(
    od: Path | None, repo_dir: Path | None, experiments: list[dict[str, Any]], slug: str
) -> list[dict[str, Any]]:
    """Everything waiting on the person for one project, newest first.

    Open questions from the overnight loop and the repo project (answerable from the
    dashboard), plus repo plans waiting for approval — those stay typed in a terminal,
    so they carry the command instead of answer buttons. Submission offers for rounds
    that followed the person's direction or pinned idea come first and say so.
    """
    from researchforge.kaggle.control import open_interactions
    from researchforge.kaggle.submission import round_of

    followed = {
        int(e.get("num") or 0): [g for g in e.get("steered") or [] if g.get("how") == "followed"]
        for e in experiments
        if e.get("source") != "repo"
    }
    out: list[dict[str, Any]] = []
    for where, d in (("kaggle", od), ("repo", repo_dir)):
        if d is None:
            continue
        try:
            items = open_interactions(d)
        except Exception:  # an unreadable file is reported elsewhere, never fatal here
            continue
        for q in items:
            if where == "kaggle":
                cmd = f'researchforge kaggle respond {slug} {q.id} "<answer>"'
            elif q.answer_key.startswith("judge:"):
                cmd = f"researchforge experiment judge {q.answer_key[6:]} <yes|no|skip>"
            else:
                cmd = "answer it on the dashboard"
            num = (
                round_of(q.answer_key, q.question)
                if where == "kaggle" and q.kind == "submission_approval"
                else None
            )
            out.append(
                {
                    "id": q.id,
                    "where": where,
                    "kind": q.kind,
                    "question": q.question,
                    "options": q.options,
                    "actions": q.actions,
                    "default": q.default,
                    "blocks": q.blocks,
                    "holds_loop": q.holds_loop,
                    "source": q.source,
                    "asked_at": _norm_ts(q.created_at),
                    "answerable": True,
                    "command": cmd,
                    "round": num,
                    "steered": followed.get(num, []) if num is not None else [],
                }
            )
    plans: dict[str, list[str]] = {}
    titles: dict[str, list[str]] = {}
    waiting = _plans_awaiting_approval(repo_dir) if repo_dir is not None else set()
    for e in experiments:
        branch = str(e.get("branch") or "")
        if e.get("source") == "repo" and e.get("raw_status") == "planned" and branch in waiting:
            plans.setdefault(branch, []).append(str(e["id"]))
            titles.setdefault(branch, []).append(f"{e['id']}: {e.get('title') or ''}")
    for plan_id, ids in sorted(plans.items()):
        out.append(
            {
                "id": plan_id,
                "where": "repo",
                "kind": "plan_approval",
                "question": f"Plan {plan_id} waits for your approval ({len(ids)} experiment(s))",
                "options": [],
                "default": "",
                "blocks": ids,
                "holds_loop": True,
                "source": "plan",
                "asked_at": None,
                "answerable": False,  # approved by typing "approve", here or in a terminal
                "approvable": True,
                "approach": _plan_summary(repo_dir, plan_id) if repo_dir is not None else "",
                "experiments": titles.get(plan_id, []),
                "command": f"researchforge experiment approve {plan_id}",
            }
        )
    out.sort(key=lambda q: q["asked_at"] or "", reverse=True)
    out.sort(key=lambda q: not q.get("steered"))  # stable: steered first, each part newest first
    return out


def _build_project(
    loc: ProjectLocation,
    od: Path | None,
    setups_by_slug: dict[str, list[str]],
    notifications: list[dict[str, Any]],
    live_subs: dict[str, Any],
) -> dict[str, Any]:
    root = loc.root
    o: dict[str, Any] | None = None
    if od is not None:
        od_slug = slug_of(od)
        o = _read_overnight(
            od, od_slug, [p for p in setups_by_slug.get(od_slug, []) if p != str(od)]
        )
    # A repo project shares its root with at most one overnight dir entry. Beside a
    # competition it is not a second engine: its experiments are folded into the
    # Kaggle loop's one history (history_import), so it is not read separately here.
    attach_repo = loc.repo_dir is not None and (od is None or od == loc.overnight_dirs[0])
    r = (
        _read_repo(root, loc.repo_dir)
        if attach_repo and loc.repo_dir is not None and o is None
        else None
    )

    slug = o["slug"] if o else None
    name = (
        slug
        or (r["project"]["name"] if r and r["project"] else "")
        or loc.registry_name
        or root.name
    )
    pid = project_id(od if od is not None else root)

    experiments = (o["experiments"] if o else []) + (r["experiments"] if r else [])
    hypotheses = (o["hypotheses"] if o else []) + (r["hypotheses"] if r else [])
    issues = (o["issues"] if o else []) + (r["issues"] if r else [])
    if o is not None and od is not None and attach_repo:
        from researchforge.kaggle.history_import import merge_status

        merged = merge_status(od)
        if merged and merged["pending"]:
            _issue(
                issues,
                "info",
                "the repo project's latest experiments are not in this history yet",
                f"researchforge kaggle merge-history {o['slug']} (the monitor does it on its "
                "next cycle)",
            )

    sched = None
    if slug:
        st = launchd_status(slug)
        sched = dict(st)
        sched["targets_this_dir"] = (
            od is not None
            and bool(st["target_dir"])
            and Path(st["target_dir"]).resolve() == od.resolve()
        )
        if st["installed"] and not sched["targets_this_dir"]:
            sched["note"] = f"The launchd job for {slug} writes to {st['target_dir']}"
        elif st["state"] == "failing":
            _issue(
                issues,
                "error",
                f"launchd watcher is failing (last exit {st['last_exit_status']})",
                st["last_error"] or f"see {st['log_path']}",
            )
        elif st["state"] == "installed_not_loaded":
            _issue(
                issues,
                "warn",
                "launchd plist exists but is not loaded",
                f"launchctl load -w {st['plist']}",
            )
        sched["live"] = st["state"] in ("running", "idle") and sched["targets_this_dir"]

    subs = (
        _submissions(slug, o["submissions_snapshot"], live_subs.get(slug))
        if slug and o is not None
        else {"status": "n/a", "source": None, "fetched_at": None, "error": "", "items": []}
    )
    # What Kaggle said was left today after our last submission (None: nothing today).
    subs["daily"] = o.get("submissions_daily") if o is not None else None

    our_lb = _our_best_lb(subs, o["competition"]["direction"]) if o is not None else None
    if our_lb is not None:
        maximize = o is not None and o["competition"]["direction"].startswith("max")
        for h in hypotheses:
            claimed = (h.get("kernel") or {}).get("claimed_score")
            if isinstance(claimed, (int, float)):
                gap = claimed - our_lb["value"] if maximize else our_lb["value"] - claimed
                h["kernel"]["vs_our_best"] = round(gap, 6)

    calls = (o["token_calls"] if o else []) + (r["ai_calls"] if r else [])
    mode = o["setup"]["ai_provider_mode"] if o else ""

    if o and o["has_setup"] and not experiments:
        _issue(
            issues,
            "info",
            "no experiments yet",
            f"researchforge kaggle prepare-round {slug} → write proposal.json → execute-round "
            f"{slug}",
        )

    last_ts = max(
        [t for t in ((o or {}).get("last_activity_ts"), (r or {}).get("last_activity_ts")) if t],
        default=None,
    )
    repo_fresh = bool(
        r and r["last_activity_ts"] and time.time() - r["last_activity_ts"] < STALE_RUN_HOURS * 3600
    )
    repo_launch = bool(r and (r.get("auto_run") or {}).get("running"))
    repo_running = bool(
        r and (r["run_in_progress"] or (r["autorun"] and r["autorun"].get("status") == "running"))
    )
    if repo_running and not repo_fresh and not repo_launch and r is not None:
        repo_running = False
        _issue(
            issues,
            "warn",
            "a run is marked in progress but the project DB has not changed for "
            f"{(time.time() - (r['last_activity_ts'] or 0)) / 3600:.0f}h — it was likely "
            "interrupted",
            "resume with `researchforge experiment resume` (or `researchforge autorun --resume`)",
        )
        for e in experiments:
            if e["source"] == "repo" and e["status"] == "running":
                e["status"] = "stale"
    running = any(e["status"] == "running" for e in experiments) or repo_running or repo_launch
    watching = bool(o and (o["sensors"]["monitor_alive"] or (sched and sched["live"])))
    recent = bool(last_ts and time.time() - last_ts < ACTIVE_WINDOW_HOURS * 3600)
    role = o["role"] if o else "primary"
    stalled = not running and any(e["status"] == "stale" for e in experiments)
    waiting = bool(o and o.get("waiting_input"))
    if waiting and o is not None:
        _issue(
            issues,
            "warn",
            f"{len(o['waiting_input']['questions']) or 'some'} question(s) wait for your answer",
            f"answer from the dashboard's 🔔, or `researchforge kaggle respond {slug}`",
        )
    paused = bool(o and (o.get("control") or {}).get("paused"))
    if paused and o is not None:
        reason = (o.get("control") or {}).get("pause_reason") or ""
        _issue(
            issues,
            "info",
            "the loop is paused" + (f": {reason}" if reason else ""),
            f"resume with `researchforge kaggle resume {slug}`",
        )
    state = (
        "running"
        if running
        else "paused"
        if paused
        else "waiting"
        if waiting
        else "stalled"
        if stalled
        else "watching"
        if watching
        else "recent"
        if recent
        else "idle"
    )
    if role != "primary":
        state = role

    kinds = (["kaggle"] if o else []) + (["repo"] if r else [])
    if o:
        category = "kaggle"
    elif r and (r.get("project") or {}).get("mode") == "explore_research_idea":
        category = "research"
    else:
        category = "repository"
    questions = _questions(
        od if o else None, loc.repo_dir if r else None, experiments, slug or name
    )
    proj: dict[str, Any] = {
        "id": pid,
        "name": name,
        "slug": slug,
        "category": category,
        "questions": questions,
        "root": str(root),
        "location": str(od if od is not None else root),
        "kinds": kinds,
        "role": role,
        "state": state,
        "found_by": sorted(loc.found_by),
        "last_activity_at": _iso(last_ts),
        "merge_candidates": ([o["merge_candidate"]] if o and o.get("merge_candidate") else [])
        + (r["merge_candidates"] if r else []),
        "overnight": {
            k: v
            for k, v in o.items()
            if k
            not in ("experiments", "hypotheses", "issues", "token_calls", "submissions_snapshot")
        }
        if o
        else None,
        "repo": {
            k: v
            for k, v in r.items()
            if k not in ("experiments", "hypotheses", "issues", "ai_calls")
        }
        if r
        else None,
        "experiments": sorted(experiments, key=lambda e: (e["source"], e["num"])),
        "hypotheses": hypotheses,
        "kernel_board": o.get("kernel_board") if o is not None else None,
        "our_best_lb": our_lb,
        "best": None,
        "scheduler": sched,
        "submissions": subs,
        "tokens": _tokens(calls, mode),
        "issues": issues,
    }
    for e in proj["experiments"]:
        if e["source"] == "overnight":
            e["tokens"] = proj["tokens"]["by_round"].get(e["id"])
    b = _best(experiments)
    proj["best"] = {"id": b["id"], "metric": b["metric"]} if b else None
    slug_lc = (slug or name).lower()
    proj_notes = [
        n
        for n in notifications
        if slug_lc and slug_lc in (n.get("title", "") + n.get("message", "")).lower()
    ]
    proj["notification_count"] = len(proj_notes)
    proj["timeline"] = _timeline(proj, proj_notes)
    return proj


def build_state(
    cwd: Path | None = None, live_submissions: dict[str, Any] | None = None
) -> dict[str, Any]:
    t0 = time.time()
    locations, discovery = discover_locations(cwd)
    notifications = read_notifications()

    setups_by_slug: dict[str, list[str]] = {}
    for loc in locations:
        for od in loc.overnight_dirs:
            if has_setup(od) and location_role(od) == "primary":
                setups_by_slug.setdefault(slug_of(od), []).append(str(od))

    projects = []
    for loc in locations:
        # One project per overnight dir; a lone repo project is its own entry.
        targets: list[Path | None] = list(loc.overnight_dirs) or [None]
        for target in targets:
            try:
                projects.append(
                    _build_project(
                        loc, target, setups_by_slug, notifications, live_submissions or {}
                    )
                )
            except Exception as exc:  # one broken project must not blank the dashboard
                projects.append(
                    {
                        "id": project_id(target or loc.root),
                        "name": slug_of(target) if target else loc.root.name,
                        "slug": slug_of(target) if target else None,
                        "root": str(loc.root),
                        "location": str(target or loc.root),
                        "kinds": [],
                        "role": "primary",
                        "state": "error",
                        "found_by": sorted(loc.found_by),
                        "last_activity_at": None,
                        "overnight": None,
                        "repo": None,
                        "experiments": [],
                        "hypotheses": [],
                        "best": None,
                        "scheduler": None,
                        "submissions": {
                            "status": "n/a",
                            "items": [],
                            "error": "",
                            "fetched_at": None,
                            "source": None,
                        },
                        "tokens": _tokens([], ""),
                        "timeline": [],
                        "notification_count": 0,
                        "issues": [
                            {
                                "level": "error",
                                "message": (
                                    f"failed to build project state: {type(exc).__name__}: {exc}"
                                ),
                                "hint": "",
                            }
                        ],
                    }
                )

    order = {
        "running": 0,
        "waiting": 1,
        "stalled": 2,
        "watching": 3,
        "recent": 4,
        "idle": 5,
        "error": 6,
        "worktree": 7,
        "backup": 8,
    }
    projects.sort(
        key=lambda p: (
            order.get(p["state"], 9),
            -(
                datetime.fromisoformat(p["last_activity_at"]).timestamp()
                if p["last_activity_at"]
                else 0
            ),
        )
    )
    live = [p for p in projects if p["role"] == "primary"]
    totals = {
        "projects": len(live),
        "copies": len(projects) - len(live),
        "active": sum(
            1 for p in live if p["state"] in ("running", "waiting", "stalled", "watching", "recent")
        ),
        "attention": sum(1 for p in live if p["state"] in ("waiting", "stalled")),
        "running": sum(1 for p in live if p["state"] == "running"),
        "experiments": sum(len(p["experiments"]) for p in live),
        "improved": sum(1 for p in live for e in p["experiments"] if e["status"] == "improved"),
        "hypotheses": sum(len(p["hypotheses"]) for p in live),
        "untested": sum(
            1
            for p in live
            for h in p["hypotheses"]
            if not h["tested"] and h["status"] not in ("rejected",)
        ),
        "schedulers_live": sum(1 for p in live if p["scheduler"] and p["scheduler"].get("live")),
        "schedulers_failing": sum(
            1
            for p in live
            if p["scheduler"]
            and p["scheduler"]["state"] == "failing"
            and p["scheduler"].get("targets_this_dir")
        ),
        "submissions": sum(len(p["submissions"]["items"]) for p in live),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "build_ms": None,
        "discovery": discovery,
        "totals": totals,
        "projects": projects,
        "notifications": notifications[:100],
        "gpu_quota": {
            "available": False,
            "note": (
                "Kaggle's API does not expose remaining GPU quota; check kaggle.com/me/settings."
            ),
        },
    }
    payload["build_ms"] = int((time.time() - t0) * 1000)
    body = json.dumps(
        {k: v for k, v in payload.items() if k not in ("generated_at", "build_ms")},
        sort_keys=True,
        default=str,
    )
    payload["fingerprint"] = hashlib.sha1(body.encode()).hexdigest()[:16]
    return payload
