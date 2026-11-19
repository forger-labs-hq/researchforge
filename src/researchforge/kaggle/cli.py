"""Kaggle Overnight CLI commands.

Entry points:
  researchforge kaggle overnight <competition>
  researchforge kaggle overnight setup <competition>
  researchforge kaggle overnight run <competition>
"""

from __future__ import annotations

import contextlib
import json
import threading as _threading
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from researchforge.kaggle.orchestrator import OvernightError, OvernightOrchestrator
from researchforge.kaggle.run_config import round_time_limit_seconds
from researchforge.kaggle.token_log import metered
from researchforge.kaggle.workspace import atomic_write_json, read_json, resolve_artifacts_dir

kaggle_app = typer.Typer(
    name="kaggle",
    help="[bold #7C3AED]Kaggle[/] — Run overnight experiments on Kaggle competitions.",
)


@kaggle_app.command("dashboard")
def dashboard_cmd(
    port: Annotated[int, typer.Option("--port", help="Port to serve on")] = 7842,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open in browser")] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print raw JSON and exit")] = False,
    daemon: Annotated[
        bool, typer.Option("--daemon", help="Start in background (survives terminal close)")
    ] = False,
    stop: Annotated[
        bool, typer.Option("--stop", help="Stop a running background dashboard")
    ] = False,
    status: Annotated[
        bool, typer.Option("--status", help="Show whether dashboard is running")
    ] = False,
    daemon_internal: Annotated[bool, typer.Option("--daemon-internal", hidden=True)] = False,
) -> None:
    """Open the Dagster-style dashboard showing all competitions, experiments, schedulers.

    Serves a rich web UI on localhost with:
      • Automation panel (schedulers, watchers per competition)
      • Experiment DAG visualization
      • Hypothesis pool browser
      • Submission tracker
      • Notification timeline
      • Insights & statistics

    Examples:
        researchforge kaggle dashboard               # open at localhost:7842
        researchforge kaggle dashboard --daemon      # start in background
        researchforge kaggle dashboard --stop        # stop background instance
        researchforge kaggle dashboard --status      # check if running
        researchforge kaggle dashboard --json        # raw JSON data
    """
    from researchforge.kaggle.dashboard import (
        collect_all,
        is_running,
        serve,
        start_daemon,
        stop_daemon,
    )

    if stop:
        if stop_daemon():
            typer.secho("✅ Dashboard stopped.", fg=typer.colors.GREEN)
        else:
            typer.echo("Dashboard is not running.")
        return

    if status:
        running, p = is_running(port)
        if running:
            typer.secho(f"✅ Dashboard running at http://localhost:{p}", fg=typer.colors.GREEN)
        else:
            typer.echo("Dashboard is not running.")
        return

    if json_output:
        data = collect_all()
        typer.echo(json.dumps(data, indent=2, default=str))
        return

    if daemon:
        running, p = is_running(port)
        if running:
            typer.echo(f"Dashboard already running at http://localhost:{p}")
        else:
            try:
                p = start_daemon(port=port)
            except RuntimeError as exc:
                typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1) from None
            typer.secho(
                f"✅ Dashboard started in background → http://localhost:{p}", fg=typer.colors.GREEN
            )
            typer.echo("   Stop with: researchforge kaggle dashboard --stop")
        if open_browser:
            import webbrowser

            webbrowser.open(f"http://localhost:{p}")
        return

    if daemon_internal:
        # Called by start_daemon() subprocess — run silently
        serve(port=port, open_browser=False, daemon_internal=True)
        return

    serve(port=port, open_browser=open_browser)


@kaggle_app.command("gather-facts")
def gather_facts_cmd(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Force re-gather even if cached")
    ] = False,
) -> None:
    """Gather all competition facts and print as JSON. Runs in ~30s, zero user input.

    Claude reads this output to decide which questions to ask before starting overnight.

    Example:
        researchforge kaggle gather-facts rsna-knee-abnormality-detection
    """
    import json as _json

    from researchforge.kaggle.facts_gatherer import gather_facts
    from researchforge.kaggle.orchestrator import OvernightOrchestrator

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Gathering facts for {slug}...", err=True)
    facts = gather_facts(slug, artifacts_dir, force_refresh=refresh)

    import dataclasses

    typer.echo(_json.dumps(dataclasses.asdict(facts), indent=2))


# Submissions are scored in minutes; checking them only every --poll-hours delayed
# the notification by up to that long.
SUBMISSION_POLL_MINUTES = 15.0
# Public notebooks change within hours (new versions with better scores); every 6 hours
# left the pool and the notebook board a cycle behind.
KERNEL_POLL_MINUTES = 60.0
# Auto-run with back_to_back: how soon after a launch ends the next may start.
AUTO_RUN_POLL_MINUTES = 10.0
SUBMISSION_NOTICES_FILE = "submission_notices.json"


@kaggle_app.command("monitor")
def monitor_cmd(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    poll_hours: Annotated[
        float, typer.Option("--poll-hours", help="How often to run all checks (hours) [default: 6]")
    ] = 6.0,
    arxiv_hours: Annotated[
        float,
        typer.Option(
            "--arxiv-hours", help="How often to run arXiv scan (hours, 0=off) [default: 24]"
        ),
    ] = 24.0,
    briefing_time: Annotated[
        str,
        typer.Option("--briefing-at", help="Send morning briefing at HH:MM local time ('' = off)"),
    ] = "07:00",
    heartbeat_stale_hours: Annotated[
        float,
        typer.Option(
            "--heartbeat-stale", help="Alert if execute_round.lock older than N hours [default: 2]"
        ),
    ] = 2.0,
    once: Annotated[
        bool, typer.Option("--once", help="Run one cycle and exit (for testing)")
    ] = False,
    submission_minutes: Annotated[
        float,
        typer.Option(
            "--submission-minutes",
            help="Check Kaggle submissions this often between cycles (minutes, 0 = only with "
            "each cycle) [default: 15]",
        ),
    ] = SUBMISSION_POLL_MINUTES,
    kernel_minutes: Annotated[
        float,
        typer.Option(
            "--kernel-minutes",
            help="Check public notebooks this often between cycles (minutes, 0 = only with "
            "each cycle) [default: 60]",
        ),
    ] = KERNEL_POLL_MINUTES,
) -> None:
    """Unified background monitor — replaces 'watch'. Runs all checks in one cycle.

    Every --submission-minutes (and with each cycle):
      • Submission status → ntfy as soon as a submission is scored

    Every --kernel-minutes (and with each cycle):
      • New and re-run public notebooks → ideas injected, the ranked notebook
        board (kernel_board.json) rebuilt → ntfy if one claims a better LB than ours

    Every 10 minutes, when auto-run is on with back_to_back:
      • The next launch starts once the last has ended and the gates pass

    Every --poll-hours:
      • Experiment heartbeat → ntfy if execute_round.lock is stale (hung experiment)
      • Hypothesis pool delta → ntfy if pool grew significantly

    Once daily (--arxiv-hours):
      • arXiv scan → new papers → ideas injected → ntfy

    At --briefing-at time:
      • Morning briefing → full summary ntfy push

    Install as a persistent background service (survives app close):
        researchforge kaggle scheduler-install <slug> --interval-hours 6

    Examples:
        researchforge kaggle monitor titanic --once
        researchforge kaggle monitor rsna-knee --poll-hours 4 --briefing-at 07:00
    """
    import os as _os
    from datetime import timedelta as _td

    from researchforge.kaggle import sensors as _sensors
    from researchforge.kaggle.orchestrator import OvernightOrchestrator

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # monitor_state.json = runtime sensor state (what happened); sensor
    # enable/disable + custom sensors live in scheduler_config.json.
    state: dict[str, Any] = read_json(artifacts_dir / "monitor_state.json", {}) or {}
    if "disabled_sensors" in state:  # migrate legacy toggle storage
        if not (artifacts_dir / "scheduler_config.json").exists():
            _sensors.load_config(artifacts_dir, slug).save(artifacts_dir)
        state.pop("disabled_sensors", None)

    def _kernels(results: dict[str, Any]) -> tuple[str, str]:
        """Public notebooks: new and re-run ones, their ideas, and the ranked board."""
        from researchforge.kaggle.kernel_watcher import WatcherConfig, run_watch_cycle

        baseline_score = None
        setup_path = artifacts_dir / "overnight_setup.json"
        if setup_path.exists():
            try:
                from researchforge.kaggle.orchestrator import OvernightSetup

                s = OvernightSetup.load(setup_path)
                baseline_score = s.baseline_result.cv_score if s.baseline_result else None
            except Exception:
                pass
        config = WatcherConfig.load(artifacts_dir)
        if config is None:
            from researchforge.kaggle.api import list_competition_kernels

            known_refs = [k.ref for k in list_competition_kernels(slug, page_size=50)]
            config = WatcherConfig(
                competition_slug=slug, interval_hours=poll_hours, seen_kernel_refs=known_refs
            )
            config.save(artifacts_dir)
            return "ok", f"initialised with {len(known_refs)} known kernels"
        ai_provider = _project_provider(artifacts_dir, "kernel_ideas")
        wr = run_watch_cycle(artifacts_dir, ai_provider, baseline_score, check_submissions=False)
        results["new_kernels"] = wr.new_kernels_found
        results["new_ideas"] = wr.new_ideas_extracted
        results["better_kernel"] = wr.better_kernel is not None
        detail = (
            f"{wr.new_kernels_found} new, {wr.updated_kernels_found} re-run notebooks, "
            f"{len(wr.new_ideas_extracted)} ideas injected"
        )
        if wr.better_kernel:
            better = wr.better_kernel
            detail += f"; ahead of us: {better['title'][:60]} (claims {better['lb_score']})"
        return "ok", detail

    def _run_one_cycle() -> dict[str, Any]:
        """Run all enabled sensors. Returns summary of what happened."""
        results: dict[str, Any] = {"slug": slug, "cycle_at": datetime.now().isoformat()}
        # Re-read every cycle so dashboard / CLI toggles apply without a restart.
        cfg = _sensors.load_config(artifacts_dir, slug)

        def _sensor(name: str, fn: Any) -> None:
            if not _sensors.is_enabled(cfg, name):
                _sensors.record(state, name, "disabled")
                return
            try:
                status, detail = fn()
                _sensors.record(state, name, status, detail)
            except Exception as exc:
                _sensors.record(state, name, "error", error=f"{type(exc).__name__}: {exc}")
                results.setdefault("errors", []).append(f"{name}: {exc}")

        _sensor("kernel_watcher_sensor", lambda: _kernels(results))

        # ── 2. Submission status (also runs between cycles, see below) ──────
        _sensor("submission_status_sensor", lambda: _submission_check(slug, artifacts_dir))
        _round_followups(artifacts_dir)

        # ── 3. Hypothesis pool delta ────────────────────────────────────────
        def _pool() -> tuple[str, str]:
            pool = read_json(artifacts_dir / "pending_hypotheses.json", []) or []
            current = len(pool)
            delta = current - state.get("last_pool_count", 0)
            if delta >= 3:
                _notify_pool_delta(slug, delta, current, artifacts_dir)
            state["last_pool_count"] = current
            results["pool_count"] = current
            untested = sum(1 for h in pool if not h.get("tested"))
            return "ok", f"{current} hypotheses ({untested} untested), {delta:+d} since last cycle"

        _sensor("hypothesis_pool_sensor", _pool)

        # ── 4. Experiment heartbeat ─────────────────────────────────────────
        def _heartbeat() -> tuple[str, str]:
            # execute-round holds execute_round.lock; both it and run-parallel
            # publish rounds in inflight_rounds.json. A lock whose process died is
            # released first (its round recorded as interrupted) so it cannot stall the loop.
            from researchforge.kaggle.experiment_loop import recover_interrupted_round

            recovered = recover_interrupted_round(artifacts_dir)
            # A repo project beside the competition is folded into this one history.
            from researchforge.kaggle.history_import import import_repo_history

            with contextlib.suppress(Exception):  # never let the import stop the monitor
                import_repo_history(artifacts_dir)
            live: dict[str, Any] = dict(read_json(artifacts_dir / "inflight_rounds.json", {}) or {})
            lock_path = artifacts_dir / "execute_round.lock"
            if lock_path.exists():
                info = read_json(lock_path, {}) or {}
                live.setdefault(str(info.get("round_num", "?")), info)
                live[str(info.get("round_num", "?"))]["_lock_mtime"] = lock_path.stat().st_mtime
            if not live:
                if recovered is not None:
                    return "ok", f"released a stale lock; round {recovered} recorded as interrupted"
                return "ok", "no experiment running"
            problems, running, first_bad = [], [], None
            for key, info in live.items():
                started = info.get("started_at") or info.get("updated_at") or ""
                try:
                    t0 = info.get("_lock_mtime") or datetime.fromisoformat(started).timestamp()
                except ValueError:
                    t0 = datetime.now().timestamp()
                age_hours = (datetime.now().timestamp() - t0) / 3600
                pid = info.get("pid")
                alive = True
                if pid:
                    try:
                        _os.kill(int(pid), 0)
                    except ProcessLookupError:
                        alive = False
                    except (PermissionError, ValueError):
                        pass
                if not alive:
                    problems.append(f"round {key}: process {pid} is gone")
                    first_bad = first_bad or (age_hours, info)
                elif age_hours > heartbeat_stale_hours:
                    problems.append(
                        f"round {key} running {age_hours:.1f}h (> {heartbeat_stale_hours}h)"
                    )
                    first_bad = first_bad or (age_hours, info)
                else:
                    running.append(f"round {key} ({info.get('stage', '?')}, {age_hours * 60:.0f}m)")
            if problems and first_bad is not None:
                _notify_experiment_hung(slug, first_bad[0], first_bad[1], artifacts_dir)
                results["experiment_hung"] = True
                raise RuntimeError("; ".join(problems) + " — experiment may be hung or crashed")
            return "ok", "running: " + ", ".join(running)

        _sensor("experiment_heartbeat_sensor", _heartbeat)

        # ── 5. arXiv scan (daily) ───────────────────────────────────────────
        def _arxiv() -> tuple[str, str]:
            if arxiv_hours <= 0:
                return "skipped", "disabled by --arxiv-hours 0"
            last_arxiv = state.get("last_arxiv_at", "")
            hours_since = (
                999.0
                if not last_arxiv
                else (datetime.now().timestamp() - datetime.fromisoformat(last_arxiv).timestamp())
                / 3600
            )
            if hours_since < arxiv_hours:
                return "skipped", f"not due — next scan in {arxiv_hours - hours_since:.1f}h"
            ideas_added = _run_arxiv_scan(slug, artifacts_dir)
            results["arxiv_ideas"] = ideas_added
            state["last_arxiv_at"] = datetime.now().isoformat()
            if ideas_added:
                _notify_arxiv_scan(slug, ideas_added, artifacts_dir)
            return "ok", f"{len(ideas_added)} arXiv ideas in pool"

        _sensor("arxiv_scanner_sensor", _arxiv)

        # ── 6. Morning briefing (once per day at configured time) ───────────
        def _briefing() -> tuple[str, str]:
            if not briefing_time:
                return "skipped", "no --briefing-at time configured"
            today = datetime.now().strftime("%Y-%m-%d")
            if state.get("last_briefing_date") == today:
                return "skipped", f"already sent today; daily at {briefing_time}"
            if datetime.now().strftime("%H:%M") < briefing_time:
                return "skipped", f"not due — daily at {briefing_time}"
            _send_morning_briefing(slug, artifacts_dir)
            state["last_briefing_date"] = today
            return "ok", "briefing sent"

        _sensor("morning_briefing_sensor", _briefing)

        # ── 7. Auto-run: start rounds on untested ideas (off by default) ────
        def _auto_run() -> tuple[str, str]:
            from researchforge.kaggle.auto_run import sensor_check

            return sensor_check(artifacts_dir, slug)

        _sensor("auto_run_sensor", _auto_run)

        # ── 8. Custom sensors (user-defined commands) ───────────────────────
        for spec in cfg.custom_sensors:
            name = spec.get("name", "")
            if not name:
                continue
            if not spec.get("enabled", True):
                _sensors.record(state, name, "disabled")
                continue
            status, detail, error = _sensors.run_custom(spec, artifacts_dir.parent)
            _sensors.record(state, name, status, detail, error)

        now = datetime.now()
        state["last_cycle_at"] = now.isoformat(timespec="seconds")
        state["poll_hours"] = poll_hours
        state["cycle_count"] = int(state.get("cycle_count", 0)) + 1
        if once:
            state.pop("next_cycle_at", None)
            state.pop("monitor_pid", None)
        else:
            state["next_cycle_at"] = (now + _td(hours=poll_hours)).isoformat(timespec="seconds")
            state["monitor_pid"] = _os.getpid()
        _sensors.save_state(artifacts_dir, state)
        return results

    if once:
        result = _run_one_cycle()
        typer.echo(json.dumps(result, indent=2))
        return

    def _submissions_between_cycles() -> None:
        """A scored submission is announced within minutes, not at the next full cycle."""
        name = "submission_status_sensor"
        if not _sensors.is_enabled(_sensors.load_config(artifacts_dir, slug), name):
            return
        try:
            status, detail = _submission_check(slug, artifacts_dir)
            _sensors.record(state, name, status, detail)
        except Exception as exc:
            _sensors.record(state, name, "error", error=f"{type(exc).__name__}: {exc}")
        _sensors.save_state(artifacts_dir, state)
        _round_followups(artifacts_dir)

    def _auto_run_between_cycles() -> None:
        """With back_to_back, a launch follows the last one within minutes."""
        from researchforge.kaggle.auto_run import AutoRunConfig, current_launch, sensor_check

        name = "auto_run_sensor"
        if not _sensors.is_enabled(_sensors.load_config(artifacts_dir, slug), name):
            return
        if not AutoRunConfig.load(artifacts_dir).back_to_back or current_launch(artifacts_dir):
            return
        try:
            status, detail = sensor_check(artifacts_dir, slug)
            _sensors.record(state, name, status, detail)
        except Exception as exc:
            _sensors.record(state, name, "error", error=f"{type(exc).__name__}: {exc}")
        _sensors.save_state(artifacts_dir, state)

    def _kernels_between_cycles() -> None:
        """New public notebooks reach the pool and the board within the hour."""
        name = "kernel_watcher_sensor"
        if not _sensors.is_enabled(_sensors.load_config(artifacts_dir, slug), name):
            return
        try:
            status, detail = _kernels({})
            _sensors.record(state, name, status, detail)
        except Exception as exc:
            _sensors.record(state, name, "error", error=f"{type(exc).__name__}: {exc}")
        _sensors.save_state(artifacts_dir, state)

    typer.echo(f"Monitoring {slug} every {poll_hours}h → {artifacts_dir}. Ctrl+C to stop.")
    while True:
        try:
            result = _run_one_cycle()
            errs = f" errors={len(result['errors'])}" if result.get("errors") else ""
            typer.echo(
                f"[{result['cycle_at'][:16]}] kernels={result.get('new_kernels', 0)} "
                f"pool={result.get('pool_count', 0)} ideas={len(result.get('new_ideas', []))}{errs}"
            )
        except Exception as exc:
            typer.echo(f"[monitor] error: {exc}", err=True)
        due = datetime.now() + _td(hours=poll_hours)
        steps = [m for m in (submission_minutes, kernel_minutes, AUTO_RUN_POLL_MINUTES) if m > 0]
        step = _td(minutes=min(steps)) if steps else None
        next_kernels = datetime.now() + _td(minutes=kernel_minutes) if kernel_minutes > 0 else None
        next_subs = datetime.now() + _td(minutes=submission_minutes)
        while step is not None and datetime.now() + step < due:
            _wait_until_due(datetime.now() + step)
            if submission_minutes > 0 and datetime.now() >= next_subs:
                _submissions_between_cycles()
                next_subs = datetime.now() + _td(minutes=submission_minutes)
            if next_kernels is not None and datetime.now() >= next_kernels:
                _kernels_between_cycles()
                next_kernels = datetime.now() + _td(minutes=kernel_minutes)
            _auto_run_between_cycles()
        _wait_until_due(due)


def _round_followups(artifacts_dir: Path) -> None:
    """Collect Kaggle rounds that outlived their wait, ask about a missed best round, and
    withdraw offers for notebook rounds that wrote nothing to submit.

    Runs with the submission checks (every few minutes), so a late result and its
    submission question arrive without waiting for the next full cycle.
    """
    from researchforge.kaggle import late_rounds, submission

    for step in (late_rounds.collect, submission.offer_missed, submission.withdraw_unsubmittable):
        try:
            got = step(artifacts_dir)
        except Exception as exc:  # never let a follow-up stop the monitor
            typer.echo(f"[monitor] {step.__module__}.{step.__name__}: {exc}", err=True)
            continue
        if got:
            typer.echo(f"[monitor] {step.__name__}: {got}")


def _submission_check(slug: str, artifacts_dir: Path) -> tuple[str, str]:
    """List Kaggle submissions; notify once for each newly scored one (submission sensor)."""
    from researchforge.kaggle.kernel_watcher import (
        WatcherConfig,
        _check_new_completed_submissions,
        _notify_submission_complete,
        _notify_submission_event,
    )

    config = WatcherConfig.load(artifacts_dir)
    seen = config.seen_submission_ids if config else []
    completed = _check_new_completed_submissions(slug, seen, artifacts_dir)
    snap = read_json(artifacts_dir / "submissions.json", {}) or {}
    if snap.get("error"):
        raise RuntimeError(snap["error"])
    for sub in completed:
        _notify_submission_complete(slug, sub, artifacts_dir)
    if config is not None and completed:
        config.seen_submission_ids.extend(sub["id"] for sub in completed)
        config.save(artifacts_dir)
    subs = snap.get("submissions", [])
    # Also announce a submission as soon as it is listed, and one that fails. The first
    # look at a project only records what is there, so old submissions stay quiet.
    notices = read_json(artifacts_dir / SUBMISSION_NOTICES_FILE, None)
    first = not isinstance(notices, dict)
    notices = notices if isinstance(notices, dict) else {}
    scored_now = {str(c["id"]) for c in completed}
    for x in subs:
        ref, st = str(x.get("ref") or ""), str(x.get("status", "")).lower()
        if not ref:
            continue
        failed = "error" in st or bool(x.get("errordescription"))  # a timeout is "complete"
        event = "failed" if failed else "scored" if "complete" in st else "listed"
        if event == "scored" or ref in scored_now:  # announced through seen_submission_ids
            notices[ref] = "scored"
            continue
        if notices.get(ref) in (event, "failed", "scored"):
            continue
        if not first:
            _notify_submission_event(slug, x, artifacts_dir, event)
        notices[ref] = event
    atomic_write_json(artifacts_dir / SUBMISSION_NOTICES_FILE, notices)
    pending = sum(1 for x in subs if "pending" in str(x.get("status", "")).lower())
    return "ok", (
        f"{len(subs)} submissions listed, {len(completed)} newly complete"
        + (f", {pending} still scoring" if pending else "")
    )


def _project_provider(artifacts_dir: Path, purpose: str) -> Any:
    """The project's driver for an unattended AI call (see auto_run.project_provider)."""
    from researchforge.kaggle.auto_run import project_provider

    return project_provider(artifacts_dir, purpose)


def _earlier_repo_work(artifacts_dir: Path) -> list[dict[str, Any]]:
    """The repo engine's experiments, brought up to date first (see history_import)."""
    from researchforge.kaggle.history_import import earlier_work, import_repo_history

    with contextlib.suppress(Exception):  # a broken repo DB must not stop a round
        import_repo_history(artifacts_dir)
    return earlier_work(artifacts_dir)


def _wait_until_due(due: datetime, step_seconds: float = 60.0) -> None:
    """Sleep until the wall clock reaches ``due``, checking every ``step_seconds``.

    One long ``time.sleep`` does not count time the Mac spends asleep, so a closed
    lid pushed every cycle back by hours. Short naps against the real clock run an
    overdue cycle within a minute of waking.
    """
    import time as _time

    while True:
        left = (due - datetime.now()).total_seconds()
        if left <= 0:
            return
        _time.sleep(min(step_seconds, left))


def _notify_pool_delta(slug: str, delta: int, total: int, artifacts_dir: Path) -> None:
    """Notify when hypothesis pool grows significantly."""
    from researchforge.kaggle.control import should_notify

    if not should_notify(artifacts_dir, "ideas"):
        return
    try:
        import os as _os

        from researchforge.notifications.service import notify_new_hypotheses

        setup = artifacts_dir / "overnight_setup.json"
        topic = _os.environ.get("NTFY_TOPIC", "")
        if not topic and setup.exists():
            topic = json.loads(setup.read_text()).get("ntfy_topic", "")
        notify_new_hypotheses(
            delta, [f"+{delta} new since last check"], project_name=slug, topic=topic
        )
    except Exception:
        pass


def _notify_experiment_hung(
    slug: str, age_hours: float, lock_info: dict[str, Any], artifacts_dir: Path
) -> None:
    """Notify when an experiment lock file is stale."""
    from researchforge.kaggle.control import should_notify

    if not should_notify(artifacts_dir, "alert"):
        return
    try:
        import os as _os

        from researchforge.notifications.service import Channel, Notification, send_notification

        topic = _os.environ.get("NTFY_TOPIC", "")
        if not topic:
            setup = artifacts_dir / "overnight_setup.json"
            if setup.exists():
                topic = json.loads(setup.read_text()).get("ntfy_topic", "")
        pid = lock_info.get("pid", "?")
        started = lock_info.get("started_at", "?")[:16]
        old = _os.environ.get("NTFY_TOPIC")
        if topic:
            _os.environ["NTFY_TOPIC"] = topic
        try:
            send_notification(
                Notification(
                    title=f"ResearchForge: experiment hung? ({slug})",
                    message=(
                        f"Round {lock_info.get('round_num', '?')} in flight for {age_hours:.1f}h "
                        f"(PID {pid}, started {started}). Check if it crashed."
                    ),
                    channel=Channel.NTFY,
                    priority="high",
                    tags=["warning"],
                )
            )
        finally:
            if topic:
                if old is None:
                    _os.environ.pop("NTFY_TOPIC", None)
                else:
                    _os.environ["NTFY_TOPIC"] = old
    except Exception:
        pass


def _run_arxiv_scan(slug: str, artifacts_dir: Path) -> list[str]:
    """Run one arXiv scan cycle and return the arXiv ideas now in the pool.

    Errors propagate so the arXiv sensor records them instead of reporting "ok".
    """
    from researchforge.intelligence.scanner import (
        _run_arxiv_cycle,
        inject_arxiv_ideas_into_hypothesis_pool,
    )

    intel_dir = artifacts_dir / "intelligence"
    intel_dir.mkdir(exist_ok=True)
    meta = read_json(artifacts_dir / "competition_metadata.json", {}) or {}
    # Same query the arxiv-scan command builds: task + metric, else the slug.
    topic = " ".join(str(meta.get(k) or "") for k in ("task_type", "metric_name")).strip()
    query = f"{topic or 'kaggle ' + slug.replace('-', ' ')} machine learning"
    ai_provider = _project_provider(artifacts_dir, "arxiv_scan")
    _run_arxiv_cycle(query, intel_dir, ai_provider, 10)
    inject_arxiv_ideas_into_hypothesis_pool(intel_dir, artifacts_dir)
    pool = read_json(artifacts_dir / "pending_hypotheses.json", []) or []
    return [
        str(h.get("idea", ""))[:50]
        for h in pool
        if isinstance(h, dict) and str(h.get("source", "")).startswith("arxiv")
    ][:5]


def _notify_arxiv_scan(slug: str, ideas: list[str], artifacts_dir: Path) -> None:
    """Notify when new arXiv papers added ideas to the pool."""
    from researchforge.kaggle.control import should_notify

    if not should_notify(artifacts_dir, "ideas"):
        return
    try:
        import os as _os

        from researchforge.notifications.service import Channel, Notification, send_notification

        topic = _os.environ.get("NTFY_TOPIC", "")
        if not topic:
            setup = artifacts_dir / "overnight_setup.json"
            if setup.exists():
                topic = json.loads(setup.read_text()).get("ntfy_topic", "")
        msg = f"{len(ideas)} new technique(s) from arXiv: " + "; ".join(ideas[:3])
        old = _os.environ.get("NTFY_TOPIC")
        if topic:
            _os.environ["NTFY_TOPIC"] = topic
        try:
            send_notification(
                Notification(
                    title=f"ResearchForge: arXiv scan ({slug})",
                    message=msg[:300],
                    channel=Channel.NTFY,
                    tags=["page_facing_up"],
                    priority="low",
                )
            )
        finally:
            if topic:
                if old is None:
                    _os.environ.pop("NTFY_TOPIC", None)
                else:
                    _os.environ["NTFY_TOPIC"] = old
    except Exception:
        pass


def _send_morning_briefing(slug: str, artifacts_dir: Path) -> None:
    """Send a morning briefing summary push notification."""
    from researchforge.kaggle.control import should_notify

    if not should_notify(artifacts_dir, "briefing"):
        return
    try:
        import os as _os
        import sqlite3 as _sql

        from researchforge.notifications.service import Channel, Notification, send_notification

        topic = _os.environ.get("NTFY_TOPIC", "")
        if not topic:
            setup = artifacts_dir / "overnight_setup.json"
            if setup.exists():
                topic = json.loads(setup.read_text()).get("ntfy_topic", "")

        # Gather stats
        db_path = artifacts_dir / "experiment_log.db"
        n_exp = n_improved = 0
        best_metric = None
        if db_path.exists():
            try:
                conn = _sql.connect(db_path)
                rows = conn.execute(
                    "SELECT full_improved, full_metric FROM experiments WHERE completed_at >= "
                    "datetime('now', '-1 day')"
                ).fetchall()
                n_exp = len(rows)
                improved = [r for r in rows if r[0]]
                n_improved = len(improved)
                if improved:
                    best_metric = max(r[1] for r in improved if r[1])
                conn.close()
            except Exception:
                pass

        pool_size = 0
        pool_path = artifacts_dir / "pending_hypotheses.json"
        if pool_path.exists():
            with contextlib.suppress(Exception):
                pool_size = len(json.loads(pool_path.read_text()))

        metric_str = f"Best: {best_metric:.4f}" if best_metric else "No metric"
        msg = (
            f"Last 24h: {n_exp} experiments, {n_improved} improved. {metric_str}. "
            f"Pool: {pool_size} hypotheses waiting."
        )

        old = _os.environ.get("NTFY_TOPIC")
        if topic:
            _os.environ["NTFY_TOPIC"] = topic
        try:
            send_notification(
                Notification(
                    title=f"ResearchForge morning briefing: {slug}",
                    message=msg,
                    channel=Channel.NTFY,
                    tags=["bar_chart"],
                    priority="low",
                )
            )
        finally:
            if topic:
                if old is None:
                    _os.environ.pop("NTFY_TOPIC", None)
                else:
                    _os.environ["NTFY_TOPIC"] = old
    except Exception:
        pass


@kaggle_app.command("watch")
def watch_cmd(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    interval_hours: Annotated[
        float, typer.Option("--interval-hours", help="How often to check for new kernels")
    ] = 6.0,
    once: Annotated[bool, typer.Option("--once", help="Run one cycle and exit")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be injected but don't persist")
    ] = False,
) -> None:
    """Watch for new public kernels and inject their ideas into the experiment pool.

    Runs as a background process (laptop must stay on).
    New kernels are checked every --interval-hours.
    Sends a notification if a kernel with a significantly better score appears.

    Example:
        researchforge kaggle watch rsna-knee-abnormality-detection --interval-hours 6
    """
    from researchforge.kaggle.kernel_watcher import WatcherConfig, run_watch_cycle, run_watcher_loop
    from researchforge.kaggle.orchestrator import OvernightOrchestrator, OvernightSetup

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)

    # Load baseline score from setup if available
    baseline_score = None
    setup_path = artifacts_dir / "overnight_setup.json"
    if setup_path.exists():
        try:
            setup = OvernightSetup.load(setup_path)
            baseline_score = setup.baseline_result.cv_score if setup.baseline_result else None
        except Exception:
            pass

    # Create or update watcher config
    config = WatcherConfig.load(artifacts_dir)
    if config is None:
        # Seed seen_kernel_refs with kernels already known (don't re-inject them)
        from researchforge.kaggle.api import KaggleAPIError, list_competition_kernels

        known_refs: list[str] = []
        try:
            known = list_competition_kernels(slug, page_size=50)
            known_refs = [k.ref for k in known]
        except KaggleAPIError:
            pass
        config = WatcherConfig(
            competition_slug=slug,
            interval_hours=interval_hours,
            seen_kernel_refs=known_refs,
        )
        config.save(artifacts_dir)
    else:
        config.interval_hours = interval_hours
        config.save(artifacts_dir)

    ai_provider = _project_provider(artifacts_dir, "kernel_ideas")

    if once or dry_run:
        try:
            result = run_watch_cycle(artifacts_dir, ai_provider, baseline_score, dry_run=dry_run)
        except Exception as exc:
            typer.secho(f"❌ Watcher cycle failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
        label = "(dry-run) " if dry_run else ""
        typer.echo(f"{label}New kernels found: {result.new_kernels_found}")
        if result.new_ideas_extracted:
            typer.echo("Ideas extracted:")
            for idea in result.new_ideas_extracted:
                typer.echo(f"  • {idea}")
        if result.better_kernel:
            bk = result.better_kernel
            typer.echo(
                f"⚡ Better kernel: {bk['title']} (LB {bk['lb_score']:.4f}, +{bk['delta']:.4f})"
            )
        return

    typer.echo(f"Watching {slug} every {interval_hours}h. Ctrl+C to stop. Keep this terminal open.")
    run_watcher_loop(artifacts_dir, ai_provider, baseline_score)


@kaggle_app.command("ideation")
def ideation(
    competition: Annotated[
        str,
        typer.Argument(help="Kaggle competition slug or URL"),
    ],
) -> None:
    """Research-only mode: read competition, kernels, papers → prioritised plan.

    No data download, no kernel push, no benchmark execution.
    Outputs ideation_plan.md and ideation_plan.json to .overnight_<slug>/.

    Use this when:
      - The dataset is too large to run locally
      - The competition deadline is imminent (need ideas fast)
      - You want to understand the landscape before committing GPU time

    Example:
        researchforge kaggle ideation rsna-knee-abnormality-detection
    """
    import json as _json

    from researchforge.ai.providers import get_provider
    from researchforge.kaggle.competition_reader import read_competition_metadata
    from researchforge.kaggle.credentials import validate_credentials
    from researchforge.kaggle.idea_sources import fetch_sources
    from researchforge.kaggle.ideation import run_ideation
    from researchforge.kaggle.notebook_baseline import (
        find_best_public_kernel,
    )
    from researchforge.kaggle.orchestrator import OvernightOrchestrator

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\n🔍 Ideation-only overnight: {slug}")
    typer.echo("   (research only — no execution, no data download)")
    typer.echo("=" * 60)

    # Credentials needed for kernel listing
    typer.echo("\n1️⃣  Checking Kaggle credentials...")
    if not validate_credentials():
        from researchforge.kaggle.credentials import setup_credentials_interactive

        if not setup_credentials_interactive():
            typer.secho("❌ Credentials required.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    typer.echo("✅ Credentials valid")

    # AI provider
    ai_provider = metered(get_provider(), artifacts_dir, "ideation")
    if ai_provider:
        typer.echo(f"\n2️⃣  AI provider: {ai_provider.name}")
    else:
        typer.echo(
            "\n2️⃣  No AI provider — will list raw ideas (set ANTHROPIC_API_KEY for ranked plan "
            "with code)"
        )

    # Competition metadata
    typer.echo("\n3️⃣  Reading competition metadata...")
    meta = read_competition_metadata(slug, artifacts_dir, ai_provider=ai_provider)
    typer.echo(f"✅ {meta.title or slug}")
    typer.echo(
        f"   Task: {meta.task_type}  |  Metric: {meta.metric_name} "
        f"({meta.metric_direction}, {meta.metric_direction_source or 'unknown source'})"
    )

    # Best public kernel + baseline score
    typer.echo("\n4️⃣  Finding best public kernel...")
    kernel = find_best_public_kernel(slug)
    kernel_title = kernel.title if kernel else ""
    kernel_ref = kernel.ref if kernel else ""
    if kernel:
        typer.echo(f"✅ {kernel_title} by {kernel.author}")
    else:
        typer.echo("⚠️  No public kernels found")

    baseline_score = None
    baseline_source = "not_available"
    if kernel_ref:
        typer.echo("\n5️⃣  Extracting baseline score from kernel title...")
        from researchforge.kaggle.notebook_baseline import _extract_lb_score

        baseline_score = _extract_lb_score(kernel_title)
        if baseline_score:
            baseline_source = "lb_score"
            typer.echo(f"✅ Baseline LB score: {baseline_score:.4f} (from title)")
        else:
            typer.echo("⚠️  No score in kernel title — baseline unknown")

    # Idea sources
    typer.echo("\n6️⃣  Gathering ideas (kernels + arXiv)...")
    idea_col = fetch_sources(meta, artifacts_dir, ai_provider=ai_provider)
    typer.echo(f"✅ {len(idea_col.all_ideas)} ideas from {len(idea_col.sources)} sources")

    # Load user answers if setup was already run
    user_answers = {}
    setup_path = artifacts_dir / "overnight_setup.json"
    if setup_path.exists():
        try:
            setup_data = _json.loads(setup_path.read_text())
            ua = setup_data.get("user_answers") or {}
            user_answers = ua.get("answers", {})
        except Exception:
            pass
    user_answers_path = artifacts_dir / "user_answers.json"
    if user_answers_path.exists() and not user_answers:
        try:
            ua = _json.loads(user_answers_path.read_text())
            user_answers = ua.get("answers", {})
        except Exception:
            pass

    # Generate ideation plan
    typer.echo("\n7️⃣  Generating prioritised improvement plan...")
    plan = run_ideation(
        slug=slug,
        artifacts_dir=artifacts_dir,
        ai_provider=ai_provider,
        metadata=meta,
        idea_col=idea_col,
        baseline_score=baseline_score,
        baseline_source=baseline_source,
        kernel_title=kernel_title,
        user_answers=user_answers,
    )

    md_path = artifacts_dir / "ideation_plan.md"
    json_path = artifacts_dir / "ideation_plan.json"

    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"✅ Ideation complete — {len(plan.ideas)} ideas generated")
    typer.echo(f"\n   Plan: {md_path}")
    typer.echo(f"   JSON: {json_path}")

    if plan.what_we_learned:
        typer.echo("\n💡 What we learned:")
        for insight in plan.what_we_learned:
            typer.echo(f"   • {insight}")

    typer.echo("\n📋 Top ideas:")
    for item in plan.ideas[:5]:
        tag = f" [{item.branch_tag}]" if item.branch_tag else ""
        typer.echo(f"   {item.rank}. {item.title}{tag} — {item.expected_impact}")

    typer.echo(f"\n   Open {md_path.name} for the full plan with code sketches.")


@kaggle_app.command("overnight")
def overnight(
    competition: Annotated[
        str,
        typer.Argument(help="Kaggle competition URL or slug"),
    ],
    gpu: Annotated[
        str | None,
        typer.Option(
            "--gpu",
            help=(
                "Where experiments run: local (this machine, using its GPU if it has one), "
                "kaggle (free GPU quota), colab, ssh (after ssh-setup), cloud (after "
                "cloud-gpu-setup) or competition. Without it, setup asks."
            ),
        ),
    ] = None,
    time_budget: Annotated[
        int,
        typer.Option("--time-hours", help="Time budget in hours"),
    ] = 8,
    max_experiments: Annotated[
        int,
        typer.Option("--max-experiments", help="Max experiments to run"),
    ] = 20,
    watch_interval: Annotated[
        float | None,
        typer.Option(
            "--watch-interval-hours",
            help=(
                "Run the background monitor (all enabled sensors: new public "
                "notebooks, submissions, hypothesis pool, ...) every N hours "
                "until this machine restarts. For a persistent service use "
                "`researchforge kaggle scheduler-install`. Default: off."
            ),
        ),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            help=(
                "Start setup over: re-gather competition facts and re-run the baseline. "
                "Without it, an already set-up project resumes where it left off."
            ),
        ),
    ] = False,
    answers_json: Annotated[
        str | None,
        typer.Option(
            "--answers-json",
            help=(
                "Pre-supply domain knowledge and other HIL answers as JSON. "
                "Accepts inline JSON or a file path. "
                'Example: \'{"known_approaches": "LGB + target encoding", '
                '"external_data": "no"}\''
            ),
        ),
    ] = None,
) -> None:
    """Improve your Kaggle score overnight. Just provide the competition URL.

    Auto-detects the right mode:
      - Code competition → full experiment loop on Kaggle GPU (rounds run as
        Kaggle notebooks, the only way such a competition accepts submissions)
      - Data larger than your free disk → ideation plan with code sketches
      - Otherwise → full experiment loop (propose → benchmark → repeat)

    No API key needed — Claude drives the loop using prepare-round / execute-round:
        researchforge kaggle prepare-round <slug>   # get context for next experiment
        # → write .overnight_<slug>/proposal.json
        researchforge kaggle execute-round <slug>   # run it, get result + ntfy push

    Want background monitoring while you experiment?
        researchforge kaggle scheduler-install <slug>   # installs OS-level watcher

    Examples:
        researchforge kaggle overnight https://kaggle.com/competitions/rsna-knee-abnormality-detection
        researchforge kaggle overnight titanic --gpu cpu
    """
    try:
        orchestrator = OvernightOrchestrator(competition)

        # Handle --answers-json: write to user_answers.json before setup runs
        if answers_json:
            import json as _json

            artifacts_dir = orchestrator.artifacts_dir
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            try:
                # Try as file path first, then inline JSON
                p = Path(answers_json)
                if p.exists():
                    raw_answers = _json.loads(p.read_text())
                else:
                    raw_answers = _json.loads(answers_json)
                from researchforge.kaggle.human_questions import UserAnswers

                ua = UserAnswers(answers=raw_answers)
                ua.save(artifacts_dir)
                typer.echo(f"✅ Loaded {len(raw_answers)} answers from --answers-json")
            except (ValueError, _json.JSONDecodeError) as e:
                typer.secho(f"⚠️  --answers-json parse error: {e}", fg=typer.colors.YELLOW, err=True)

        # Already set up: resume instead of repeating setup (and the baseline).
        # --refresh is the explicit way to start setup over.
        if not refresh and (orchestrator.artifacts_dir / "overnight_setup.json").is_file():
            if gpu and gpu.strip().lower() != "auto":
                # A set-up project switches compute the way `kaggle compute` does,
                # with the same data-policy check, rather than ignoring --gpu.
                from researchforge.kaggle.control_cli import compute_cmd

                compute_cmd(orchestrator.slug, gpu)
            _print_resume(orchestrator.artifacts_dir)
            _ensure_dashboard()
            return

        # Handle --refresh: delete cached metadata so it's re-gathered
        if refresh:
            artifacts_dir = orchestrator.artifacts_dir
            for cache_file in ["competition_metadata.json", "facts.json", "idea_sources.json"]:
                p = artifacts_dir / cache_file
                if p.exists():
                    p.unlink()
                    typer.echo(f"🗑  Cleared cache: {cache_file}", err=True)

        results = orchestrator.run_full_flow(
            gpu_preference=gpu,
            time_budget_hours=time_budget,
            max_experiments=max_experiments,
        )
        # If watcher requested but we ran ideation (no loop to embed it in), start background
        if watch_interval and watch_interval > 0 and results.get("mode") == "ideation":
            _start_watcher_background(orchestrator.slug, orchestrator.artifacts_dir, watch_interval)
        _ensure_dashboard()
    except OvernightError as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


def _print_resume(artifacts_dir: Path) -> None:
    """What an already set-up project has done so far, and the command that continues it."""
    from researchforge.kaggle.control_cli import build_status

    st = build_status(artifacts_dir)
    best = st["best"]
    h = st["hypotheses"]
    typer.secho(f"\n▶  Resuming {st['slug']} — already set up.", fg=typer.colors.GREEN, bold=True)
    baseline = st["baseline"] if st["baseline"] is not None else "not measured"
    typer.echo(f"   Baseline {baseline} ({st['metric']}, {st['direction']})")
    typer.echo(
        f"   {st['rounds_completed']} overnight rounds so far; "
        + (f"best R{best['round']} = {best['metric']}" if best else "no scored round yet")
    )
    repo = st["repo_project"]
    if repo and repo["experiments"]:
        typer.echo(
            f"   {repo['experiments']} experiments in the repo project here — "
            f"see them with: {repo['next_command']}"
        )
    typer.echo(f"   {h['untested']} untested ideas in the pool")
    if st["paused"]:
        typer.echo(f"   ⏸  Paused — continue with: researchforge kaggle resume {st['slug']}")
    if st["open_questions"]:
        typer.echo(f"   {len(st['open_questions'])} question(s) wait on you: kaggle status")
    typer.echo(f"\n   Next: {st['next_command']}")
    typer.echo("   (To redo setup and the baseline from scratch, add --refresh.)")


def _ensure_dashboard() -> None:
    """Start the background dashboard if it isn't running and print its URL.

    Opt out with RESEARCHFORGE_NO_DASHBOARD=1.
    """
    import os as _os

    if _os.environ.get("RESEARCHFORGE_NO_DASHBOARD"):
        return
    try:
        from researchforge.kaggle.dashboard import start_daemon

        port = start_daemon()
        typer.echo(f"\n📊 Dashboard: http://localhost:{port}")
    except Exception as exc:
        typer.echo(
            f"\n📊 Dashboard not started ({exc}). Run: researchforge kaggle dashboard --daemon"
        )


def _start_watcher_background(slug: str, artifacts_dir: Path, interval_hours: float) -> None:
    """Start `researchforge kaggle monitor` as a detached background process."""
    import subprocess as sp
    import sys

    from researchforge.kaggle.kernel_watcher import WatcherConfig
    from researchforge.kaggle.scheduler import SchedulerConfig

    # Seed SchedulerConfig (migrate from WatcherConfig if present)
    sched = SchedulerConfig.load(artifacts_dir)
    watcher_legacy = WatcherConfig.load(artifacts_dir)

    if sched is None:
        # Kernels the old watcher already reported stay "seen", so the first
        # cycle does not re-announce them as new.
        known_refs: list[str] = list(watcher_legacy.seen_kernel_refs) if watcher_legacy else []
        if not known_refs:
            try:
                from researchforge.kaggle.api import list_competition_kernels

                known_refs = [k.ref for k in list_competition_kernels(slug, page_size=50)]
            except Exception:
                pass
        sched = SchedulerConfig(
            competition_slug=slug,
            watcher_enabled=True,
            watcher_interval_hours=interval_hours,
            watcher_seen_kernel_refs=known_refs,
        )
    else:
        sched.watcher_interval_hours = interval_hours
        sched.watcher_enabled = True
        sched.enabled = True

    # Inherit ntfy topic from env if available
    import os

    sched.ntfy_topic = os.environ.get("NTFY_TOPIC", "")
    sched.save(artifacts_dir)

    # The same monitor the launchd job runs, so the sensors honour
    # scheduler_config.json and record their status for the dashboard.
    log_dir = Path.home() / ".local" / "share" / "researchforge" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / f"monitor_{slug}.log").open("a", encoding="utf-8")
        env = {**os.environ, "RESEARCHFORGE_OVERNIGHT_DIR": str(artifacts_dir)}
        sp.Popen(
            [
                sys.executable,
                "-m",
                "researchforge",
                "kaggle",
                "monitor",
                slug,
                "--poll-hours",
                str(interval_hours),
            ],
            cwd=str(artifacts_dir.parent),
            env=env,
            stdin=sp.DEVNULL,
            stdout=log,
            stderr=sp.STDOUT,
            start_new_session=True,
        )
        log.close()
        typer.echo(
            f"\n👁  Monitor started (all enabled sensors every {interval_hours}h, "
            f"log: {log_dir / f'monitor_{slug}.log'}). It stops when this machine "
            f"restarts; `researchforge kaggle scheduler-install {slug}` keeps it running."
        )
    except Exception as exc:
        typer.echo(f"\n👁  Monitor not started ({exc}). Run: researchforge kaggle monitor {slug}")


@kaggle_app.command("ssh-setup")
def ssh_setup(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    host: Annotated[str, typer.Option("--host", help="SSH target: user@hostname or hostname")] = "",
    key: Annotated[
        str, typer.Option("--key", help="Path to SSH private key (default: ~/.ssh/id_rsa)")
    ] = "",
    remote_dir: Annotated[
        str,
        typer.Option("--remote-dir", help="Base directory on remote server for experiments"),
    ] = "~/researchforge_experiments",
    python_path: Annotated[
        str,
        typer.Option("--python", help="Python executable on remote server"),
    ] = "python",
    rsync_data: Annotated[
        bool,
        typer.Option(
            "--rsync-data/--no-rsync-data",
            help="Copy competition data to remote server before each experiment",
        ),
    ] = False,
    test: Annotated[
        bool,
        typer.Option("--test/--no-test", help="Test SSH connectivity after saving config"),
    ] = True,
) -> None:
    """Configure a remote SSH server as a compute target for overnight experiments.

    After setup, use --gpu ssh with the overnight command to run experiments there.

    Examples:
        researchforge kaggle ssh-setup titanic --host user@gpu-box.example.com
        researchforge kaggle ssh-setup titanic --host gpu-box --key ~/.ssh/my_key --rsync-data
    """
    from researchforge.kaggle.orchestrator import OvernightOrchestrator
    from researchforge.kaggle.ssh_config import SSHConfig

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not host:
        typer.secho(
            "❌ --host is required. Example: --host user@myserver.com",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    cfg = SSHConfig(
        host=host,
        key_path=key,
        remote_dir=remote_dir,
        python_path=python_path,
        rsync_data=rsync_data,
    )
    cfg.save(artifacts_dir)
    typer.echo(f"✅ SSH config saved → {artifacts_dir}/ssh_config.json")
    typer.echo(f"   host       : {cfg.host}")
    typer.echo(f"   key        : {cfg.key_path or '(default SSH key)'}")
    typer.echo(f"   remote_dir : {cfg.remote_dir}")
    typer.echo(f"   python     : {cfg.python_path}")
    typer.echo(f"   rsync_data : {cfg.rsync_data}")

    if test:
        typer.echo("\n🔗 Testing SSH connection...")
        ok, msg = cfg.test_connection()
        if ok:
            typer.secho(f"✅ {msg}", fg=typer.colors.GREEN)
            py_ver = cfg.remote_python_version()
            if py_ver:
                typer.echo(f"   Python on remote: {py_ver}")
        else:
            typer.secho(f"⚠️  {msg}", fg=typer.colors.YELLOW)
            typer.echo(
                "   Connection test failed — check host, key, and that sshd is running.\n"
                "   The config was saved anyway; fix the issue and retry with --test."
            )

    typer.echo(
        f"\nTo run experiments on this server:\n  researchforge kaggle overnight {slug} --gpu ssh"
    )


@kaggle_app.command("cloud-gpu-setup")
def cloud_gpu_setup(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    provider: Annotated[
        str,
        typer.Option("--provider", help="Cloud GPU provider: runpod, lambda, vast or aws"),
    ] = "runpod",
    api_key_env: Annotated[
        str,
        typer.Option(
            "--api-key-env", help="Env var name that holds the API key (e.g. RUNPOD_API_KEY)"
        ),
    ] = "",
    gpu_type: Annotated[
        str,
        typer.Option(
            "--gpu-type", help="GPU type/size to provision (leave blank to use cheapest available)"
        ),
    ] = "",
    region: Annotated[
        str,
        typer.Option("--region", help="Cloud region (Lambda Labs only, e.g. us-west-2)"),
    ] = "",
    python_path: Annotated[
        str,
        typer.Option("--python", help="Python executable on the provisioned instance"),
    ] = "python",
    rsync_data: Annotated[
        bool,
        typer.Option(
            "--rsync-data/--no-rsync-data",
            help="Copy competition data to the instance before each experiment",
        ),
    ] = False,
    list_gpus: Annotated[
        bool,
        typer.Option("--list-gpus", help="List available GPU types for the provider and exit"),
    ] = False,
    image: Annotated[
        str,
        typer.Option("--image", help="Docker image (RunPod/vast) or AMI id (AWS, required)"),
    ] = "",
    max_price: Annotated[
        float,
        typer.Option("--max-price", help="Max $/hr (vast.ai offer filter; 0 = no limit)"),
    ] = 0.0,
    ssh_key_name: Annotated[
        str, typer.Option("--ssh-key-name", help="AWS: EC2 key pair name (required)")
    ] = "",
    security_group_id: Annotated[
        str,
        typer.Option("--security-group-id", help="AWS: security group allowing SSH (required)"),
    ] = "",
) -> None:
    """Configure a pay-per-hour cloud GPU as a compute target.

    Supported providers: RunPod (RUNPOD_API_KEY), Lambda Labs (LAMBDA_API_KEY), vast.ai
    (VAST_API_KEY) and AWS EC2 (the aws CLI with your normal AWS credentials).
    Each execute-round provisions a fresh instance, runs the experiment, and terminates it.

    Examples:
        export RUNPOD_API_KEY=your_key
        researchforge kaggle cloud-gpu-setup titanic --provider runpod
        researchforge kaggle cloud-gpu-setup titanic --provider lambda --gpu-type gpu_1x_a10
        researchforge kaggle cloud-gpu-setup titanic --provider runpod --list-gpus
    """

    from researchforge.kaggle.cloud_gpu_config import CloudGPUConfig, list_gpu_types
    from researchforge.kaggle.orchestrator import OvernightOrchestrator

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Resolve API key env var
    if provider not in ("runpod", "lambda", "vast", "aws"):
        typer.secho(
            f"❌ Unknown provider '{provider}'. Use runpod, lambda, vast or aws.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    if not api_key_env and provider != "aws":
        api_key_env = {
            "runpod": "RUNPOD_API_KEY",
            "lambda": "LAMBDA_API_KEY",
            "vast": "VAST_API_KEY",
        }[provider]

    api_key = CloudGPUConfig(provider=provider, api_key_env=api_key_env).api_key()

    if list_gpus:
        if not api_key:
            typer.secho(
                f"❌ Set {api_key_env} to list available GPUs (dashboard → Keys & accounts, "
                f"or: researchforge kaggle keys set {provider}).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Fetching GPU types from {provider}...")
        gpus = list_gpu_types(provider, api_key)
        if not gpus:
            typer.echo("No GPU types returned (check API key / connectivity).")
        for g in sorted(gpus, key=lambda x: x.get("price_per_hr", 999)):
            price = f"${g.get('price_per_hr', 0):.3f}/hr"
            name = g.get("name", "")
            display = g.get("display", name)
            mem = f" {g.get('memory_gb', '')}GB" if g.get("memory_gb") else ""
            avail = "" if g.get("available", True) else "  [unavailable]"
            typer.echo(f"  {price}  {name:<40} {display}{mem}{avail}")
        return

    cfg = CloudGPUConfig(
        provider=provider,
        api_key_env=api_key_env,
        gpu_type=gpu_type,
        region=region,
        python_path=python_path,
        rsync_data=rsync_data,
        base_image=image,
        max_price_per_hr=max_price,
        ssh_key_name=ssh_key_name,
        security_group_id=security_group_id,
    )
    if provider == "aws" and not (image and ssh_key_name and security_group_id):
        typer.secho(
            "❌ AWS needs --image (a GPU AMI), --ssh-key-name and --security-group-id.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    cfg.save(artifacts_dir)
    typer.echo(f"✅ Cloud GPU config saved → {artifacts_dir}/cloud_gpu_config.json")
    typer.echo(f"   provider   : {cfg.provider}")
    typer.echo(f"   api_key_env: {cfg.api_key_env}")
    typer.echo(f"   gpu_type   : {cfg.gpu_type or '(cheapest available)'}")
    typer.echo(f"   python     : {cfg.python_path}")
    typer.echo(f"   rsync_data : {cfg.rsync_data}")

    if not api_key:
        missing = (
            "the aws CLI is not installed" if provider == "aws" else f"{api_key_env} is not set"
        )
        typer.secho(
            f"\n⚠️  {missing} — fix that before running experiments.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo(f"\n✅ {api_key_env} is set")
        if gpu_type:
            typer.echo(f"   Will request GPU: {gpu_type}")
        else:
            typer.echo(
                f"   GPU type not specified — will use cheapest available {provider} GPU.\n"
                f"   Run --list-gpus to see options and pick one with --gpu-type."
            )

    typer.echo(
        f"\nTo run experiments on cloud GPU:\n  researchforge kaggle overnight {slug} --gpu cloud"
    )


@kaggle_app.command("scheduler-setup")
def scheduler_setup(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    watcher_hours: Annotated[
        float,
        typer.Option("--watcher-hours", help="How often to check for new public notebooks (0=off)"),
    ] = 6.0,
    start_at: Annotated[
        str,
        typer.Option(
            "--start-at", help="Start experiments at this time today (HH:MM local), e.g. 23:00"
        ),
    ] = "",
    briefing_at: Annotated[
        str,
        typer.Option(
            "--briefing-at", help="Send morning briefing at this time (HH:MM local), e.g. 07:00"
        ),
    ] = "",
    briefing_on_completion: Annotated[
        bool,
        typer.Option(
            "--briefing-on-completion/--no-briefing-on-completion",
            help="Send ntfy notification immediately when experiments finish",
        ),
    ] = True,
    briefing_only_if_improved: Annotated[
        bool,
        typer.Option("--only-if-improved", help="Skip briefing if nothing improved"),
    ] = False,
) -> None:
    """Configure the overnight scheduler: watcher interval, start time, morning briefing.

    Examples:
        # Watch every 6h, send briefing at 7am
        researchforge kaggle scheduler-setup titanic --watcher-hours 6 --briefing-at 07:00

        # Start experiments at 11pm, notify immediately on completion
        researchforge kaggle scheduler-setup titanic --start-at 23:00 --briefing-on-completion

        # Disable watcher, just send morning briefing at 8am
        researchforge kaggle scheduler-setup titanic --watcher-hours 0 --briefing-at 08:00
    """
    import os as _os

    from researchforge.kaggle.orchestrator import OvernightOrchestrator
    from researchforge.kaggle.scheduler import SchedulerConfig

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load existing or create new
    sched = SchedulerConfig.load(artifacts_dir) or SchedulerConfig(competition_slug=slug)

    sched.watcher_enabled = watcher_hours > 0
    sched.watcher_interval_hours = watcher_hours if watcher_hours > 0 else 6.0
    sched.ntfy_topic = _os.environ.get("NTFY_TOPIC", sched.ntfy_topic)

    if start_at:
        sched.experiment_start_mode = "at_time"
        sched.experiment_start_time = start_at
    if briefing_at and briefing_on_completion:
        sched.briefing_mode = "both"
        sched.briefing_time = briefing_at
    elif briefing_at:
        sched.briefing_mode = "at_time"
        sched.briefing_time = briefing_at
    elif briefing_on_completion:
        sched.briefing_mode = "on_completion"
    else:
        sched.briefing_mode = "never"

    sched.briefing_only_if_improved = briefing_only_if_improved
    sched.save(artifacts_dir)

    typer.echo(f"\n✅ Scheduler configured for {slug}:")
    typer.echo(
        "   Kernel watcher : "
        + (f"every {sched.watcher_interval_hours}h" if sched.watcher_enabled else "disabled")
    )
    typer.echo(
        f"   Experiments    : {sched.experiment_start_mode}"
        + (f" at {sched.experiment_start_time}" if sched.experiment_start_time else "")
    )
    typer.echo(
        f"   Briefing       : {sched.briefing_mode}"
        + (f" at {sched.briefing_time}" if sched.briefing_time else "")
    )
    typer.echo(f"   ntfy topic     : {sched.ntfy_topic or '(not set — set NTFY_TOPIC env var)'}")
    typer.echo(f"\n   Start daemon with: researchforge kaggle monitor {slug}")


def _run_watcher_daemon(slug: str, artifacts_dir_str: str, interval_hours: float) -> None:
    """Entry point for background scheduler process (legacy compat)."""
    from researchforge.kaggle.scheduler import run_scheduler_daemon

    run_scheduler_daemon(slug, artifacts_dir_str)


@kaggle_app.command("scheduler-install")
def scheduler_install(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    interval_hours: Annotated[
        float,
        typer.Option(
            "--interval-hours",
            help="How often to poll (hours). Default: uses saved scheduler config or 6h.",
        ),
    ] = 0.0,
    uninstall: Annotated[
        bool,
        typer.Option("--uninstall", help="Remove the service instead of installing it."),
    ] = False,
) -> None:
    """Install a background service so the monitor runs on a schedule without Claude Code.

    launchd on macOS, a systemd user timer on Linux, Task Scheduler on Windows. The
    service runs the sensors and sends notifications even when the editor is closed;
    the machine itself must be on.

    Install:
        researchforge kaggle scheduler-install titanic
        researchforge kaggle scheduler-install titanic --interval-hours 4

    Remove:
        researchforge kaggle scheduler-install titanic --uninstall
        # or:
        researchforge kaggle scheduler-uninstall titanic
    """
    import os as _os
    import subprocess as _sp
    import sys as _sys

    from researchforge.kaggle.orchestrator import OvernightOrchestrator
    from researchforge.kaggle.scheduler import SchedulerConfig

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    label = f"com.researchforge.watcher.{slug}"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"

    # ── Uninstall path ────────────────────────────────────────────────────
    if uninstall and _sys.platform != "darwin":
        from researchforge.kaggle import service_install

        removed, detail = service_install.uninstall(_sys.platform, slug)
        typer.echo(f"✅ Removed {detail}" if removed else f"No service installed ({detail})")
        return
    if uninstall:
        if plist_path.exists():
            _sp.run(
                ["launchctl", "unload", "-w", str(plist_path)], capture_output=True, check=False
            )
            plist_path.unlink()
            typer.secho(f"✅ Removed launchd service: {label}", fg=typer.colors.GREEN)
        else:
            typer.echo(f"No service installed for {slug} (looked for {plist_path})")
        return

    # ── Install path ──────────────────────────────────────────────────────
    sched = SchedulerConfig.load(artifacts_dir) if artifacts_dir.exists() else None
    hours = interval_hours or (sched.watcher_interval_hours if sched else 6.0)
    interval_secs = int(hours * 3600)

    # The base interpreter, not a venv shim: a venv python reads pyvenv.cfg
    # from the venv dir at startup, which macOS TCC blocks under ~/Desktop
    # (seen as "PermissionError ... pyvenv.cfg", launchd exit 78).
    python_exe = getattr(_sys, "_base_executable", "") or _sys.executable
    # launchd opens StandardOutPath itself before exec; under ~/Desktop TCC
    # refuses that and the job dies with EX_CONFIG (78) without logging a line.
    log_dir = Path.home() / ".local" / "share" / "researchforge" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"scheduler_{slug}.log"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if _sys.platform != "darwin":
        _install_service_non_macos(slug, artifacts_dir, hours, log_path)
        return

    # macOS TCC blocks launchd background agents from accessing ~/Desktop by
    # default.  Work around this by writing a small launcher script to
    # ~/.local/share/researchforge/ (which is always accessible) that:
    #   1. Adds the venv's site-packages and this package's source root to sys.path
    #   2. Pins the artifacts dir (so the job never writes to a different copy)
    #   3. Changes directory to the project root and runs the monitor
    launcher_dir = Path.home() / ".local" / "share" / "researchforge" / "launchers"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = launcher_dir / f"monitor_{slug}.py"

    import researchforge as _rf_pkg

    py_ver = f"{_sys.version_info.major}.{_sys.version_info.minor}"
    site_packages = str(
        Path(_sys.executable).parent.parent / "lib" / f"python{py_ver}" / "site-packages"
    )
    package_root = str(Path(_rf_pkg.__file__).resolve().parent.parent)
    cwd = str(artifacts_dir.parent)

    launcher_code = f"""# Auto-generated launcher for researchforge kaggle monitor {slug}
# Located outside ~/Desktop to avoid macOS TCC sandbox restrictions.
import sys, os
sys.path.insert(0, {site_packages!r})
sys.path.insert(0, {package_root!r})
os.environ['RESEARCHFORGE_OVERNIGHT_DIR'] = {str(artifacts_dir)!r}
os.chdir({cwd!r})
sys.argv = ['researchforge', 'kaggle', 'monitor', {slug!r}, '--poll-hours', {str(hours)!r}]
from researchforge.cli import app
app(standalone_mode=True)
"""
    launcher_path.write_text(launcher_code, encoding="utf-8")
    launcher_path.chmod(0o755)

    # Inherit the current environment
    env_vars = {
        k: v
        for k, v in _os.environ.items()
        if k
        in (
            "PATH",
            "HOME",
            "KAGGLE_USERNAME",
            "KAGGLE_KEY",
            "NTFY_TOPIC",
            "NTFY_SERVER",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "RESEARCHFORGE_LLM",
            "OLLAMA_HOST",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
        )
        and v
    }
    env_vars.setdefault("PATH", str(Path(python_exe).parent) + ":/usr/local/bin:/usr/bin:/bin")
    env_vars["HOME"] = str(Path.home())

    env_xml = "\n".join(
        f"        <key>{k}</key>\n        <string>{v}</string>" for k, v in env_vars.items()
    )

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_exe}</string>
        <string>{launcher_path}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
{env_xml}
    </dict>
    <key>StartInterval</key>
    <integer>{interval_secs}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
    <key>WorkingDirectory</key>
    <string>{str(Path.home() / ".local" / "share" / "researchforge")}</string>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>60</integer>
</dict>
</plist>
"""

    plist_path.parent.mkdir(parents=True, exist_ok=True)

    # Unload existing version before replacing
    if plist_path.exists():
        _sp.run(["launchctl", "unload", "-w", str(plist_path)], capture_output=True, check=False)

    plist_path.write_text(plist_content)
    result = _sp.run(
        ["launchctl", "load", "-w", str(plist_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        typer.secho(
            f"❌ launchctl load failed: {result.stderr.strip()}\n"
            f"   Plist written to {plist_path} — fix manually and reload.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.secho(f"\n✅ Installed launchd service: {label}", fg=typer.colors.GREEN)
    typer.echo(f"   Polls every     : {hours}h ({interval_secs}s)")
    typer.echo("   Runs at load    : yes (fires once immediately)")
    typer.echo(f"   Log file        : {log_path}")
    typer.echo(f"   Plist           : {plist_path}")
    typer.echo("\n   The watcher now runs 24/7 — survives Claude Code being closed.")
    typer.echo(f"   Check status    : launchctl list {label}")
    typer.echo(f"   View logs       : tail -f {log_path}")
    typer.echo(f"   Remove service  : researchforge kaggle scheduler-uninstall {slug}")


def _install_service_non_macos(
    slug: str, artifacts_dir: Path, hours: float, log_path: Path
) -> None:
    """scheduler-install on Linux (systemd --user timer) and Windows (Task Scheduler)."""
    import os as _os
    import sys as _sys
    import sysconfig

    import researchforge as _rf_pkg
    from researchforge.kaggle import service_install

    launcher_dir = Path.home() / ".local" / "share" / "researchforge" / "launchers"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = launcher_dir / f"monitor_{slug}.py"
    env = {
        k: v
        for k, v in _os.environ.items()
        if k
        in (
            "KAGGLE_USERNAME",
            "KAGGLE_KEY",
            "NTFY_TOPIC",
            "NTFY_SERVER",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "RESEARCHFORGE_LLM",
            "OLLAMA_HOST",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
        )
        and v
    }
    package_root = str(Path(_rf_pkg.__file__).resolve().parent.parent)
    # Windows has no StandardOutput= equivalent, so the launcher appends to the log itself.
    redirect = (
        f"_log = open({str(log_path)!r}, 'a', encoding='utf-8', buffering=1)\n"
        "sys.stdout = sys.stderr = _log\n"
        if _sys.platform.startswith("win")
        else ""
    )
    launcher_path.write_text(
        f"""# Auto-generated launcher for researchforge kaggle monitor {slug}
import sys, os
sys.path.insert(0, {sysconfig.get_paths()["purelib"]!r})
sys.path.insert(0, {package_root!r})
os.environ.update({env!r})
os.environ['RESEARCHFORGE_OVERNIGHT_DIR'] = {str(artifacts_dir)!r}
os.chdir({str(artifacts_dir.parent)!r})
{redirect}sys.argv = [
    'researchforge', 'kaggle', 'monitor', {slug!r}, '--poll-hours', {str(hours)!r}
]
from researchforge.cli import app
app(standalone_mode=True)
""",
        encoding="utf-8",
    )
    launcher_path.chmod(0o700)  # it holds the API keys the monitor needs

    ok, detail = service_install.install(
        _sys.platform, slug, _sys.executable, launcher_path, hours, log_path
    )
    if not ok:
        typer.secho(f"❌ {detail}", fg=typer.colors.RED, err=True)
        typer.echo(f"   Launcher written to {launcher_path} — you can run it from cron instead.")
        raise typer.Exit(1)
    typer.secho(f"\n✅ Installed {detail}", fg=typer.colors.GREEN)
    typer.echo(f"   Polls every     : {hours}h")
    typer.echo(f"   Log file        : {log_path}")
    typer.echo(f"   Remove service  : researchforge kaggle scheduler-uninstall {slug}")


@kaggle_app.command("scheduler-uninstall")
def scheduler_uninstall(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
) -> None:
    """Remove the launchd watcher service for a competition.

    This stops the background watcher from running. You can reinstall it later
    with `researchforge kaggle scheduler-install <slug>`.
    """
    from researchforge.kaggle.orchestrator import OvernightOrchestrator

    slug = OvernightOrchestrator._parse_slug(competition)
    import sys as _sys

    if _sys.platform != "darwin":
        from researchforge.kaggle import service_install

        removed, detail = service_install.uninstall(_sys.platform, slug)
        typer.echo(f"✅ Removed {detail}" if removed else f"No service installed ({detail})")
        return
    label = f"com.researchforge.watcher.{slug}"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"

    if not plist_path.exists():
        typer.echo(f"No launchd service found for {slug}.")
        return

    import subprocess as _sp

    _sp.run(["launchctl", "unload", "-w", str(plist_path)], capture_output=True, check=False)
    plist_path.unlink()
    typer.secho(f"✅ Removed: {label}", fg=typer.colors.GREEN)
    typer.echo("   (watcher will no longer run in the background)")


@kaggle_app.command("sensors")
def sensors_cmd(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    enable: Annotated[
        list[str] | None, typer.Option("--enable", help="Sensor name to enable (repeatable)")
    ] = None,
    disable: Annotated[
        list[str] | None, typer.Option("--disable", help="Sensor name to disable (repeatable)")
    ] = None,
    add: Annotated[str, typer.Option("--add", help="Name of a custom sensor to add")] = "",
    command: Annotated[
        str, typer.Option("--command", help="Shell command for --add; exit 0 = healthy")
    ] = "",
    timeout: Annotated[
        int, typer.Option("--timeout", help="Timeout (s) for a custom sensor command")
    ] = 120,
    remove: Annotated[str, typer.Option("--remove", help="Name of a custom sensor to remove")] = "",
    json_output: Annotated[
        bool, typer.Option("--json", help="Print sensor config + state as JSON")
    ] = False,
) -> None:
    """List or configure the monitor's sensors (persisted in scheduler_config.json).

    Built-in: kernel_watcher_sensor, submission_status_sensor, hypothesis_pool_sensor,
    experiment_heartbeat_sensor, arxiv_scanner_sensor, morning_briefing_sensor,
    auto_run_sensor (off by default; configure with `kaggle auto-run`).

    Examples:
        researchforge kaggle sensors titanic
        researchforge kaggle sensors titanic --disable arxiv_scanner_sensor
        researchforge kaggle sensors titanic --add disk_space --command "df -h . | tail -1"
    """
    from researchforge.kaggle import sensors as _sensors

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in enable or []:
            _sensors.set_enabled(artifacts_dir, slug, name, True)
        for name in disable or []:
            _sensors.set_enabled(artifacts_dir, slug, name, False)
        if add:
            _sensors.add_custom(artifacts_dir, slug, add, command, timeout)
        if remove and not _sensors.remove_custom(artifacts_dir, slug, remove):
            typer.secho(f"No custom sensor named {remove}", fg=typer.colors.YELLOW, err=True)
    except (KeyError, ValueError) as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    cfg = _sensors.load_config(artifacts_dir, slug)
    runtime = (read_json(artifacts_dir / "monitor_state.json", {}) or {}).get("sensors", {})
    rows = [
        {
            "name": b["name"],
            "custom": False,
            "enabled": _sensors.is_enabled(cfg, b["name"]),
            **runtime.get(b["name"], {}),
        }
        for b in _sensors.BUILTIN_SENSORS
    ] + [
        {
            "name": c["name"],
            "custom": True,
            "enabled": c.get("enabled", True),
            "command": c.get("command", ""),
            **runtime.get(c["name"], {}),
        }
        for c in cfg.custom_sensors
    ]
    if json_output:
        typer.echo(json.dumps({"artifacts_dir": str(artifacts_dir), "sensors": rows}, indent=2))
        return
    typer.echo(f"Sensors for {slug} ({artifacts_dir})")
    for r in rows:
        flag = "on " if r["enabled"] else "off"
        last = r.get("last_run_at", "never run")
        status = r.get("status", "—")
        typer.echo(
            f"  [{flag}] {r['name']:<30} {status:<9} {last}{'  (custom)' if r['custom'] else ''}"
        )


@kaggle_app.command("setup")
def setup(
    competition: Annotated[
        str,
        typer.Argument(help="Kaggle competition slug or URL"),
    ],
) -> None:
    """Setup phase only (for debugging)."""
    try:
        orchestrator = OvernightOrchestrator(competition)
        orchestrator.setup()
    except OvernightError as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


@kaggle_app.command("run")
def run(
    competition: Annotated[
        str,
        typer.Argument(help="Kaggle competition slug or URL"),
    ],
    rounds: Annotated[
        int | None,
        typer.Option("--rounds", help="Stop after this many rounds (default: setup's maximum)"),
    ] = None,
) -> None:
    """Run experiments with an AI API key (after setup)."""
    try:
        from researchforge.kaggle.orchestrator import OvernightSetup

        orchestrator = OvernightOrchestrator(competition)
        # Load setup from artifacts directory
        artifacts_dir = orchestrator.artifacts_dir
        setup_path = artifacts_dir / "overnight_setup.json"

        if not setup_path.exists():
            typer.secho(
                f"❌ Setup not found at {setup_path}\n"
                f"   Please run 'researchforge kaggle overnight {competition}' first",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        from researchforge.kaggle.auto_run import api_provider
        from researchforge.kaggle.experiment_loop import OvernightExperimentLoop

        # The environment's key, or one stored with `kaggle auto-run-key`.
        if api_provider() is None:
            typer.secho(
                "❌ The autonomous loop needs ANTHROPIC_API_KEY, GEMINI_API_KEY, "
                "OPENAI_API_KEY, or a local model via RESEARCHFORGE_LLM=ollama/<model>.\n"
                f"   Without a key, Claude Code drives it: researchforge kaggle prepare-round "
                f"{orchestrator.slug} → proposal.json → researchforge kaggle execute-round "
                f"{orchestrator.slug}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        setup = OvernightSetup.load(setup_path)
        loop = OvernightExperimentLoop(setup)
        summary = loop.run_overnight(setup.time_budget_hours, rounds or setup.max_experiments)
        loop.save_summary(summary)

        typer.echo("\n✅ Experiment loop completed!")
        typer.echo(
            f"   Rounds: {summary.total_rounds_attempted} attempted, "
            f"{summary.rounds_improved} improved"
        )
        typer.echo(f"   Duration: {summary.total_duration_seconds / 3600:.1f}h")
        best = summary.best_round
        if best is not None and best.full_result is not None:
            typer.echo(f"   Best metric: R{best.round_num} {best.full_result.metric_value:.4f}")

    except OvernightError as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


# ---------------------------------------------------------------------------
# Live round state shared with the dashboard
# ---------------------------------------------------------------------------

_INFLIGHT_LOCK = _threading.Lock()


def _set_inflight(artifacts_dir: Path, round_num: int, info: dict[str, Any] | None) -> None:
    """Track rounds currently executing in inflight_rounds.json (None = finished)."""
    with _INFLIGHT_LOCK:
        path = artifacts_dir / "inflight_rounds.json"
        data = read_json(path, {}) or {}
        if info is None:
            data.pop(str(round_num), None)
        else:
            data[str(round_num)] = {
                **data.get(str(round_num), {}),
                **info,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        if data:
            atomic_write_json(path, data)
        else:
            path.unlink(missing_ok=True)


def _mark_hypothesis_tested(
    artifacts_dir: Path, proposal_data: dict[str, Any], round_num: int, outcome: str
) -> str | None:
    """Mark the pool hypothesis a proposal tested. Returns the matched idea, if any.

    Matches the proposal's ``hypothesis`` field (or, failing that, its
    source_credit / title) against pool ideas on a case-insensitive 60-char
    prefix — exact intent, no fuzzy guessing.
    """
    with _INFLIGHT_LOCK:  # run-parallel workers share the pool file
        return _mark_hypothesis_tested_locked(artifacts_dir, proposal_data, round_num, outcome)


def _mark_hypothesis_tested_locked(
    artifacts_dir: Path, proposal_data: dict[str, Any], round_num: int, outcome: str
) -> str | None:
    from researchforge.kaggle.hypothesis_pool import mark_tested

    return mark_tested(artifacts_dir, proposal_data, round_num, outcome)


def _round_outcome(improved: bool, error: str | None) -> str:
    if improved:
        return "improved"
    return "failed" if error else "no_improvement"


def _notify_round_result(setup: Any, result: dict[str, Any]) -> None:
    """Send a per-experiment push notification via ntfy.

    Topic resolution order:
      1. setup.ntfy_topic  (persisted at overnight-setup time from NTFY_TOPIC env)
      2. NTFY_TOPIC env var at execute-round time
      3. Skip silently — user hasn't configured notifications
    """
    import os

    topic = (getattr(setup, "ntfy_topic", "") or "").strip()
    if not topic:
        topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return  # no topic configured — silent skip

    from researchforge.kaggle.control import should_notify

    category = "improvement" if result.get("improved") and not result.get("error") else "result"
    if not should_notify(getattr(setup, "artifacts_dir", None), category):
        return

    try:
        from researchforge.notifications.service import Channel, Notification, send_notification

        slug = getattr(setup, "competition_slug", "?")
        rnum = result.get("round_num", "?")
        title_str = result.get("title", "")
        metric = result.get("full_metric")
        improved = result.get("improved", False)
        error = result.get("error")

        if error:
            body = f"R{rnum} '{title_str}' — FAILED\n{error[:200]}"
            tags = ["x"]
            priority = "default"
        elif improved:
            body = f"R{rnum} '{title_str}' — IMPROVED ✓\nMetric: {metric:.4f}"
            tags = ["white_check_mark"]
            priority = "default"
        else:
            shown = f"{metric:.4f}" if isinstance(metric, (int, float)) else "n/a"
            body = f"R{rnum} '{title_str}' — no improvement\nMetric: {shown}"
            tags = ["arrow_right"]
            priority = "low"

        # Temporarily set NTFY_TOPIC so _ntfy_topic() picks up the right topic
        old = os.environ.get("NTFY_TOPIC")
        os.environ["NTFY_TOPIC"] = topic
        try:
            send_notification(
                Notification(
                    title=f"ResearchForge: {slug}",
                    message=body,
                    channel=Channel.NTFY,
                    priority=priority,
                    tags=tags,
                )
            )
        finally:
            if old is None:
                os.environ.pop("NTFY_TOPIC", None)
            else:
                os.environ["NTFY_TOPIC"] = old
    except Exception:
        pass  # notifications are best-effort; never crash the experiment


def _detect_merge_opportunity(all_rounds: list[Any]) -> dict[str, Any] | None:
    """Return merge opportunity info if ≥2 independent improved branches exist.

    Returns:
        None  — no merge opportunity
        {"parent_a": int, "parent_b": int, "metric_a": float, "metric_b": float}
    """
    # Collect rounds that improved over baseline, grouped by branch_tag
    improved: list[Any] = []
    for r in all_rounds:
        if r.full_result and r.full_result.metric_improved:
            improved.append(r)

    # Need at least 2 improvements
    if len(improved) < 2:
        return None

    # Group by branch_tag; treat empty tag as a unique branch per round_num
    branches: dict[str, list[Any]] = {}
    for r in improved:
        tag = r.branch_tag if r.branch_tag else f"_round_{r.round_num}"
        branches.setdefault(tag, []).append(r)

    # Pick best representative per branch (highest metric)
    best_per_branch: list[Any] = []
    for rounds in branches.values():
        best = max(rounds, key=lambda r: r.full_result.metric_value if r.full_result else 0)
        best_per_branch.append(best)

    if len(best_per_branch) < 2:
        return None

    # Sort by metric value descending, pick top-2 as merge candidates
    best_per_branch.sort(
        key=lambda r: r.full_result.metric_value if r.full_result else 0, reverse=True
    )
    a, b = best_per_branch[0], best_per_branch[1]
    return {
        "parent_a": a.round_num,
        "parent_b": b.round_num,
        "metric_a": a.full_result.metric_value if a.full_result else None,
        "metric_b": b.full_result.metric_value if b.full_result else None,
    }


@kaggle_app.command("prepare-round")
def prepare_round(
    competition: Annotated[
        str,
        typer.Argument(help="Kaggle competition slug or URL"),
    ],
) -> None:
    """Print proposal context for the next round as JSON (Claude Code path — no API key needed).

    Claude reads this output, writes a proposal to .overnight_<slug>/proposal.json,
    then calls execute-round.

    Output includes:
      - Ranked hypothesis pool (what to try next, based on DAG state)
      - Recommended next hypothesis
      - data_path_for_experiments: the ONLY path to use in experiment code
      - Merge opportunity if two independent improvements exist
      - Whether resynthesize is due (every 5 rounds)
    """
    import json
    import sqlite3

    from researchforge.kaggle.compute_policy import ComputePolicy
    from researchforge.kaggle.orchestrator import OvernightSetup

    orchestrator = OvernightOrchestrator(competition)
    artifacts_dir = orchestrator.artifacts_dir
    setup_path = artifacts_dir / "overnight_setup.json"

    if not setup_path.exists():
        typer.secho(
            f"❌ Setup not found. Run 'researchforge kaggle overnight {competition}' first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    setup = OvernightSetup.load(setup_path)

    from researchforge.kaggle import control as _control

    loop_control = _control.LoopControl.load(artifacts_dir)
    if loop_control.paused:
        # Claude reads this and stops; nothing new starts until `kaggle resume`.
        typer.echo(
            json.dumps(
                {
                    "paused": True,
                    "pause_reason": loop_control.pause_reason,
                    "paused_at": loop_control.paused_at,
                    "instruction": "The user paused this loop. Do not write a proposal or "
                    "run anything. It resumes with: researchforge kaggle resume "
                    f"{setup.competition_slug}",
                },
                indent=2,
            )
        )
        return

    # Load compute policy for data path
    policy = ComputePolicy.load(artifacts_dir)
    data_path = policy.experiment_data_path() if policy else str(artifacts_dir / "data")
    internet_available = policy.is_internet_available() if policy else True

    from researchforge.kaggle.patch_proposer import PatchProposer

    proposer = PatchProposer(setup)

    # Determine next round number
    db_path = artifacts_dir / "experiment_log.db"
    round_num = 1
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS experiments (round_num INTEGER PRIMARY KEY, "
                "parent_round_num INTEGER, branch_tag TEXT, source_credit TEXT, "
                "patch_description TEXT, patch_code TEXT, screening_passed INTEGER, "
                "screening_metric REAL, screening_duration REAL, full_passed INTEGER, "
                "full_metric REAL, full_improved INTEGER, full_duration REAL, "
                "rejected_reason TEXT, started_at TEXT, completed_at TEXT)"
            )
            row = conn.execute("SELECT MAX(round_num) FROM experiments").fetchone()
            if row and row[0] is not None:
                round_num = row[0] + 1

    # The person's directions for this round (dashboard Steer or `kaggle steer`).
    from researchforge.kaggle import steering as _steering

    directions = _steering.mark_round(artifacts_dir, round_num)
    up_next = _steering.up_next(artifacts_dir)

    # Load all previous rounds
    from researchforge.kaggle.experiment_loop import load_rounds

    all_rounds = load_rounds(db_path)
    proposer.previous_rounds.extend(all_rounds)

    # Detect merge opportunity: ≥2 independent improvements on separate branches
    merge_opportunity = _detect_merge_opportunity(all_rounds)

    # DAG summary for Claude
    improved_rounds = [r for r in all_rounds if getattr(r, "improved_over_baseline", False)]
    failed_rounds = [
        r
        for r in all_rounds
        if r.rejected_reason or (r.full_result and not r.full_result.metric_improved)
    ]
    best_round = None
    if improved_rounds:
        direction = (
            setup.contract.objective.primary_metric.direction if setup.contract else "maximize"
        )
        if str(direction) == "maximize":
            best_round = max(
                improved_rounds, key=lambda r: r.full_result.metric_value if r.full_result else 0
            )
        else:
            best_round = min(
                improved_rounds,
                key=lambda r: r.full_result.metric_value if r.full_result else float("inf"),
            )

    # New arXiv ideas join the pool before it is ranked, so this round sees them.
    _inject_arxiv_ideas_if_ready(artifacts_dir)

    # One ranking shared with the API-key loop (see kaggle/hypothesis_pool.py).
    # Ideas that wait on an unanswered question stay out until `kaggle respond`.
    from researchforge.kaggle.hypothesis_pool import rank_untested

    lb_direction = str(
        setup.contract.objective.primary_metric.direction if setup.contract else "maximize"
    )
    untested = rank_untested(
        artifacts_dir, lb_direction, blocked=_control.blocked_ideas(artifacts_dir)
    )
    tagged_pool = untested[:10]

    # Build the context text for Claude
    context_text = proposer._build_context(round_num)

    # Resynthesize due?
    resynthesize_due = round_num > 1 and (round_num - 1) % 5 == 0

    # Next-batch status (pre-computed in background)
    from researchforge.intelligence.scanner import get_next_batch_status

    next_batch = get_next_batch_status(artifacts_dir / "intelligence")

    ctx_data = {
        "round_num": round_num,
        "competition_slug": setup.competition_slug,
        "experiments_remaining": setup.max_experiments - (round_num - 1),
        "resynthesize_due": resynthesize_due,
        "resynthesize_note": (
            f"IMPORTANT: Run 'researchforge kaggle resynthesize {setup.competition_slug}' "
            "before proposing this round — it updates the hypothesis pool from results."
            if resynthesize_due
            else ""
        ),
        # DATA PATH — Claude MUST use this in all experiment code
        "data_path_for_experiments": data_path,
        "internet_available_in_experiments": internet_available,
        # DAG state
        "dag_state": {
            "total_rounds": len(all_rounds),
            "improved": len(improved_rounds),
            "failed": len(failed_rounds),
            "best_round_num": best_round.round_num if best_round else None,
            "best_metric": best_round.full_result.metric_value
            if best_round and best_round.full_result
            else None,
            "baseline_metric": setup.baseline_result.cv_score if setup.baseline_result else None,
        },
        # Merge opportunity
        "merge_opportunity": merge_opportunity,
        "merge_note": (
            "MERGE AVAILABLE: Two independent improvements detected. "
            "Consider combining both patches into one experiment. "
            f"Parents: rounds {merge_opportunity['parent_a']} and {merge_opportunity['parent_b']}. "
            "Set parent_round_num to the best single round, and combine both code changes."
            if merge_opportunity
            else ""
        ),
        # Questions waiting for the human — never block the loop; skip what they block
        "open_questions": [
            {"id": q.id, "question": q.question, "options": q.options, "blocks": q.blocks}
            for q in _control.open_interactions(artifacts_dir)
        ],
        "kaggle_gpu_cap_reached": _control.kaggle_gpu_cap_reached(
            artifacts_dir, setup.selected_gpu.value if setup.selected_gpu else "cpu"
        ),
        # Hypothesis pool, ranked by kaggle/hypothesis_pool.py
        "hypothesis_pool": tagged_pool,
        "hypothesis_pool_untested_total": len(untested),
        "person_notes": _control.person_notes(artifacts_dir),
        # Where the round's code runs and what it may not do there (Colab only, …).
        "compute": _round_target.brief(artifacts_dir),
        # What Kaggle made of our own submissions: local metric vs public LB.
        "leaderboard": _submission.leaderboard_table(artifacts_dir),
        "leaderboard_note": _submission.LEADERBOARD_NOTE,
        # Public notebooks claiming a better leaderboard score than our best submission.
        "public_notebooks_ahead": _kernel_board.ahead_of_us(artifacts_dir),
        "public_notebooks_note": (
            "These public notebooks claim a better leaderboard score than our best "
            "submission (gap), best first; 'tried' lists our rounds that ran them. A local "
            "score that keeps improving while the LB does not means the local benchmark "
            "does not predict the LB: running the leading notebook as a round is then the "
            "next step. A round that runs one unscored keeps its submission and is offered "
            "for submission; to ask about one yourself: researchforge kaggle offer-submission "
            f"{setup.competition_slug} <round> --note '<its LB claim>'"
        ),
        # The person's directions: they outrank the idea ranking. Answer each new one once.
        "directions": _steering.for_round(directions),
        "directions_note": (
            "Follow these directions over the idea ranking. For each one with "
            f"reply_needed, run once: researchforge kaggle steer {setup.competition_slug} "
            '--reply <id> "<what you will do about it, in one or two sentences>"'
            if directions
            else ""
        ),
        # The person's queue: next-round directions, then pinned ideas. Its head comes first.
        "up_next": up_next,
        "up_next_note": _steering.FOLLOW_RULE if up_next else "",
        # What the repo engine tried before this loop was the only engine. Its scores
        # come from another benchmark: read them as outcomes, not as numbers to beat.
        "earlier_repo_experiments": _earlier_repo_work(artifacts_dir),
        "hypothesis_pool_note": (
            "metric_confidence: 'blind' = local benchmark cannot measure this change. "
            "Use 'high'/'medium' first for fast signal. "
            "Run 'blind' experiments on Kaggle GPU for real signal. "
            "claimed_score / claim_vs_our_best: the public-LB score a notebook's author "
            "claims and its gap to our best submission (our_best_lb). It is the author's "
            "word, not a measurement: pull the notebook (kernel_ref) to see how it gets "
            "there, and judge it by our own benchmark."
        ),
        "recommended_next": (
            (up_next[0]["text"] if up_next else None)
            # Prefer high/medium confidence first for fast feedback
            or next(
                (
                    h["idea"]
                    for h in tagged_pool
                    if h.get("metric_confidence") in ("high", "medium")
                ),
                None,
            )
            or (tagged_pool[0]["idea"] if tagged_pool else None)
            or (
                f"Explore variants of round {best_round.round_num} (best result)"
                if best_round
                else "Start with the top idea from the initial research"
            )
        ),
        # Next batch pre-computed?
        "next_batch_ready": next_batch.ready if next_batch else False,
        "next_batch_hypotheses": next_batch.hypothesis_ids if next_batch else [],
        # Full context for Claude
        "context": context_text,
        "proposal_output_path": str(artifacts_dir / "proposal.json"),
        "proposal_schema": {
            "title": "<short title, max 60 chars>",
            "description": "<one paragraph>",
            "code_changes": (
                "<complete Python code — use "
                f"data_path_for_experiments='{data_path}' for data loading>"
            ),
            "rationale": "<why this improves the metric>",
            "expected_impact": "<e.g. +1-3% AUC>",
            "source_credit": "<arXiv paper or kernel name, or empty string>",
            "parent_round_num": best_round.round_num if best_round else None,
            "branch_tag": "",
            "dataset_sources": "<list of 'username/dataset-slug' to attach, or [] for none>",
            "kernel_sources": "<optional: notebooks whose outputs it reads, 'owner/notebook'>",
            "model_sources": "<optional: Kaggle Models, 'owner/model/framework/variation/version'>",
            "docker_image": "<optional: the Kaggle Docker image a ported notebook needs>",
            "machine_shape": "<optional: Kaggle GPU type, e.g. 'NvidiaTeslaT4'>",
            "hypothesis": "<exact 'idea' text from hypothesis_pool this proposal tests, or empty>",
            "follows": "<ids from up_next this round follows, e.g. ['d-1a2b3c']; [] for none>",
            "declines": "<{'<up_next id>': '<why this round cannot follow it>'}, or {}>",
        },
        # AI self-hypothesize: Claude can add its own ideas to the pool.
        # After reading the results above, if you see a promising direction
        # that isn't in the hypothesis_pool yet, add it before writing the proposal:
        #
        #   researchforge kaggle propose-hypothesis <slug> \
        #       --idea "Try w_mid=1.5 based on monotonic local gain" \
        #       --rationale "MRR grew through w_mid=3.0; real LB at 1.0 shows headroom" \
        #       --source "claude_observation" --priority 2
        #
        # Bulk (multiple ideas at once):
        #   researchforge kaggle propose-hypothesis <slug> \
        #       --bulk '[{"idea": "...", "rationale": "...", "priority": 3}]'
        #
        # The proposed hypotheses appear in hypothesis_pool on the next prepare-round.
        "ai_self_hypothesize_command": (
            f"researchforge kaggle propose-hypothesis {setup.competition_slug} "
            f"--idea '<your idea>' --rationale '<why>' --source 'claude_observation'"
        ),
    }

    # Write context file for Claude to reference
    atomic_write_json(artifacts_dir / "pending_round_context.json", ctx_data)
    typer.echo(json.dumps(ctx_data, indent=2))


def _inject_arxiv_ideas_if_ready(artifacts_dir: Path) -> None:
    """Move any new arXiv ideas into the hypothesis pool."""
    try:
        from researchforge.intelligence.scanner import inject_arxiv_ideas_into_hypothesis_pool

        inject_arxiv_ideas_into_hypothesis_pool(artifacts_dir / "intelligence", artifacts_dir)
    except Exception:
        pass


@kaggle_app.command("propose-hypothesis")
def propose_hypothesis(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    idea: Annotated[
        str,
        typer.Option(
            "--idea",
            help="Hypothesis idea text. Can be a single string or inline JSON array of ideas.",
        ),
    ] = "",
    rationale: Annotated[
        str,
        typer.Option("--rationale", help="Why this idea might improve the score."),
    ] = "",
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Credit for the idea: 'claude_observation', 'arxiv:<id>', 'kernel:<ref>', etc.",
        ),
    ] = "claude_observation",
    priority: Annotated[
        int,
        typer.Option("--priority", help="Priority 1-10 (1=highest). Default 5."),
    ] = 5,
    bulk_json: Annotated[
        str,
        typer.Option(
            "--bulk",
            help=(
                "JSON array of hypothesis objects: "
                '[{"idea": "...", "rationale": "...", "source": "...", "priority": 5}, ...]'
                " or a file path to a JSON file."
            ),
        ),
    ] = "",
) -> None:
    """Add one or more AI-proposed hypotheses directly to the experiment pool.

    Claude (or any AI) calls this after reading prepare-round output and seeing
    a promising direction — without waiting for a full resynthesize cycle.

    Single hypothesis:
        researchforge kaggle propose-hypothesis titanic \\
            --idea "Try w_mid=1.5 based on local gain trend from exp-010" \\
            --rationale "Local MRR grew monotonically through w_mid=3.0; real LB gain \\
                         at w_mid=1.0 suggests more headroom" \\
            --source "claude_observation"

    Bulk (array):
        researchforge kaggle propose-hypothesis titanic \\
            --bulk '[{"idea": "w_mid=1.5 variant", "rationale": "...", "priority": 3}, ...]'
    """
    import json as _json

    from researchforge.kaggle.orchestrator import OvernightOrchestrator

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pool_path = artifacts_dir / "pending_hypotheses.json"

    # Load existing pool
    existing: list[dict[str, Any]] = []
    if pool_path.exists():
        try:
            existing = _json.loads(pool_path.read_text())
        except Exception:
            existing = []

    # Build list of new hypotheses
    new_hyps: list[dict[str, Any]] = []

    if bulk_json:
        # Accept inline JSON or file path
        try:
            p = Path(bulk_json)
            raw = p.read_text() if p.exists() else bulk_json
            parsed = _json.loads(raw)
            if isinstance(parsed, list):
                new_hyps = parsed
            elif isinstance(parsed, dict):
                new_hyps = [parsed]
        except (_json.JSONDecodeError, OSError) as e:
            typer.secho(f"❌ --bulk JSON parse error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
    elif idea:
        # Single hypothesis from flags
        new_hyps = [
            {
                "idea": idea,
                "rationale": rationale,
                "source": source,
                "priority": priority,
                "tested": False,
            }
        ]
    else:
        typer.secho(
            "❌ Provide either --idea <text> or --bulk <json>.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Normalise and stamp each entry
    import datetime as _dt

    stamped: list[dict[str, Any]] = []
    for h in new_hyps:
        if not h.get("idea"):
            continue
        stamped.append(
            {
                "idea": h.get("idea", "").strip(),
                "rationale": h.get("rationale", ""),
                "source": h.get("source", source),
                "priority": int(h.get("priority", priority)),
                "tested": False,
                "proposed_at": _dt.datetime.now().isoformat(),
                "proposed_by": "ai",
            }
        )

    if not stamped:
        typer.secho("❌ No valid hypotheses found in input.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Deduplicate: skip ideas already in pool (case-insensitive prefix match)
    existing_ideas = {e.get("idea", "").lower()[:60] for e in existing}
    added = [h for h in stamped if h["idea"].lower()[:60] not in existing_ideas]
    skipped = len(stamped) - len(added)

    atomic_write_json(pool_path, existing + added)

    typer.echo(f"✅ Added {len(added)} hypothesis(es) to pool for {slug}")
    for h in added:
        typer.echo(f"   [{h['priority']}] {h['idea'][:80]}")
    if skipped:
        typer.echo(f"   ({skipped} duplicate(s) skipped)")

    # Push notification
    try:
        from researchforge.notifications.service import notify_new_hypotheses

        setup_path = artifacts_dir / "overnight_setup.json"
        topic = ""
        if setup_path.exists():
            topic = _json.loads(setup_path.read_text()).get("ntfy_topic", "")
        notify_new_hypotheses(
            len(added),
            [h["idea"][:40] for h in added],
            project_name=slug,
            topic=topic,
        )
    except Exception:
        pass

    typer.echo(
        f"\n💡 Hypotheses will be picked up by the next "
        f"'researchforge kaggle prepare-round {slug}' call."
    )


@kaggle_app.command("arxiv-scan")
def arxiv_scan(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    query: Annotated[
        str,
        typer.Option(
            "--query", "-q", help="arXiv search query (auto-generated from competition if omitted)"
        ),
    ] = "",
    max_results: Annotated[
        int,
        typer.Option("--max-results", help="Max papers to scan"),
    ] = 10,
    background: Annotated[
        bool,
        typer.Option("--background", help="Run as background daemon (checks daily)"),
    ] = False,
    once: Annotated[
        bool,
        typer.Option("--once", help="Run one scan cycle and exit"),
    ] = True,
) -> None:
    """Scan arXiv for new papers and inject technique ideas into the hypothesis pool.

    Run once to get immediate ideas, or --background for a daily daemon.
    Ideas are automatically picked up by prepare-round on next call.

    Examples:
        researchforge kaggle arxiv-scan titanic
        researchforge kaggle arxiv-scan titanic --query "tabular feature engineering" --once
        researchforge kaggle arxiv-scan titanic --background  # daily scanner daemon
    """
    import json as _json

    from researchforge.intelligence.scanner import (
        _run_arxiv_cycle,
        inject_arxiv_ideas_into_hypothesis_pool,
    )
    from researchforge.kaggle.orchestrator import OvernightOrchestrator

    slug = OvernightOrchestrator._parse_slug(competition)
    artifacts_dir = resolve_artifacts_dir(slug)
    intelligence_dir = artifacts_dir / "intelligence"
    intelligence_dir.mkdir(parents=True, exist_ok=True)
    ai_provider = _project_provider(artifacts_dir, "arxiv_scan")

    # Build query from competition metadata if not provided
    if not query:
        meta_path = artifacts_dir / "competition_metadata.json"
        if meta_path.exists():
            try:
                meta = _json.loads(meta_path.read_text())
                task = meta.get("task_type", "")
                metric = meta.get("metric_name", "")
                query = f"{task} {metric} machine learning"
            except Exception:
                pass
        if not query:
            query = f"kaggle {slug.replace('-', ' ')} machine learning"

    typer.echo(f"Scanning arXiv: {query!r}", err=True)

    if background:
        import subprocess as sp
        import sys

        sp.Popen(
            [
                sys.executable,
                "-c",
                f"from researchforge.intelligence.scanner import run_arxiv_scanner; "
                f"from pathlib import Path; "
                f"run_arxiv_scanner({query!r}, Path({str(intelligence_dir)!r}))",
            ],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            start_new_session=True,
        )
        typer.echo(
            "arXiv scanner started (daily). Ideas injected into hypothesis pool automatically."
        )
        return

    # One-shot scan
    _run_arxiv_cycle(query, intelligence_dir, ai_provider, max_results)
    injected = inject_arxiv_ideas_into_hypothesis_pool(intelligence_dir, artifacts_dir)

    # Report what was found
    pool_path = intelligence_dir / "arxiv_ideas.json"
    if pool_path.exists():
        ideas = _json.loads(pool_path.read_text())
        typer.echo(
            f"Found {len(ideas)} ideas total, {injected} newly injected into hypothesis pool."
        )
        for idea in ideas[-5:]:
            typer.echo(f"  • {idea.get('idea', '')[:80]}")
            typer.echo(f"    Source: {idea.get('title', '')[:60]} ({idea.get('arxiv_id', '')})")
    else:
        typer.echo("No ideas found in this scan.")


@kaggle_app.command("resynthesize")
def resynthesize(
    competition: Annotated[
        str,
        typer.Argument(help="Kaggle competition slug or URL"),
    ],
) -> None:
    """Re-synthesize the research landscape from experiment results.

    Call this every 5 rounds. It:
      - Summarises what worked and what failed
      - Generates new hypotheses from proven combinations
      - Detects merge opportunities
      - Updates the hypothesis pool for the next rounds
      - Updates the contract target if consistently beating it

    Claude reads the output before proposing the next round.
    """
    import json
    import sqlite3

    from researchforge.kaggle.orchestrator import OvernightSetup

    orchestrator = OvernightOrchestrator(competition)
    artifacts_dir = orchestrator.artifacts_dir
    setup_path = artifacts_dir / "overnight_setup.json"

    if not setup_path.exists():
        typer.secho("❌ Setup not found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    setup = OvernightSetup.load(setup_path)
    db_path = artifacts_dir / "experiment_log.db"
    ai_provider = _project_provider(artifacts_dir, "resynthesize")

    # Build results summary from DB
    results_summary = []
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT round_num, patch_description, branch_tag, source_credit, "
                "full_passed, full_metric, full_improved, rejected_reason "
                "FROM experiments ORDER BY round_num"
            ).fetchall()
            from researchforge.kaggle.experiment_loop import archived_rounds

            hidden = archived_rounds(artifacts_dir)
            rows = [r for r in rows if r[0] not in hidden]
            for row in rows:
                results_summary.append(
                    {
                        "round": row[0],
                        "description": row[1],
                        "branch": row[2] or "",
                        "source": row[3] or "",
                        "passed": bool(row[4]) if row[4] is not None else False,
                        "metric": row[5],
                        "improved": bool(row[6]) if row[6] is not None else False,
                        "rejected_reason": row[7] or "",
                    }
                )

    if not results_summary:
        typer.echo(json.dumps({"status": "no_results_yet", "hypotheses_updated": False}))
        return

    # Build the re-synthesis prompt
    improved = [r for r in results_summary if r["improved"]]
    failed = [r for r in results_summary if not r["passed"] or (r["passed"] and not r["improved"])]
    baseline = setup.baseline_result.cv_score if setup.baseline_result else None
    metric_name = setup.contract.objective.primary_metric.name if setup.contract else "metric"
    direction = (
        str(setup.contract.objective.primary_metric.direction) if setup.contract else "maximize"
    )
    best_metric = (
        max((r["metric"] for r in improved if r["metric"]), default=None)
        if direction == "maximize"
        else min((r["metric"] for r in improved if r["metric"]), default=None)
    )

    _RESYNTH_SYSTEM = """\
You are a Kaggle expert re-synthesising experiment results to guide the next rounds.
Return ONLY valid JSON — no prose, no markdown.

Schema:
{
  "what_we_learned": ["<finding 1>", "<finding 2>", ...],
  "directions_confirmed": ["<technique that works>", ...],
  "directions_pruned": ["<technique to stop trying>", ...],
  "new_hypotheses": [
    {"idea": "<specific testable technique>", "rationale": "<why>", "priority": 1}
  ],
  "contract_target_update": null or <new float target>,
  "contract_target_reason": "<why update if updating>"
}

Rules:
- what_we_learned: 3-5 plain-language findings a non-expert understands
- directions_pruned: techniques that failed in ≥2 rounds — stop wasting compute
- new_hypotheses: ideas from proven combinations NOT yet tried (merges, variants, extensions)
- contract_target_update: update if the best result consistently beats the current target
  (set it to the current best + a small delta)
"""

    results_text = (
        f"Competition: {setup.competition_slug}\n"
        f"Metric: {metric_name} ({direction})\n"
        f"Baseline: {baseline}\n"
        f"Best so far: {best_metric}\n\n"
        f"IMPROVED rounds ({len(improved)}):\n"
        + "\n".join(f"  Round {r['round']}: {r['description']} → {r['metric']}" for r in improved)
        + f"\n\nFAILED/NO-IMPROVEMENT rounds ({len(failed)}):\n"
        + "\n".join(
            f"  Round {r['round']}: {r['description']} → {r['rejected_reason'] or 'no improvement'}"
            for r in failed[:10]
        )
    )

    learned = []
    new_hypotheses = []
    target_update = None

    # Determine AI provider mode from setup
    ai_provider_mode = getattr(setup, "ai_provider_mode", "claude_code")

    if ai_provider is not None and ai_provider_mode != "claude_code":
        # AUTONOMOUS MODE: call API directly
        try:
            raw = ai_provider.generate(_RESYNTH_SYSTEM, results_text, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(raw)
            learned = parsed.get("what_we_learned", [])
            new_hypotheses = parsed.get("new_hypotheses", [])
            target_update = parsed.get("contract_target_update")
        except Exception:
            learned = [f"Improved in {len(improved)}/{len(results_summary)} rounds so far."]
            new_hypotheses = []
    else:
        # CLAUDE CODE MODE: output the synthesis prompt for Claude to answer.
        # Claude reads this output, synthesises in the chat, then writes
        # the JSON response to .overnight_<slug>/resynthesize_response.json
        # which this command reads on the NEXT call.
        response_path = artifacts_dir / "resynthesize_response.json"
        if response_path.exists():
            # Claude already wrote the response — read it
            try:
                parsed = json.loads(response_path.read_text())
                learned = parsed.get("what_we_learned", [])
                new_hypotheses = parsed.get("new_hypotheses", [])
                target_update = parsed.get("contract_target_update")
                response_path.unlink()  # consumed
            except Exception:
                learned = []
                new_hypotheses = []
        else:
            # Output the synthesis prompt for Claude to answer
            synthesis_request = {
                "mode": "claude_code_synthesis",
                "instruction": (
                    "Read the results below and synthesise what we learned. "
                    "Then write your synthesis as JSON to the path in 'write_response_to' "
                    "and call resynthesize again."
                ),
                "write_response_to": str(response_path),
                "response_schema": {
                    "what_we_learned": ["<finding>"],
                    "directions_pruned": ["<technique to stop trying>"],
                    "new_hypotheses": [
                        {"idea": "<technique>", "rationale": "<why>", "priority": 1}
                    ],
                    "contract_target_update": None,
                },
                "results": results_text,
                "system_prompt_for_synthesis": _RESYNTH_SYSTEM,
            }
            typer.echo(json.dumps(synthesis_request, indent=2))
            typer.echo(
                f"\n⏸  Claude Code synthesis needed.\n"
                f"   Read the 'results' field above, synthesise what we learned,\n"
                f"   and write the JSON response to:\n"
                f"   {response_path}\n"
                f"   Then call 'researchforge kaggle resynthesize {setup.competition_slug}' again.",
                err=True,
            )
            return  # Stop here — wait for Claude to write the response

        if not learned:
            # No response yet and no prior response — use heuristic
            learned = [
                f"{len(improved)}/{len(results_summary)} rounds improved the {metric_name}.",
                f"Best result: {best_metric} (baseline: {baseline})"
                if best_metric
                else "No improvement yet.",
            ]
            if failed:
                learned.append(
                    f"Stop trying: {', '.join(r['description'][:30] for r in failed[:3])}"
                )

    # Update pending_hypotheses.json with new ideas
    pool_path = artifacts_dir / "pending_hypotheses.json"
    existing: list[dict[str, Any]] = []
    if pool_path.exists():
        try:
            existing = json.loads(pool_path.read_text())
        except Exception:
            existing = []

    # Add new hypotheses (avoid duplicates)
    existing_ideas_lower = {h.get("idea", "").lower() for h in existing}
    added = 0
    for h in new_hypotheses:
        if h.get("idea", "").lower() not in existing_ideas_lower:
            existing.append(
                {
                    "idea": h["idea"],
                    "rationale": h.get("rationale", ""),
                    "priority": h.get("priority", 5),
                    "source": "resynthesize",
                    "tested": False,
                }
            )
            added += 1

    # Sort by priority
    existing.sort(key=lambda x: x.get("priority", 5))
    pool_path.write_text(json.dumps(existing, indent=2))

    # Optionally update contract target
    if target_update is not None and setup.contract:
        try:
            setup.contract.objective.primary_metric.target_value = float(target_update)
            setup.save()
        except Exception:
            target_update = None

    # Save resynthesize summary
    summary = {
        "synthesized_at": __import__("datetime").datetime.now().isoformat(),
        "rounds_analyzed": len(results_summary),
        "rounds_improved": len(improved),
        "best_metric": best_metric,
        "baseline_metric": baseline,
        "what_we_learned": learned,
        "new_hypotheses_added": added,
        "hypothesis_pool_size": len(existing),
        "contract_target_updated": target_update is not None,
        "new_contract_target": target_update,
    }
    (artifacts_dir / "resynthesize_summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))

    # Print readable summary to stderr for Claude to see
    typer.echo("\n💡 What we learned:", err=True)
    for insight in learned:
        typer.echo(f"   • {insight}", err=True)
    typer.echo(f"\n📋 {added} new hypotheses added to pool (total: {len(existing)})", err=True)
    if target_update:
        typer.echo(f"🎯 Contract target updated to {target_update:.4f}", err=True)


@kaggle_app.command("execute-round")
def execute_round(
    competition: Annotated[
        str,
        typer.Argument(help="Kaggle competition slug or URL"),
    ],
) -> None:
    """Execute the proposal at .overnight_<slug>/proposal.json. Prints result JSON."""
    import json
    import sqlite3
    from datetime import datetime

    from researchforge.kaggle.benchmark_runner import BenchmarkRunner
    from researchforge.kaggle.experiment_loop import (
        ExperimentRound,
        OvernightExperimentLoop,
    )
    from researchforge.kaggle.orchestrator import OvernightSetup
    from researchforge.kaggle.patch_proposer import PatchProposal

    orchestrator = OvernightOrchestrator(competition)
    artifacts_dir = orchestrator.artifacts_dir
    proposal_path = artifacts_dir / "proposal.json"
    lock_path = artifacts_dir / "execute_round.lock"

    if not (artifacts_dir / "overnight_setup.json").exists():
        typer.secho("❌ Setup not found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not proposal_path.exists():
        typer.secho(
            f"❌ No proposal at {proposal_path}. Run prepare-round first, then write the proposal.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    from researchforge.kaggle import control as _control

    if _control.LoopControl.load(artifacts_dir).paused:
        typer.secho(
            "⏸  This loop is paused — not starting a round. "
            f"Resume with: researchforge kaggle resume {orchestrator.slug}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(1)
    _target = (read_json(artifacts_dir / "overnight_setup.json", {}) or {}).get(
        "selected_gpu"
    ) or "cpu"
    if _control.kaggle_gpu_cap_reached(artifacts_dir, _target):
        typer.secho(
            "⛔ Today's Kaggle GPU cap is reached "
            f"({_control.kaggle_gpu_runs_today(artifacts_dir)} runs). Switch compute with "
            f"'researchforge kaggle compute {orchestrator.slug} <target>' or raise the cap with "
            f"'researchforge kaggle settings {orchestrator.slug} --kaggle-gpu-daily-cap N'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # The head of the person's Up next queue is followed or declined with a reason.
    from researchforge.kaggle import steering as _steering

    try:
        _proposed = json.loads(proposal_path.read_text())
    except ValueError as exc:
        typer.secho(f"❌ {proposal_path.name} is not valid JSON: {exc}", fg="red", err=True)
        raise typer.Exit(1) from None
    _ignored = _steering.check_proposal(artifacts_dir, _proposed)
    if _ignored:
        typer.secho(f"❌ Not run: {_ignored}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # A lock left by a process that died is released, and its round recorded as interrupted.
    from researchforge.kaggle.experiment_loop import recover_interrupted_round

    recovered = recover_interrupted_round(artifacts_dir)
    if recovered is not None:
        typer.secho(
            f"⚠️  Released a lock left by an execute-round that ended without finishing"
            f"{f' (round {recovered} recorded as interrupted)' if recovered > 0 else ''}.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    # Bug #22: concurrency guard — refuse to run if another execute-round is in progress
    if lock_path.exists():
        try:
            lock_info = json.loads(lock_path.read_text())
            started = lock_info.get("started_at", "unknown time")
            pid = lock_info.get("pid", "unknown PID")
        except Exception:
            started, pid = "unknown time", "unknown PID"
        typer.secho(
            f"❌ Another execute-round is already running (started {started}, PID {pid}).\n"
            f"   Wait for it to finish, or delete {lock_path} if it crashed.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Write lock file
    import os as _os

    lock_info = {
        "started_at": datetime.now().isoformat(),
        "pid": _os.getpid(),
        "proposal": proposal_path.read_text()[:200],
    }
    atomic_write_json(lock_path, lock_info)

    setup = OvernightSetup.load(artifacts_dir / "overnight_setup.json")
    _control.record_compute_use(
        artifacts_dir, setup.selected_gpu.value if setup.selected_gpu else "cpu"
    )
    proposal_data = json.loads(proposal_path.read_text())
    proposal = PatchProposal(
        title=proposal_data.get("title", ""),
        description=proposal_data.get("description", ""),
        code_changes=proposal_data.get("code_changes", ""),
        rationale=proposal_data.get("rationale", ""),
        expected_impact=proposal_data.get("expected_impact", ""),
        source_credit=proposal_data.get("source_credit", ""),
        parent_round_num=proposal_data.get("parent_round_num"),
        branch_tag=proposal_data.get("branch_tag", ""),
    )
    # Everything the proposal attaches to its Kaggle notebook: datasets, other notebooks'
    # outputs, Kaggle Models, Docker image, GPU type (bug #20 was datasets alone).
    from researchforge.kaggle.api import kernel_inputs

    inputs = kernel_inputs(proposal_data)
    refused = _round_target.refusal(artifacts_dir, proposal.code_changes)
    if refused:
        typer.secho(f"❌ Not run: {refused}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    db_path = artifacts_dir / "experiment_log.db"
    round_num = 1
    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.setup = setup
    loop.db_path = db_path
    loop.benchmark_runner = BenchmarkRunner(setup)
    loop._init_database()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(round_num) FROM experiments").fetchone()
        if row and row[0] is not None:
            round_num = row[0] + 1
    _steering.record_round(artifacts_dir, round_num, proposal_data)

    rnd = ExperimentRound(
        round_num=round_num,
        patch_description=proposal.title,
        patch_code=proposal.code_changes,
        parent_round_num=proposal.parent_round_num,
        branch_tag=proposal.branch_tag,
        source_credit=proposal.source_credit,
    )

    # Publish the in-flight round so the dashboard shows it while it runs.
    live = {
        "round_num": round_num,
        "title": proposal.title,
        "branch_tag": proposal.branch_tag,
        "parent_round_num": proposal.parent_round_num,
        "pid": _os.getpid(),
        "started_at": lock_info["started_at"],
        "stage": "screening",
        "hypothesis": proposal_data.get("hypothesis", ""),
    }
    atomic_write_json(lock_path, {**lock_info, **live})
    _set_inflight(artifacts_dir, round_num, live)

    # A stop signal (a shell's time limit, `kill`) exits normally so the cleanup below
    # runs and the round is recorded as interrupted instead of leaving a stale lock.
    import signal as _signal

    def _exit_on_term(signum: int, frame: Any) -> None:
        raise SystemExit(143)

    _signal.signal(_signal.SIGTERM, _exit_on_term)
    persisted = False
    try:
        rnd.screening_result = loop.benchmark_runner.run_screening(proposal.code_changes, 300)
        if rnd.passed_screening:
            live["stage"] = "full"
            atomic_write_json(lock_path, {**lock_info, **live})
            _set_inflight(artifacts_dir, round_num, live)
            limit = round_time_limit_seconds(artifacts_dir)
            # Pass the notebook inputs through to the Kaggle GPU runner
            if inputs and hasattr(loop.benchmark_runner, "_run_kaggle_gpu"):
                rnd.full_result = loop.benchmark_runner._run_kaggle_gpu(
                    proposal.code_changes, limit, inputs=inputs
                )
            else:
                rnd.full_result = loop.benchmark_runner.run_full_benchmark(
                    proposal.code_changes, limit
                )
        rnd.completed_at = datetime.now()
        loop._persist_round(rnd)
        persisted = True
    except BaseException as exc:
        if not persisted:
            rnd.rejected_reason = (
                f"interrupted: {type(exc).__name__} during the {live['stage']} stage"
            )
            rnd.completed_at = datetime.now()
            with contextlib.suppress(Exception):
                loop._persist_round(rnd)
        raise
    finally:
        # Bug #22: always release the lock, even on error
        if lock_path.exists():
            lock_path.unlink()
        _set_inflight(artifacts_dir, round_num, None)

    # Archive the used proposal
    proposal_path.rename(artifacts_dir / f"proposal_round_{round_num}.json")

    result_data = {
        "round_num": round_num,
        "title": proposal.title,
        "branch_tag": proposal.branch_tag,
        "screening_passed": rnd.passed_screening,
        "screening_metric": rnd.screening_result.metric_value if rnd.screening_result else None,
        "full_passed": rnd.passed_full_benchmark,
        "full_metric": rnd.full_result.metric_value if rnd.full_result else None,
        "improved": rnd.improved_over_baseline,
        "error": (rnd.full_result.error_message if rnd.full_result else None)
        or (rnd.screening_result.error_message if rnd.screening_result else None),
    }
    result_data["tested_hypothesis"] = _mark_hypothesis_tested(
        artifacts_dir,
        proposal_data,
        round_num,
        _round_outcome(bool(result_data["improved"]), str(result_data["error"] or "") or None),
    )
    typer.echo(json.dumps(result_data, indent=2))

    # Per-experiment push notification
    _notify_round_result(setup, result_data)

    # Offer a submission when the submission policy says so (never submits by itself)
    if result_data["improved"] and not result_data["error"]:
        _control.offer_submission(
            artifacts_dir,
            round_num,
            proposal.title,
            rnd.full_result.metric_value if rnd.full_result else None,
            setup.selected_gpu.value if setup.selected_gpu else "cpu",
        )
    # A round that ran unscored (a public notebook, say) but left a submission.
    from researchforge.kaggle.submission import offer_missed as _offer_missed

    with contextlib.suppress(Exception):
        _offer_missed(artifacts_dir)


@kaggle_app.command("run-parallel")
def run_parallel(
    competition: Annotated[str, typer.Argument(help="Kaggle competition slug or URL")],
    proposals: Annotated[
        str,
        typer.Option(
            "--proposals",
            help=(
                "Comma-separated proposal file paths, or 'auto' to find all "
                "proposal_*.json in the artifacts dir"
            ),
        ),
    ] = "auto",
    max_workers: Annotated[
        int,
        typer.Option(
            "--max-workers", help="Max parallel experiment threads [default: number of proposals]"
        ),
    ] = 0,
) -> None:
    """Run multiple proposals in parallel across available compute targets.

    Dispatches each proposal to a separate thread. When multiple experiments
    improve, the merge opportunity is automatically surfaced.

    Proposal files are written by prepare-round (one file per experiment slot).
    Name them proposal_a.json, proposal_b.json, etc. — or use --proposals auto
    to pick up all proposal_*.json files in the artifacts directory.

    Examples:
        researchforge kaggle run-parallel titanic
        researchforge kaggle run-parallel titanic --proposals proposal_a.json,proposal_b.json
        researchforge kaggle run-parallel titanic --max-workers 2
    """
    import json
    import sqlite3
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime

    from researchforge.kaggle.benchmark_runner import BenchmarkRunner
    from researchforge.kaggle.experiment_loop import (
        ExperimentRound,
        OvernightExperimentLoop,
    )
    from researchforge.kaggle.orchestrator import OvernightSetup
    from researchforge.kaggle.patch_proposer import PatchProposal

    orchestrator = OvernightOrchestrator(competition)
    artifacts_dir = orchestrator.artifacts_dir

    if not (artifacts_dir / "overnight_setup.json").exists():
        typer.secho("❌ Setup not found. Run overnight first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    from researchforge.kaggle import control as _control

    if _control.LoopControl.load(artifacts_dir).paused:
        typer.secho(
            "⏸  This loop is paused — not starting rounds. "
            f"Resume with: researchforge kaggle resume {orchestrator.slug}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(1)

    setup = OvernightSetup.load(artifacts_dir / "overnight_setup.json")

    # Collect proposal files
    if proposals == "auto":
        # proposal_round_<n>.json are archives of rounds already executed.
        proposal_files = sorted(
            p
            for p in artifacts_dir.glob("proposal_*.json")
            if not p.name.startswith("proposal_round_")
        )
        # Also include plain proposal.json if it exists
        plain = artifacts_dir / "proposal.json"
        if plain.exists():
            proposal_files = [plain] + list(proposal_files)
    else:
        proposal_files = [
            artifacts_dir / p.strip() if not Path(p.strip()).is_absolute() else Path(p.strip())
            for p in proposals.split(",")
        ]

    proposal_files = [p for p in proposal_files if p.exists()]
    for path in proposal_files:
        with contextlib.suppress(ValueError, OSError):
            refused = _round_target.refusal(
                artifacts_dir, str(json.loads(path.read_text()).get("code_changes") or "")
            )
            if refused:
                typer.secho(f"❌ {path.name} not run: {refused}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
    if not proposal_files:
        typer.secho(
            "❌ No proposal files found. Write proposal_a.json, proposal_b.json, etc. "
            "in the artifacts directory first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    # One proposal in the batch follows (or declines) the head of the person's Up next.
    from researchforge.kaggle import steering as _steering

    _verdicts = []
    for path in proposal_files:
        try:
            _verdicts.append(_steering.check_proposal(artifacts_dir, json.loads(path.read_text())))
        except ValueError as exc:
            typer.secho(f"❌ {path.name} is not valid JSON: {exc}", fg="red", err=True)
            raise typer.Exit(1) from None
    if all(_verdicts):
        typer.secho(
            f"❌ Not run: no proposal follows Up next — {_verdicts[0]}.", fg="red", err=True
        )
        raise typer.Exit(1)

    n = len(proposal_files)
    workers = max_workers if max_workers > 0 else n
    typer.echo(f"\n🔀 Running {n} experiment(s) in parallel (max {workers} threads)...")

    # Shared state
    db_path = artifacts_dir / "experiment_log.db"
    db_lock = threading.Lock()

    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.setup = setup
    loop.db_path = db_path
    loop.benchmark_runner = BenchmarkRunner(setup)
    loop._init_database()

    # Get base round_num (we'll assign sequential ones under a lock)
    _round_counter: list[int] = [1]
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(round_num) FROM experiments").fetchone()
        if row and row[0] is not None:
            _round_counter[0] = row[0] + 1

    results: list[dict[str, Any]] = []

    def _run_one(proposal_path: Path) -> dict[str, Any]:
        proposal_data = json.loads(proposal_path.read_text())
        proposal = PatchProposal(
            title=proposal_data.get("title", ""),
            description=proposal_data.get("description", ""),
            code_changes=proposal_data.get("code_changes", ""),
            rationale=proposal_data.get("rationale", ""),
            expected_impact=proposal_data.get("expected_impact", ""),
            source_credit=proposal_data.get("source_credit", ""),
            parent_round_num=proposal_data.get("parent_round_num"),
            branch_tag=proposal_data.get("branch_tag", ""),
        )
        from researchforge.kaggle.api import kernel_inputs

        inputs = kernel_inputs(proposal_data)

        # Assign unique round number under lock
        with db_lock:
            round_num = _round_counter[0]
            _round_counter[0] += 1
            _steering.record_round(artifacts_dir, round_num, proposal_data)

        rnd = ExperimentRound(
            round_num=round_num,
            patch_description=proposal.title,
            patch_code=proposal.code_changes,
            parent_round_num=proposal.parent_round_num,
            branch_tag=proposal.branch_tag,
            source_credit=proposal.source_credit,
        )

        live = {
            "round_num": round_num,
            "title": proposal.title,
            "branch_tag": proposal.branch_tag,
            "parent_round_num": proposal.parent_round_num,
            "started_at": datetime.now().isoformat(),
            "stage": "screening",
            "hypothesis": proposal_data.get("hypothesis", ""),
        }
        _set_inflight(artifacts_dir, round_num, live)
        try:
            rnd.screening_result = loop.benchmark_runner.run_screening(proposal.code_changes, 300)
            if rnd.passed_screening:
                _set_inflight(artifacts_dir, round_num, {"stage": "full"})
                limit = round_time_limit_seconds(artifacts_dir)
                if inputs and hasattr(loop.benchmark_runner, "_run_kaggle_gpu"):
                    rnd.full_result = loop.benchmark_runner._run_kaggle_gpu(
                        proposal.code_changes, limit, inputs=inputs
                    )
                else:
                    rnd.full_result = loop.benchmark_runner.run_full_benchmark(
                        proposal.code_changes, limit
                    )
            rnd.completed_at = datetime.now()
            # Persist under lock to avoid SQLite write conflicts
            with db_lock:
                loop._persist_round(rnd)
        except Exception as exc:
            rnd.rejected_reason = str(exc)[:200]
        finally:
            _set_inflight(artifacts_dir, round_num, None)

        result = {
            "round_num": round_num,
            "proposal_file": proposal_path.name,
            "title": proposal.title,
            "branch_tag": proposal.branch_tag,
            "screening_passed": rnd.passed_screening,
            "screening_metric": rnd.screening_result.metric_value if rnd.screening_result else None,
            "full_passed": rnd.passed_full_benchmark,
            "full_metric": rnd.full_result.metric_value if rnd.full_result else None,
            "improved": rnd.improved_over_baseline,
            "error": (rnd.full_result.error_message if rnd.full_result else None)
            or (rnd.screening_result.error_message if rnd.screening_result else None)
            or rnd.rejected_reason,
        }
        result["tested_hypothesis"] = _mark_hypothesis_tested(
            artifacts_dir,
            proposal_data,
            round_num,
            _round_outcome(bool(result["improved"]), str(result["error"] or "") or None),
        )

        # Archive proposal
        with contextlib.suppress(OSError):
            proposal_path.rename(artifacts_dir / f"proposal_round_{round_num}.json")

        return result

    # Run in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, p): p for p in proposal_files}
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                status = (
                    "✅ improved"
                    if r.get("improved")
                    else ("❌ failed" if r.get("error") else "→ no improvement")
                )
                metric_str = (
                    f"  metric={r['full_metric']:.4f}" if r.get("full_metric") is not None else ""
                )
                typer.echo(f"  R{r['round_num']} {r['title'][:50]!r} — {status}{metric_str}")
            except Exception as exc:
                typer.secho(f"  ⚠️  Worker error: {exc}", fg=typer.colors.YELLOW)

    # Detect merge opportunity across the just-completed batch
    improved_this_batch = [r for r in results if r.get("improved")]
    merge_note = ""
    if len(improved_this_batch) >= 2:
        a, b = improved_this_batch[0], improved_this_batch[1]
        merge_note = (
            f"\n🔀 MERGE OPPORTUNITY: rounds {a['round_num']} and {b['round_num']} "
            f"both improved ({a.get('full_metric', '?'):.4f} / {b.get('full_metric', '?'):.4f}). "
            "Consider combining their patches in the next proposal."
        )

    summary = {
        "parallel_batch": {
            "experiments_run": len(results),
            "improved": len(improved_this_batch),
            "results": results,
        },
        "merge_opportunity": (
            {
                "parent_a": improved_this_batch[0]["round_num"],
                "parent_b": improved_this_batch[1]["round_num"],
            }
            if len(improved_this_batch) >= 2
            else None
        ),
    }
    typer.echo(json.dumps(summary, indent=2))
    if merge_note:
        typer.secho(merge_note, fg=typer.colors.GREEN)

    # Send one notification for the batch
    if results:
        best = max(
            (r for r in results if r.get("full_metric") is not None),
            key=lambda r: r["full_metric"],
            default=None,
        )
        batch_result = {
            "round_num": f"batch({','.join(str(r['round_num']) for r in results)})",
            "title": f"{len(results)} experiments",
            "branch_tag": "",
            "improved": bool(improved_this_batch),
            "full_metric": best["full_metric"] if best else None,
            "error": None,
        }
        _notify_round_result(setup, batch_result)


@kaggle_app.command("finish")
def finish(
    competition: Annotated[
        str,
        typer.Argument(help="Kaggle competition slug or URL"),
    ],
) -> None:
    """Generate the morning report and send push notification. Call when all rounds are done."""
    import json
    import sqlite3
    from datetime import datetime

    from researchforge.kaggle.experiment_loop import (
        BenchmarkResult,
        ExperimentRound,
        OvernightExperimentLoop,
        RunSummary,
        ScreeningResult,
    )
    from researchforge.kaggle.orchestrator import OvernightSetup

    orchestrator = OvernightOrchestrator(competition)
    artifacts_dir = orchestrator.artifacts_dir

    if not (artifacts_dir / "overnight_setup.json").exists():
        typer.secho("❌ Setup not found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    setup = OvernightSetup.load(artifacts_dir / "overnight_setup.json")
    db_path = artifacts_dir / "experiment_log.db"

    rounds = []
    start_time = datetime.now()
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT round_num, patch_description, patch_code, parent_round_num, branch_tag, "
                "source_credit, screening_passed, screening_metric, screening_duration, "
                "full_passed, full_metric, full_improved, full_duration, rejected_reason, "
                "started_at, completed_at FROM experiments ORDER BY round_num"
            ).fetchall()
            from researchforge.kaggle.experiment_loop import archived_rounds

            hidden = archived_rounds(artifacts_dir)
            for row in rows:
                if row[0] in hidden:
                    continue
                r = ExperimentRound(
                    round_num=row[0],
                    patch_description=row[1] or "",
                    patch_code=row[2] or "",
                    parent_round_num=row[3],
                    branch_tag=row[4] or "",
                    source_credit=row[5] or "",
                    rejected_reason=row[13],
                    started_at=datetime.fromisoformat(row[14]) if row[14] else datetime.now(),
                    completed_at=datetime.fromisoformat(row[15]) if row[15] else None,
                )
                if row[6] is not None:
                    r.screening_result = ScreeningResult(
                        passed=bool(row[6]), metric_value=row[7], duration_seconds=row[8] or 0
                    )
                if row[9] is not None:
                    r.full_result = BenchmarkResult(
                        passed=bool(row[9]),
                        metric_value=row[10] or 0.0,
                        metric_improved=bool(row[11]),
                        duration_seconds=row[12] or 0,
                    )
                rounds.append(r)
        if rounds:
            start_time = rounds[0].started_at

    ended_at = datetime.now()
    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.setup = setup
    loop.db_path = db_path
    loop.rounds = rounds
    loop.started_at = start_time

    summary = RunSummary(
        setup=setup,
        rounds=rounds,
        started_at=start_time,
        ended_at=ended_at,
        total_duration_seconds=(ended_at - start_time).total_seconds(),
        budget_remaining_seconds=0,
        experiments_remaining=0,
    )
    loop.save_summary(summary)

    from researchforge.notifications.service import notify_loop_complete

    best = summary.best_round
    best_metric = best.full_result.metric_value if best and best.full_result else None
    notify_loop_complete(
        competition_slug=setup.competition_slug,
        rounds_attempted=summary.total_rounds_attempted,
        rounds_improved=summary.rounds_improved,
        best_metric=best_metric,
        baseline_metric=setup.baseline_result.cv_score if setup.baseline_result else None,
    )

    typer.echo(
        json.dumps(
            {
                "competition": setup.competition_slug,
                "rounds_attempted": summary.total_rounds_attempted,
                "rounds_improved": summary.rounds_improved,
                "best_metric": best_metric,
                "baseline_metric": setup.baseline_result.cv_score
                if setup.baseline_result
                else None,
                "summary_path": str(artifacts_dir / "overnight_summary.json"),
            },
            indent=2,
        )
    )


# Human-in-the-loop commands (status / pause / resume / respond / compute / add-hypothesis /
# settings) live in their own module and register on kaggle_app at import time.
from researchforge.kaggle import control_cli as _control_cli  # noqa: E402, F401
from researchforge.kaggle import kernel_board as _kernel_board  # noqa: E402
from researchforge.kaggle import keys_cli as _keys_cli  # noqa: E402, F401
from researchforge.kaggle import round_target as _round_target  # noqa: E402
from researchforge.kaggle import submission as _submission  # noqa: E402
