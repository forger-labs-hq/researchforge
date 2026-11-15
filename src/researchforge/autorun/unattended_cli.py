"""``researchforge auto-run``: unattended rounds for a repository project.

The commands behind the dashboard's repo automation card; see
:mod:`researchforge.autorun.unattended` for how launches, gates and approval work.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from researchforge.autorun import unattended as ua
from researchforge.utils.output import JsonOption

auto_run_app = typer.Typer(
    help="Run the repository loop unattended: a background monitor starts rounds with "
    "Claude Code, Cursor or an API key; plans wait for your approval unless you gave a "
    "standing one.",
    no_args_is_help=True,
)

PathOption = Annotated[
    Path | None,
    typer.Option("--path", help="Project folder (default: the project around the current dir)."),
]


def _root(path: Path | None) -> Path:
    from researchforge.config.paths import find_project_root

    root = find_project_root(path)
    if root is None:
        typer.echo("Not inside a ResearchForge project (no .researchforge/ found).", err=True)
        raise typer.Exit(code=1)
    return root


def _guard_unattended(what: str) -> None:
    """Approvals are the person's: refuse them from an auto-run launch or its driver."""
    import os

    if os.environ.get("RESEARCHFORGE_AUTO_RUN"):
        typer.echo(f"{what} is the person's decision; an unattended run cannot make it.", err=True)
        raise typer.Exit(code=1)


@auto_run_app.command("status")
def status_cmd(path: PathOption = None, json_output: JsonOption = False) -> None:
    """Settings, gates, the running or last launch, plans waiting and the service."""
    root = _root(path)
    st = ua.status(root, check_login=True)
    if json_output:
        typer.echo(json.dumps(st, indent=2, default=str))
        return
    c = st["config"]
    typer.echo(
        f"{root.name}: auto-run {'on' if st['enabled'] else 'off'} · driver {c['driver']}"
        f" · approval {'standing' if st['standing'] else 'ask each plan'}"
    )
    typer.echo(
        f"  {c['rounds_per_launch']} round(s) per launch · {st['rounds_today']} of "
        f"{c['max_rounds_per_day']} today · limit {c['max_minutes']} min"
    )
    r = st["readiness"]
    typer.echo(
        f"  driver: {'ready' if r['ready'] else 'not ready'} — {r['detail']}"
        + (f" (fix: {r['fix']})" if r["fix"] else "")
    )
    typer.echo("  " + ("Waiting: " + "; ".join(st["gates"]) if st["gates"] else "Checks clear."))
    for w in st["plans_waiting"]:
        typer.echo(f"  {w['plan_id']} waits for approval: {w['approach'][:100]}")
    if st["running"]:
        typer.echo(f"  running since {st['running']['started_at']} (pid {st['running']['pid']})")
    if st["last"]:
        last = st["last"]
        typer.echo(
            f"  last launch {last.get('status')} {last.get('finished_at', '')}: "
            f"{last.get('rounds_run', 0)} round(s) {last.get('error', '')}"
        )
    s = st["service"]
    typer.echo(
        f"  service: {'installed' if s['installed'] else 'not installed'}"
        f"{' and loaded' if s['loaded'] else ''} ({s['label']})"
    )


@auto_run_app.command("settings")
def settings_cmd(
    path: PathOption = None,
    on: Annotated[bool | None, typer.Option("--on/--off", help="Turn auto-run on or off.")] = None,
    driver: Annotated[str | None, typer.Option("--driver", help="claude, cursor or byok")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    rounds_per_launch: Annotated[int | None, typer.Option("--rounds-per-launch", min=1)] = None,
    daily_max: Annotated[int | None, typer.Option("--daily-max", min=1)] = None,
    max_minutes: Annotated[int | None, typer.Option("--max-minutes", min=10)] = None,
    merge: Annotated[bool | None, typer.Option("--merge/--no-merge")] = None,
    metric_gate: Annotated[
        bool | None,
        typer.Option("--metric-gate/--no-metric-gate", help="Off: record every result."),
    ] = None,
    ntfy_topic: Annotated[str | None, typer.Option("--ntfy-topic")] = None,
    poll_minutes: Annotated[int | None, typer.Option("--poll-minutes", min=5)] = None,
    papers_hours: Annotated[float | None, typer.Option("--papers-hours", min=0)] = None,
    briefing_at: Annotated[str | None, typer.Option("--briefing-at")] = None,
) -> None:
    """Change auto-run settings. Approval has its own command."""
    root = _root(path)
    try:
        cfg = ua.update_settings(
            root,
            {
                "enabled": on,
                "driver": driver,
                "model": model,
                "rounds_per_launch": rounds_per_launch,
                "max_rounds_per_day": daily_max,
                "max_minutes": max_minutes,
                "merge": merge,
                "metric_gate": metric_gate,
                "ntfy_topic": ntfy_topic,
                "poll_minutes": poll_minutes,
                "papers_hours": papers_hours,
                "briefing_at": briefing_at,
            },
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(cfg.__dict__, indent=2))
    if on and not ua.service_status(root)["installed"]:
        typer.echo(
            "Auto-run is on; install the background service so it runs with no "
            "terminal open:  researchforge auto-run service"
        )


@auto_run_app.command("approval")
def approval_cmd(
    mode: Annotated[str, typer.Argument(help="ask (each plan waits for you) or standing")],
    path: PathOption = None,
) -> None:
    """Ask before every plan, or give the loop your standing approval (typed)."""
    root = _root(path)
    if mode == "standing":
        _guard_unattended("A standing approval")
        if not sys.stdin.isatty():
            typer.echo("Run this in your own terminal: it asks you to type 'approve'.", err=True)
            raise typer.Exit(code=1)
        cfg = ua.RepoAutoRunConfig.load(root)
        typer.echo(
            f"With a standing approval, auto-run runs every plan it writes in {root} without "
            f"asking: up to {cfg.rounds_per_launch} round(s) per launch and "
            f"{cfg.max_rounds_per_day} a day, each experiment under the contract's limits. "
            "You can go back to asking with: researchforge auto-run approval ask"
        )
        typed = typer.prompt("Type 'approve' to give the standing approval")
    else:
        typed = ""
    try:
        cfg = ua.set_approval(root, mode, typed=typed)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    typer.echo("Standing approval given." if cfg.standing else "Each plan now waits for you.")


@auto_run_app.command("start")
def start_cmd(path: PathOption = None) -> None:
    """Start one launch now, through the same gates as the monitor."""
    root = _root(path)
    status, detail = ua.start_if_ready(root)
    typer.echo(detail)
    if status != "ok":
        raise typer.Exit(code=1)


@auto_run_app.command("stop")
def stop_cmd(path: PathOption = None) -> None:
    """Stop the running launch. Experiments it finished are kept."""
    stopped = ua.stop_launch(_root(path))
    typer.echo(f"Asked pid {stopped['pid']} to stop." if stopped else "No launch is running.")


@auto_run_app.command("pause")
def pause_cmd(
    path: PathOption = None, reason: Annotated[str, typer.Option("--reason")] = ""
) -> None:
    """Pause: a running launch finishes, no new one starts."""
    from researchforge.kaggle import control

    control.pause(ua.rf_dir(_root(path)), reason)
    typer.echo("Paused.")


@auto_run_app.command("resume")
def resume_cmd(path: PathOption = None) -> None:
    """Resume after a pause."""
    from researchforge.kaggle import control

    control.resume(ua.rf_dir(_root(path)))
    typer.echo("Resumed.")


@auto_run_app.command("monitor")
def monitor_cmd(
    path: PathOption = None,
    once: Annotated[bool, typer.Option("--once", help="One cycle, then exit.")] = False,
) -> None:
    """Run the monitor: a cycle every poll_minutes (the service runs one per interval)."""
    root = _root(path)
    while True:
        state = ua.monitor_cycle(root)
        stamp = state.get("last_cycle_at")
        for name, rec in (state.get("sensors") or {}).items():
            typer.echo(f"[{stamp}] {name}: {rec.get('status')} — {rec.get('detail')}")
        if once:
            return
        time.sleep(ua.RepoAutoRunConfig.load(root).poll_minutes * 60)


@auto_run_app.command("service")
def service_cmd(
    path: PathOption = None,
    uninstall: Annotated[bool, typer.Option("--uninstall", help="Remove the service.")] = False,
) -> None:
    """Install (or remove) the background service: launchd, systemd timer or Task Scheduler."""
    root = _root(path)
    try:
        detail = ua.uninstall_service(root) if uninstall else ua.install_service(root)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(detail)
    if not uninstall:
        typer.echo(f"Log: {ua.service_log(root)}")
