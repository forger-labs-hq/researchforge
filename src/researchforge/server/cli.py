"""`researchforge serve` — local, read-only live monitor."""

from __future__ import annotations

import webbrowser
from typing import Annotated

import typer

from researchforge.config.paths import is_initialized
from researchforge.server.monitor import (
    DEFAULT_PORT,
    hub_record_path,
    monitor_path,
    pick_port,
    read_hub,
    read_monitor,
    spawn_background_hub,
    spawn_background_monitor,
    stop_hub,
    stop_monitor,
)

INSTALL_HINT = (
    "The monitoring server needs the optional dependencies — install with:\n"
    '  pip install "researchforge[serve]"'
)


def _require_serve_deps() -> None:
    try:
        import uvicorn  # noqa: F401

        import researchforge.server.app  # noqa: F401
    except ImportError:
        typer.echo(INSTALL_HINT)
        raise typer.Exit(code=1) from None


def serve_command(
    host: Annotated[
        str, typer.Option("--host", help="Bind address (loopback by default; change with care).")
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            min=1,
            max=65535,
            help=f"Preferred port (default {DEFAULT_PORT}); falls back to a free one if busy.",
        ),
    ] = DEFAULT_PORT,
    background: Annotated[
        bool,
        typer.Option("--background", help="Run detached; manage with --status / --stop."),
    ] = False,
    stop: Annotated[
        bool, typer.Option("--stop", help="Stop the background monitor and exit.")
    ] = False,
    status: Annotated[
        bool, typer.Option("--status", help="Show whether a background monitor is running.")
    ] = False,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the monitor in the default browser.")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", hidden=True, help="Internal: plain foreground run.")
    ] = False,
) -> None:
    """Run the live monitoring server (read-only view of this project)."""
    if stop:
        stopped = stop_monitor()
        if stopped is None:
            typer.echo("No background monitor running.")
        else:
            typer.echo(f"Stopped background monitor (pid {stopped.pid}, {stopped.url}).")
        return

    if status:
        record = read_monitor()
        if record is None:
            typer.echo("No background monitor running.")
        else:
            typer.echo(f"Background monitor running: {record.url} (pid {record.pid})")
        return

    if not is_initialized():
        typer.echo("Not an initialized ResearchForge project. Run `researchforge init`.")
        raise typer.Exit(code=1)
    _require_serve_deps()

    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.echo(
            f"WARNING: binding to {host} exposes this project's research notes and results "
            "beyond this machine. The server is read-only, but anyone who can reach it can "
            "read everything."
        )

    if background:
        existing = read_monitor()
        if existing is not None:
            typer.echo(f"Background monitor already running: {existing.url} (pid {existing.pid})")
            if open_browser:
                webbrowser.open(existing.url)
            return
        chosen = pick_port(host, port)
        if chosen != port:
            typer.echo(f"Port {port} is busy — using {chosen} instead.")
        record = spawn_background_monitor(host, chosen)
        typer.echo(
            f"Monitoring in the background at {record.url} (pid {record.pid}; "
            "`researchforge serve --stop` to stop)"
        )
        if open_browser:
            webbrowser.open(record.url)
        return

    import uvicorn

    from researchforge.server.app import create_app

    chosen = pick_port(host, port)
    if chosen != port:
        typer.echo(f"Port {port} is busy — using {chosen} instead.")
    url = f"http://{host}:{chosen}/"
    typer.echo(f"Monitoring at {url} (read-only; Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        uvicorn.run(create_app(), host=host, port=chosen, log_level="warning")
    finally:
        if foreground:
            # We may be the recorded background monitor; clear the stale record.
            record = read_monitor()
            if record is not None and record.port == chosen:
                monitor_path().unlink(missing_ok=True)


def hub_command(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            min=1,
            max=65535,
            help=f"Preferred port (default {DEFAULT_PORT}); falls back to a free one if busy.",
        ),
    ] = DEFAULT_PORT,
    background: Annotated[
        bool,
        typer.Option("--background", help="Run detached; manage with --status / --stop."),
    ] = False,
    stop: Annotated[bool, typer.Option("--stop", help="Stop the background hub and exit.")] = False,
    status: Annotated[
        bool, typer.Option("--status", help="Show whether the hub is running.")
    ] = False,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the hub in the default browser.")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", hidden=True, help="Internal: plain foreground run.")
    ] = False,
    prune: Annotated[
        bool,
        typer.Option("--prune", help="Remove missing projects from the registry and exit."),
    ] = False,
) -> None:
    """Run the hub: every project on this machine, in one read-only dashboard."""
    if prune:
        from researchforge.config.registry import prune_missing
        from researchforge.utils.console import console

        removed = prune_missing()
        if removed:
            for e in removed:
                console.print(f"  [rf.error]−[/]  [rf.muted]{e.name}[/]  [rf.path]{e.path}[/]")
            console.print(
                f"[rf.success]✓[/] Removed [bold]{len(removed)}[/] missing project"
                f"{'s' if len(removed) != 1 else ''} from the registry."
            )
        else:
            console.print("[rf.muted]No missing projects to remove.[/]")
        return

    if stop:
        stopped = stop_hub()
        if stopped is None:
            typer.echo("No hub running.")
        else:
            typer.echo(f"Stopped hub (pid {stopped.pid}, {stopped.url}).")
        return

    if status:
        record = read_hub()
        if record is None:
            typer.echo("No hub running.")
        else:
            typer.echo(f"Hub running: {record.url} (pid {record.pid})")
        return

    _require_serve_deps()
    host = "127.0.0.1"

    if background:
        existing = read_hub()
        if existing is not None:
            typer.echo(f"Hub already running: {existing.url} (pid {existing.pid})")
            if open_browser:
                webbrowser.open(existing.url)
            return
        chosen = pick_port(host, port)
        if chosen != port:
            typer.echo(f"Port {port} is busy — using {chosen} instead.")
        record = spawn_background_hub(host, chosen)
        typer.echo(
            f"Hub running in the background at {record.url} (pid {record.pid}; "
            "`researchforge hub --stop` to stop)"
        )
        if open_browser:
            webbrowser.open(record.url)
        return

    import uvicorn

    from researchforge.server.hub import create_hub_app

    chosen = pick_port(host, port)
    if chosen != port:
        typer.echo(f"Port {port} is busy — using {chosen} instead.")
    url = f"http://{host}:{chosen}/"
    typer.echo(f"Hub at {url} (read-only; Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        uvicorn.run(create_hub_app(), host=host, port=chosen, log_level="warning")
    finally:
        if foreground:
            # We may be the recorded background hub; clear the stale record.
            record = read_hub()
            if record is not None and record.port == chosen:
                hub_record_path().unlink(missing_ok=True)
