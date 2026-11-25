"""`researchforge plugins`: what is installed, what it adds, and what failed."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import typer

from researchforge.utils.output import JsonOption, echo_json


def _adds(plugin: Any) -> list[str]:
    adds = []
    for attr, label in (
        ("register_cli", "commands"),
        ("dashboard_views", "dashboard views"),
        ("api_routes", "API routes"),
        ("on_monitor_cycle", "monitor hook"),
    ):
        if callable(getattr(plugin, attr, None)):
            adds.append(label)
    return adds


def plugins_command(json_output: JsonOption = False) -> None:
    """List installed plugins (`researchforge.plugins` entry points) and load errors."""
    from researchforge import plugins

    loaded = plugins.load_plugins()
    rows = [
        {
            "name": lp.name,
            "version": lp.version,
            "distribution": lp.distribution,
            "entry_point": lp.entry_point,
            "adds": _adds(lp.plugin),
        }
        for lp in loaded
    ]
    problems = [asdict(e) for e in plugins.errors()]
    if json_output:
        echo_json({"plugins": rows, "errors": problems, "api_version": plugins.API_VERSION})
        return
    if not rows and not problems:
        typer.echo("No plugins installed.")
        return
    for row in rows:
        adds = ", ".join(row["adds"]) or "nothing yet"
        typer.echo(f"{row['name']} {row['version']} ({row['distribution']}) — adds {adds}")
    for e in problems:
        typer.echo(f"⚠ {e['entry_point']}: {e['stage']} failed — {e['error']}", err=True)
