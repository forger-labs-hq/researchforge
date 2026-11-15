"""``researchforge kaggle keys`` — the Keys & accounts page, from a terminal or an agent.

An agent in Claude Code or Cursor uses ``status``, ``open`` and ``wait``: it learns
which keys are missing, opens the dashboard's Keys page in the person's own browser,
and waits for the key to turn ready — it never sees a value. ``set`` asks for the value
at a hidden prompt and only works in the person's own terminal.
"""

from __future__ import annotations

import json
import sys
import time
import webbrowser
from typing import Annotated

import typer

from researchforge.kaggle import keys
from researchforge.kaggle.cli import kaggle_app

keys_app = typer.Typer(
    help="Keys & accounts: see, set and test every credential (values are never shown).",
    no_args_is_help=True,
)
kaggle_app.add_typer(keys_app, name="keys")

NameArg = Annotated[str, typer.Argument(help=f"One of: {', '.join(keys.SLOTS)}")]


def _check(name: str) -> str:
    if name.strip().lower() not in keys.SLOTS:
        typer.secho(f"❌ Unknown key {name!r}. Use one of: {', '.join(keys.SLOTS)}", fg="red")
        raise typer.Exit(1)
    return name.strip().lower()


@keys_app.command("status")
def status_cmd(
    name: Annotated[str | None, typer.Argument(help="One key (default: all)")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Which keys are set, where, and what the last test said. Never prints a value."""
    rows = keys.status(_check(name) if name else None)
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    for r in rows:
        test = r["test"]
        mark = "✅" if r["set"] and (not test or test["ok"]) else "⚠️ " if r["set"] else "○ "
        line = f"{mark} {r['name']:<10} {r['label']:<22} "
        line += r["where"] or f"not set — {r['get_it']}"
        if test:
            line += f"  [test {'ok' if test['ok'] else 'failed'}: {test['detail']}]"
        typer.echo(line)


@keys_app.command("open")
def open_cmd(
    name: Annotated[str, typer.Argument(help="The key to scroll to (default: the page)")] = "",
) -> None:
    """Open the dashboard's Keys page in your browser (starting the dashboard if needed)."""
    from researchforge.kaggle.dashboard import start_daemon

    port = start_daemon()
    url = keys.page_url(_check(name) if name else "", port)
    opened = webbrowser.open(url)
    typer.echo(f"{'Opened' if opened else 'Open'}: {url}")
    typer.echo("Paste the key there — it is saved on this machine and never shown again.")


@keys_app.command("wait")
def wait_cmd(
    name: NameArg,
    timeout: Annotated[int, typer.Option("--timeout", help="Give up after N seconds")] = 600,
) -> None:
    """Wait until a key is set (and, once tested, passed). Exit 1 on timeout."""
    name = _check(name)
    end = time.time() + max(5, timeout)
    while time.time() < end:
        if keys.ready(name):
            row = keys.status(name)[0]
            typer.echo(f"✅ {row['label']} is set ({row['where']})")
            return
        time.sleep(3)
    typer.secho(f"⏱  {keys.SLOTS[name].label} is still not set.", fg="yellow")
    raise typer.Exit(1)


@keys_app.command("set")
def set_cmd(
    name: NameArg,
    username: Annotated[
        str, typer.Option("--username", help="kaggle only: username for a legacy key")
    ] = "",
) -> None:
    """Set a key at a hidden prompt — run it in your own terminal, never through an agent."""
    name = _check(name)
    if not sys.stdin.isatty():
        typer.secho(
            "❌ Run this in your own terminal (it asks at a hidden prompt), or paste the key "
            f"on the dashboard: researchforge kaggle keys open {name}",
            fg="red",
            err=True,
        )
        raise typer.Exit(1)
    slot = keys.SLOTS[name]
    typer.echo(f"{slot.label}: {slot.get_it}")
    value = typer.prompt(slot.env or slot.label, hide_input=True)
    try:
        where = keys.set_key(name, value, username)
    except ValueError as exc:
        typer.secho(f"❌ {exc}", fg="red")
        raise typer.Exit(1) from None
    typer.secho(f"✅ Saved to {where}. Testing it…", fg="green")
    result = keys.test_key(name)
    typer.echo(f"   {'✅' if result['ok'] else '⚠️ '} {result['detail']}")


@keys_app.command("test")
def test_cmd(name: NameArg) -> None:
    """Make one harmless call with the stored key (a tiny prompt for the Claude token)."""
    result = keys.test_key(_check(name))
    typer.echo(f"{'✅' if result['ok'] else '❌'} {result['detail']}")
    if not result["ok"]:
        raise typer.Exit(1)


@keys_app.command("remove")
def remove_cmd(name: NameArg) -> None:
    """Remove a key ResearchForge stored (Kaggle's own files are left alone)."""
    try:
        typer.echo(keys.remove_key(_check(name)))
    except ValueError as exc:
        typer.secho(f"❌ {exc}", fg="red")
        raise typer.Exit(1) from None
