"""Notification setup CLI.

researchforge notify setup --topic <ntfy-topic>
researchforge notify test
researchforge notify status
"""

from __future__ import annotations

from typing import Annotated

import typer

notify_app = typer.Typer(
    name="notify",
    no_args_is_help=True,
    help="Configure push notifications (ntfy.sh).",
)


@notify_app.command("setup")
def setup(
    topic: Annotated[
        str,
        typer.Option("--topic", "-t", help="Your ntfy.sh topic: a name nobody can guess"),
    ] = "",
    server: Annotated[
        str,
        typer.Option("--server", help="ntfy server URL (default: https://ntfy.sh)"),
    ] = "",
    test: Annotated[
        bool,
        typer.Option("--test/--no-test", help="Send a test notification after saving"),
    ] = True,
) -> None:
    """Persist ntfy settings so notifications work in every process, including Claude Code.

    Run this once. The topic is saved to ~/.config/researchforge/notify.json
    and read automatically by every ResearchForge command — no need to export
    NTFY_TOPIC every session.

    Examples:
        researchforge notify setup --topic <a-name-nobody-can-guess>
        researchforge notify setup --topic myteam-rf --server https://ntfy.myhost.com
    """
    from researchforge.notifications.service import _NOTIFY_CONFIG_PATH, save_notify_config

    if not topic:
        # Interactive prompt
        typer.echo("Your ntfy topic is shown on ntfy.sh after you subscribe.")
        typer.echo("Pick a name nobody can guess: anyone who knows it can read your alerts.")
        topic = typer.prompt("ntfy topic").strip()

    if not topic:
        typer.secho("❌ Topic is required.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    save_notify_config(topic=topic, server=server)
    typer.echo(f"✅ Saved to {_NOTIFY_CONFIG_PATH}")
    typer.echo(f"   topic : {topic}")
    if server:
        typer.echo(f"   server: {server}")

    if test:
        typer.echo("\n📬 Sending test notification...")
        from researchforge.notifications.service import Channel, Notification, send_notification

        sent = send_notification(
            Notification(
                title="ResearchForge: notifications configured ✓",
                message=(
                    f"Topic '{topic}' is working. You'll be notified on every experiment, "
                    "hypothesis, and submission."
                ),
                channel=Channel.NTFY,
                tags=["white_check_mark"],
            )
        )
        if sent:
            typer.secho("✅ Test notification sent — check your device.", fg=typer.colors.GREEN)
        else:
            typer.secho(
                "⚠️  Notification failed. Check the topic name and your internet connection.",
                fg=typer.colors.YELLOW,
            )

    typer.echo(
        "\nNotifications will now fire automatically for:\n"
        "  • Every experiment result (execute-round / autorun)\n"
        "  • New hypotheses (import / autorun synthesis)\n"
        "  • Better kernel found (overnight watcher)\n"
        "  • Submission approved (experiment approve)"
    )


@notify_app.command("test")
def test_cmd() -> None:
    """Send a test notification using the currently configured topic."""
    from researchforge.notifications.service import (
        Channel,
        Notification,
        _ntfy_server,
        _ntfy_topic,
        send_notification,
    )

    topic = _ntfy_topic()
    server = _ntfy_server()
    if not topic:
        typer.secho(
            "❌ No topic set — run: researchforge notify setup --topic <topic>", fg=typer.colors.RED
        )
        raise typer.Exit(1)
    typer.echo(f"Topic : {topic}")
    typer.echo(f"Server: {server}")
    sent = send_notification(
        Notification(
            title="ResearchForge: test",
            message="This is a test notification.",
            channel=Channel.NTFY,
            tags=["bell"],
        )
    )
    if sent:
        typer.secho("✅ Sent.", fg=typer.colors.GREEN)
    else:
        typer.secho("❌ Failed — check topic and internet.", fg=typer.colors.RED)


@notify_app.command("status")
def status() -> None:
    """Show current notification configuration."""
    import os

    from researchforge.notifications.service import (
        _NOTIFY_CONFIG_PATH,
        _load_notify_config,
        _ntfy_server,
        _ntfy_topic,
    )

    cfg = _load_notify_config()
    env_topic = os.environ.get("NTFY_TOPIC", "")
    env_server = os.environ.get("NTFY_SERVER", "")

    typer.echo(f"Config file : {_NOTIFY_CONFIG_PATH}")
    typer.echo(f"  exists    : {_NOTIFY_CONFIG_PATH.exists()}")
    typer.echo(f"  topic     : {cfg.get('ntfy_topic', '(not set)')}")
    typer.echo(f"  server    : {cfg.get('ntfy_server', '(not set)')}")
    typer.echo(f"NTFY_TOPIC env : {env_topic or '(not set)'}")
    typer.echo(f"NTFY_SERVER env: {env_server or '(not set)'}")
    typer.echo(f"\nResolved topic : {_ntfy_topic() or '(none — alerts are not sent)'}")
    typer.echo(f"Resolved server: {_ntfy_server()}")
