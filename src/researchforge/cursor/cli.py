"""`researchforge cursor` sub-app: manage the project-level Cursor rules."""

from __future__ import annotations

import json

import typer

from researchforge.cursor.installer import (
    InstallReport,
    RuleAction,
    install_rules,
    rules_status,
    uninstall_rules,
)
from researchforge.utils.output import JsonOption

cursor_app = typer.Typer(name="cursor", no_args_is_help=True, help="Cursor IDE rules.")

ForceOption = typer.Option(
    False,
    "--force",
    help="Overwrite/remove rules even if they were modified after installation.",
)

UserOption = typer.Option(
    False,
    "--user",
    help="Target ~/.cursor/rules/ (every project on this machine) instead of this repository.",
)

_ACTION_STYLES: dict[RuleAction, tuple[str, str]] = {
    RuleAction.INSTALLED: ("rf.success", "+"),
    RuleAction.UPDATED: ("rf.accent", "↑"),
    RuleAction.UNCHANGED: ("rf.muted", "="),
    RuleAction.SKIPPED_MODIFIED: ("rf.warning", "!"),
    RuleAction.REMOVED: ("rf.error", "−"),
    RuleAction.LEFT_MODIFIED: ("rf.warning", "!"),
    RuleAction.MISSING: ("rf.muted", "?"),
    RuleAction.MODIFIED: ("rf.warning", "!"),
}


def _echo_report(report: InstallReport, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return
    from researchforge.utils.console import console

    for result in report.results:
        style, marker = _ACTION_STYLES[result.action]
        console.print(f"  [{style}]{marker}[/]  {result.rule}  [{style}]{result.action.value}[/]")


@cursor_app.command("install")
def install_command(
    force: bool = ForceOption, user: bool = UserOption, json_output: JsonOption = False
) -> None:
    """Install the ResearchForge rules into this repository's .cursor/rules/."""
    report = install_rules(force=force, user=user)
    _echo_report(report, json_output)
    if not json_output:
        from researchforge.utils.console import console

        if report.conflicts:
            console.print(
                "[rf.warning]![/] Some rules were modified after installation and were left "
                "untouched; re-run with [bold]--force[/] to overwrite them."
            )
        console.print(f"  [rf.muted]→[/]  [rf.path]{report.rules_dir}[/]")


@cursor_app.command("uninstall")
def uninstall_command(
    force: bool = ForceOption, user: bool = UserOption, json_output: JsonOption = False
) -> None:
    """Remove the installed ResearchForge rules (user-modified files are kept)."""
    report = uninstall_rules(force=force, user=user)
    _echo_report(report, json_output)
    if not json_output:
        from researchforge.utils.console import console

        left = [r for r in report.results if r.action is RuleAction.LEFT_MODIFIED]
        if left:
            console.print(
                "[rf.warning]![/] Modified rules were left in place; "
                "re-run with [bold]--force[/] to remove them too."
            )


@cursor_app.command("status")
def status_command(user: bool = UserOption, json_output: JsonOption = False) -> None:
    """Show whether each packaged rule is installed, modified, or missing."""
    report = rules_status(user=user)
    _echo_report(report, json_output)
