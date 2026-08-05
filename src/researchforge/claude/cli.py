"""`researchforge claude` sub-app: manage the project-level Claude skills."""

from __future__ import annotations

import json

import typer

from researchforge.claude.installer import (
    InstallReport,
    SkillAction,
    install_skills,
    skills_status,
    uninstall_skills,
)
from researchforge.utils.output import JsonOption

claude_app = typer.Typer(name="claude", no_args_is_help=True, help="Claude Code skills.")

ForceOption = typer.Option(
    False,
    "--force",
    help="Overwrite/remove skills even if they were modified after installation.",
)

UserOption = typer.Option(
    False,
    "--user",
    help="Target ~/.claude/skills/ (every project on this machine) instead of this repository.",
)

_ACTION_STYLES: dict[SkillAction, tuple[str, str]] = {
    SkillAction.INSTALLED: ("rf.success", "+"),
    SkillAction.UPDATED: ("rf.accent", "↑"),
    SkillAction.UNCHANGED: ("rf.muted", "="),
    SkillAction.SKIPPED_MODIFIED: ("rf.warning", "!"),
    SkillAction.REMOVED: ("rf.error", "−"),
    SkillAction.LEFT_MODIFIED: ("rf.warning", "!"),
    SkillAction.MISSING: ("rf.muted", "?"),
    SkillAction.MODIFIED: ("rf.warning", "!"),
}


def _echo_report(report: InstallReport, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return
    from researchforge.utils.console import console

    for result in report.results:
        style, marker = _ACTION_STYLES[result.action]
        console.print(f"  [{style}]{marker}[/]  {result.skill}  [{style}]{result.action.value}[/]")


@claude_app.command("install")
def install_command(
    force: bool = ForceOption, user: bool = UserOption, json_output: JsonOption = False
) -> None:
    """Install the ResearchForge skills into this repository's .claude/skills/."""
    report = install_skills(force=force, user=user)
    _echo_report(report, json_output)
    if not json_output:
        from researchforge.utils.console import console

        if report.conflicts:
            console.print(
                "[rf.warning]![/] Some skills were modified after installation and were left "
                "untouched; re-run with [bold]--force[/] to overwrite them."
            )
        console.print(f"  [rf.muted]→[/]  [rf.path]{report.skills_dir}[/]")


@claude_app.command("uninstall")
def uninstall_command(
    force: bool = ForceOption, user: bool = UserOption, json_output: JsonOption = False
) -> None:
    """Remove the installed ResearchForge skills (user-modified files are kept)."""
    report = uninstall_skills(force=force, user=user)
    _echo_report(report, json_output)
    if not json_output:
        from researchforge.utils.console import console

        left = [r for r in report.results if r.action is SkillAction.LEFT_MODIFIED]
        if left:
            console.print(
                "[rf.warning]![/] Modified skills were left in place; "
                "re-run with [bold]--force[/] to remove them too."
            )


@claude_app.command("status")
def status_command(user: bool = UserOption, json_output: JsonOption = False) -> None:
    """Show whether each packaged skill is installed, modified, or missing."""
    report = skills_status(user=user)
    _echo_report(report, json_output)
