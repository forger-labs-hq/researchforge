"""Dependency checks backing the `researchforge doctor` command."""

from __future__ import annotations

import shutil
import subprocess
import sys

from pydantic import BaseModel

PYTHON_MINIMUM: tuple[int, int] = (3, 12)


class CheckResult(BaseModel):
    name: str
    ok: bool
    required: bool
    detail: str
    hint: str | None = None


def check_python(minimum: tuple[int, int] = PYTHON_MINIMUM) -> CheckResult:
    current = (sys.version_info.major, sys.version_info.minor)
    ok = current >= minimum
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return CheckResult(
        name="python",
        ok=ok,
        required=True,
        detail=f"Python {version_str}",
        hint=None if ok else f"Python {minimum[0]}.{minimum[1]}+ is required.",
    )


def _check_tool_version(
    name: str, *, required: bool, hint: str, version_args: tuple[str, ...] = ("--version",)
) -> CheckResult:
    executable = shutil.which(name)
    if executable is None:
        return CheckResult(name=name, ok=False, required=required, detail="not found", hint=hint)

    try:
        result = subprocess.run(  # noqa: S603
            [executable, *version_args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return CheckResult(
            name=name,
            ok=True,
            required=required,
            detail="found (version unknown)",
        )

    detail = (result.stdout or result.stderr or "found").strip().splitlines()[0]
    return CheckResult(name=name, ok=True, required=required, detail=detail)


def check_git() -> CheckResult:
    return _check_tool_version(
        "git", required=True, hint="Install Git: https://git-scm.com/downloads"
    )


def check_docker() -> CheckResult:
    result = _check_tool_version(
        "docker",
        required=False,
        hint="Install Docker for stronger experiment isolation: https://docs.docker.com/get-docker/",
    )
    if not result.ok:
        return result

    from researchforge.execution.environment import probe_docker

    probe = probe_docker()
    if probe.daemon_running:
        detail = f"{result.detail} (daemon running)"
    else:
        detail = f"{result.detail} (daemon NOT running)"
    return result.model_copy(update={"detail": detail})


def check_gh() -> CheckResult:
    return _check_tool_version(
        "gh",
        required=False,
        hint="Install the GitHub CLI to enable draft PR creation: https://cli.github.com/",
    )


def check_claude_skills() -> CheckResult:
    """Informational: whether the Claude Code skills are installed here."""
    from researchforge.claude.installer import SkillAction, skills_status

    states = [result.action for result in skills_status().results]
    installed = sum(1 for a in states if a is SkillAction.UNCHANGED)
    modified = sum(1 for a in states if a is SkillAction.MODIFIED)
    if installed + modified == 0:
        return CheckResult(
            name="claude skills",
            ok=True,
            required=False,
            detail="not installed",
            hint="Run `researchforge claude install` to use ResearchForge from Claude Code.",
        )
    detail = f"{installed}/{len(states)} installed"
    if modified:
        detail += f", {modified} locally modified"
    return CheckResult(name="claude skills", ok=True, required=False, detail=detail)


def check_cursor_rules() -> CheckResult:
    """Informational: whether the Cursor rules are installed here."""
    from researchforge.cursor.installer import RuleAction, rules_status

    states = [result.action for result in rules_status().results]
    installed = sum(1 for a in states if a is RuleAction.UNCHANGED)
    modified = sum(1 for a in states if a is RuleAction.MODIFIED)
    if installed + modified == 0:
        return CheckResult(
            name="cursor rules",
            ok=True,
            required=False,
            detail="not installed",
            hint="Run `researchforge cursor install` to use ResearchForge from Cursor.",
        )
    detail = f"{installed}/{len(states)} installed"
    if modified:
        detail += f", {modified} locally modified"
    return CheckResult(name="cursor rules", ok=True, required=False, detail=detail)


def run_all_checks() -> list[CheckResult]:
    return [
        check_python(),
        check_git(),
        check_docker(),
        check_gh(),
        check_claude_skills(),
        check_cursor_rules(),
    ]
