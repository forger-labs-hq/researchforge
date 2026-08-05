import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge import cli
from researchforge.utils.system_checks import (
    check_docker,
    check_gh,
    check_git,
    check_python,
    run_all_checks,
)


def test_check_python_ok_when_at_or_above_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    result = check_python(minimum=(3, 12))

    assert result.name == "python"
    assert result.required is True
    assert result.ok == (sys.version_info >= (3, 12))


def test_check_python_fails_below_minimum() -> None:
    result = check_python(minimum=(99, 0))

    assert result.ok is False
    assert result.hint is not None


def test_check_git_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    result = check_git()

    assert result.ok is False
    assert result.required is True
    assert result.hint is not None


def test_check_git_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="git version 2.43.0\n", stderr=""),
    )

    result = check_git()

    assert result.ok is True
    assert "2.43.0" in result.detail


def test_check_docker_missing_does_not_fail_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    result = check_docker()

    assert result.ok is False
    assert result.required is False


def test_check_gh_missing_does_not_fail_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    result = check_gh()

    assert result.ok is False
    assert result.required is False


def test_check_tool_survives_subprocess_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")

    def _raise_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=5)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    result = check_docker()

    assert result.ok is True
    assert "version unknown" in result.detail


def test_run_all_checks_returns_six_results() -> None:
    results = run_all_checks()

    names = {r.name for r in results}
    assert names == {"python", "git", "docker", "gh", "claude skills", "cursor rules"}


def test_check_claude_skills_states(isolated_project_dir: Path) -> None:
    from researchforge.claude.installer import install_skills
    from researchforge.utils.system_checks import check_claude_skills

    result = check_claude_skills()
    assert result.ok is True
    assert result.required is False
    assert result.detail == "not installed"
    assert result.hint is not None and "claude install" in result.hint

    install_skills()
    result = check_claude_skills()
    assert result.detail == "12/12 installed"

    skill = isolated_project_dir / ".claude" / "skills" / "researchforge-run" / "SKILL.md"
    skill.write_text("edited\n", encoding="utf-8")
    result = check_claude_skills()
    assert result.detail == "11/12 installed, 1 locally modified"


def test_doctor_cli_exits_zero_when_only_optional_tools_missing(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _which(name: str) -> str | None:
        return "/usr/bin/git" if name == "git" else None

    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="git version 2.43.0\n", stderr=""),
    )

    result = cli_runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0


def test_doctor_cli_exits_one_when_git_missing(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    result = cli_runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1


def test_doctor_cli_json_output_shape(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _which(name: str) -> str | None:
        return "/usr/bin/git" if name == "git" else None

    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="git version 2.43.0\n", stderr=""),
    )

    result = cli_runner.invoke(cli.app, ["doctor", "--json"])

    payload = json.loads(result.output)
    assert isinstance(payload, list)
    for entry in payload:
        assert {"name", "ok", "required", "detail", "hint"} <= entry.keys()
