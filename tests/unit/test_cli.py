import re

from typer.testing import CliRunner

from researchforge.cli import app

_ANSI_ESCAPES = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip ANSI escape codes; rich emits them on CI runners even without a TTY."""
    return _ANSI_ESCAPES.sub("", output)


def test_help_lists_all_commands(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("doctor", "init", "status"):
        assert command in _plain(result.output)


def test_help_groups_commands_by_part_of_the_workflow(cli_runner: CliRunner) -> None:
    output = _plain(cli_runner.invoke(app, ["--help"]).output)

    for panel in ("Setup", "Research", "Experiments", "Results & shipping"):
        assert panel in output


def test_help_separates_the_hosted_hub_commands(cli_runner: CliRunner) -> None:
    output = _plain(cli_runner.invoke(app, ["--help"]).output)

    assert "Hub (hosted" in output
    hub_panel = output.index("Hub (hosted")
    experiments_panel = output.index("Experiments")
    assert hub_panel != experiments_panel, "hub commands must not sit with the OSS workflow"


def test_every_top_level_command_is_in_a_named_group(cli_runner: CliRunner) -> None:
    """An ungrouped command lands in rich's default "Commands" panel."""
    output = _plain(cli_runner.invoke(app, ["--help"]).output)

    assert "─ Commands ─" not in output


def test_doctor_help_mentions_json(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    assert "--json" in _plain(result.output)


def test_init_help_mentions_json(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert "--json" in _plain(result.output)


def test_status_help_mentions_json(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["status", "--help"])

    assert result.exit_code == 0
    assert "--json" in _plain(result.output)


def test_no_args_shows_help(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, [])

    output = _plain(result.output)
    assert "doctor" in output
    assert "init" in output
    assert "status" in output
