import json
from contextlib import closing
from pathlib import Path

from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.config.paths import researchforge_dir
from researchforge.storage.db import get_connection


def test_init_creates_db_with_one_project_row(
    cli_runner: CliRunner, isolated_project_dir: Path
) -> None:
    result = cli_runner.invoke(app, ["init"])

    assert result.exit_code == 0

    db_file = researchforge_dir(isolated_project_dir) / "researchforge.db"
    assert db_file.is_file()

    with closing(get_connection(db_file)) as conn:
        rows = conn.execute("SELECT * FROM projects").fetchall()
    assert len(rows) == 1
    assert rows[0]["name"] == isolated_project_dir.name
    assert rows[0]["status"] == "initialized"


def test_init_is_idempotent(cli_runner: CliRunner, isolated_project_dir: Path) -> None:
    first = cli_runner.invoke(app, ["init"])
    second = cli_runner.invoke(app, ["init"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "already" in second.output.lower()

    db_file = researchforge_dir(isolated_project_dir) / "researchforge.db"
    with closing(get_connection(db_file)) as conn:
        rows = conn.execute("SELECT * FROM projects").fetchall()
    assert len(rows) == 1


def test_init_does_not_create_later_phase_subdirectories(
    cli_runner: CliRunner, isolated_project_dir: Path
) -> None:
    cli_runner.invoke(app, ["init"])

    root = researchforge_dir(isolated_project_dir)
    for name in ("worktrees", "artifacts", "papers", "reports"):
        assert not (root / name).exists()


def test_init_json_output(cli_runner: CliRunner, isolated_project_dir: Path) -> None:
    result = cli_runner.invoke(app, ["init", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == isolated_project_dir.name
    assert payload["status"] == "initialized"


class TestInitCursor:
    """`--cursor` installs the workflow rules, not only the gateway.

    The gateway alone tells Cursor the rules exist by name; without the rules
    themselves those references resolve to nothing.
    """

    def test_the_workflow_rules_land_beside_the_gateway(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        from researchforge.cursor.installer import list_packaged_rules

        result = cli_runner.invoke(app, ["init", "--cursor"])

        assert result.exit_code == 0, result.output
        rules_dir = isolated_project_dir / ".cursor" / "rules"
        assert (rules_dir / "researchforge.mdc").is_file()
        for name in list_packaged_rules():
            assert (rules_dir / f"{name}.mdc").is_file(), name

    def test_the_loop_rule_is_among_them(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        cli_runner.invoke(app, ["init", "--cursor"])

        rule = isolated_project_dir / ".cursor" / "rules" / "researchforge-autorun.mdc"
        assert rule.is_file()

    def test_the_gateway_lists_every_installed_rule(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        """A rule Cursor cannot be pointed at may as well not be installed."""
        from researchforge.cursor.installer import list_packaged_rules

        cli_runner.invoke(app, ["init", "--cursor"])

        gateway = (isolated_project_dir / ".cursor" / "rules" / "researchforge.mdc").read_text(
            encoding="utf-8"
        )
        for name in list_packaged_rules():
            assert f"@{name}" in gateway, name

    def test_json_reports_what_was_installed(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(app, ["init", "--cursor", "--json"])

        payload = json.loads(result.output)
        assert payload["cursor_gateway"]["action"] == "installed"
        installed = {r["rule"] for r in payload["cursor_rules"]["results"]}
        assert "researchforge-autorun" in installed

    def test_reinstalling_changes_nothing(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        cli_runner.invoke(app, ["init", "--cursor"])
        result = cli_runner.invoke(app, ["init", "--cursor", "--json"])

        payload = json.loads(result.output)
        assert payload["cursor_gateway"]["action"] == "unchanged"
        assert all(r["action"] == "unchanged" for r in payload["cursor_rules"]["results"])
