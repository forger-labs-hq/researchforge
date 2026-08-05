"""Global registry, walk-up discovery, and the hub dashboard."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.cli import app as cli_app
from researchforge.config.registry import (
    load_registry,
    register_project,
    registry_path,
    touch_project,
)


class TestRegistry:
    def test_register_and_upsert(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        (root / ".researchforge").mkdir(parents=True)
        first = register_project(root)
        assert first.slug == "proj"
        assert first.path == str(root.resolve())

        again = register_project(root)
        assert again.slug == first.slug
        assert len(load_registry()) == 1
        assert again.last_active >= first.last_active

    def test_slug_collision_gets_suffix(self, tmp_path: Path) -> None:
        a = tmp_path / "one" / "proj"
        b = tmp_path / "two" / "proj"
        for root in (a, b):
            (root / ".researchforge").mkdir(parents=True)
        assert register_project(a).slug == "proj"
        assert register_project(b).slug == "proj-2"

    def test_corrupt_registry_tolerated(self, tmp_path: Path) -> None:
        registry_path().parent.mkdir(parents=True, exist_ok=True)
        registry_path().write_text("not json at all", encoding="utf-8")
        assert load_registry() == []
        root = tmp_path / "proj"
        (root / ".researchforge").mkdir(parents=True)
        assert register_project(root).slug == "proj"

    def test_touch_ignores_uninitialized(self, tmp_path: Path) -> None:
        assert touch_project(tmp_path / "nothing-here") is None
        assert load_registry() == []


class TestWalkUpDiscovery:
    def test_command_in_subfolder_finds_project(
        self,
        cli_runner: CliRunner,
        funnel_project: Path,
        isolated_project_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sub = isolated_project_dir / "cloned-repo" / "src"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        result = cli_runner.invoke(cli_app, ["status"])
        assert result.exit_code == 0, result.output
        assert f"Using project at {isolated_project_dir.resolve()}" in result.stderr

    def test_init_is_exempt_but_warns_about_ancestor(
        self,
        cli_runner: CliRunner,
        funnel_project: Path,
        isolated_project_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sub = isolated_project_dir / "cloned-repo"
        sub.mkdir()
        monkeypatch.chdir(sub)
        result = cli_runner.invoke(cli_app, ["init"])
        assert result.exit_code == 0, result.output
        assert "an initialized project already exists at" in result.stderr
        assert (sub / ".researchforge").is_dir()  # cwd honored, no walk-up

    def test_init_registers_project(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        assert cli_runner.invoke(cli_app, ["init"]).exit_code == 0
        entries = load_registry()
        assert [entry.path for entry in entries] == [str(isolated_project_dir.resolve())]

    def test_commands_touch_registry(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        assert cli_runner.invoke(cli_app, ["status"]).exit_code == 0
        assert [entry.path for entry in load_registry()] == [str(isolated_project_dir.resolve())]


class TestAutoSpawn:
    def test_commands_ensure_hub_unless_disabled(
        self,
        cli_runner: CliRunner,
        isolated_project_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import researchforge.server.monitor as monitor

        spawned: list[int] = []
        monkeypatch.setattr(
            monitor,
            "spawn_background_hub",
            lambda host, port: spawned.append(port),
        )
        monkeypatch.delenv("RESEARCHFORGE_NO_HUB")
        assert cli_runner.invoke(cli_app, ["doctor"]).exit_code == 0
        assert len(spawned) == 1

    def test_no_hub_env_disables_autospawn(
        self,
        cli_runner: CliRunner,
        isolated_project_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import researchforge.server.monitor as monitor

        spawned: list[int] = []
        monkeypatch.setattr(
            monitor,
            "spawn_background_hub",
            lambda host, port: spawned.append(port),
        )
        assert cli_runner.invoke(cli_app, ["doctor"]).exit_code == 0
        assert spawned == []


class TestHubApp:
    @pytest.fixture
    def hub_client(self, validated_project: Path, isolated_project_dir: Path):  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient

        from researchforge.server.hub import create_hub_app

        register_project(isolated_project_dir)
        return TestClient(create_hub_app())

    def test_home_lists_projects_with_locations(
        self, hub_client, isolated_project_dir: Path
    ) -> None:  # type: ignore[no-untyped-def]
        response = hub_client.get("/")
        assert response.status_code == 200
        text = response.text
        assert "Projects" in text
        assert str(isolated_project_dir.resolve()) in text
        slug = load_registry()[0].slug
        assert f"href='/p/{slug}/'" in text

    def test_project_pages_under_prefix(self, hub_client) -> None:  # type: ignore[no-untyped-def]
        slug = load_registry()[0].slug
        overview = hub_client.get(f"/p/{slug}/")
        assert overview.status_code == 200
        assert f"href='/p/{slug}/experiments'" in overview.text
        assert "⌂ all projects" in overview.text

        runs = hub_client.get(f"/p/{slug}/runs/run-001")
        assert runs.status_code == 200
        assert f"href='/p/{slug}/dashboard?run=run-001'" in runs.text

        dashboard = hub_client.get(f"/p/{slug}/dashboard")
        assert dashboard.status_code == 200
        assert f"href='/p/{slug}/experiments/exp-001'" in dashboard.text

        experiment = hub_client.get(f"/p/{slug}/experiments/exp-001")
        assert experiment.status_code == 200

        api = hub_client.get(f"/p/{slug}/api/state")
        assert api.status_code == 200
        assert "link_prefix" not in api.json()

    def test_unknown_slug_404(self, hub_client) -> None:  # type: ignore[no-untyped-def]
        assert hub_client.get("/p/nope/").status_code == 404

    def test_missing_project_shown_honestly(self, hub_client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        gone = tmp_path / "was-deleted"
        (gone / ".researchforge").mkdir(parents=True)
        register_project(gone)
        (gone / ".researchforge").rmdir()
        text = hub_client.get("/").text
        assert "missing" in text
        assert str(gone.resolve()) in text

    def test_single_project_serve_links_unprefixed(
        self, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        from fastapi.testclient import TestClient

        from researchforge.server.app import create_app

        client = TestClient(create_app())
        text = client.get("/").text
        assert "href='/experiments'" in text
        assert "all projects" not in text

    def test_empty_registry_home(self, isolated_project_dir: Path) -> None:
        from fastapi.testclient import TestClient

        from researchforge.server.hub import create_hub_app

        client = TestClient(create_hub_app())
        text = client.get("/").text
        assert "No projects yet" in text


class TestHubCli:
    def test_status_and_stop_without_hub(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        assert "No hub running" in cli_runner.invoke(cli_app, ["hub", "--status"]).output
        assert "No hub running" in cli_runner.invoke(cli_app, ["hub", "--stop"]).output

    def test_hub_json_record_roundtrip(self, isolated_project_dir: Path) -> None:
        from researchforge.server.monitor import hub_record_path, read_hub

        assert read_hub() is None
        hub_record_path().parent.mkdir(parents=True, exist_ok=True)
        hub_record_path().write_text(
            json.dumps(
                {
                    "pid": 1,  # launchd: alive on any unix
                    "url": "http://127.0.0.1:9000/",
                    "host": "127.0.0.1",
                    "port": 9000,
                    "started_at": "2026-07-21T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        record = read_hub()
        assert record is not None and record.port == 9000
        hub_record_path().write_text("garbage", encoding="utf-8")
        assert read_hub() is None
