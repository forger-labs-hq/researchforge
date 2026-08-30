"""Monitoring server: pages, API, read-only guarantee, and the serve CLI."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.cli import app as cli_app
from researchforge.domain.experiment import ExperimentExecution
from researchforge.server.data import read_state


@pytest.fixture
def client(validated_project: Path, isolated_project_dir: Path):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from researchforge.server.app import create_app

    return TestClient(create_app())


class TestPages:
    def test_overview(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/")
        assert response.status_code == 200
        assert "Next action" in response.text
        assert "ship branch" in response.text  # validated fixture's next step
        assert "monitor · read-only" in response.text
        assert "content='30'" in response.text  # no run in progress -> slow refresh

    def test_research_empty_state_is_honest(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/research")
        assert response.status_code == 200
        assert "No research sessions yet" in response.text
        assert "No hypotheses imported yet" not in response.text  # fixture has hyp-001

    def test_experiments_live_table(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/experiments")
        assert response.status_code == 200
        assert "run-001" in response.text
        assert "validated" in response.text
        assert "rejected" in response.text
        assert "p95_latency_ms" in response.text  # rejection reason shown

    def test_dashboard_page_has_charts_and_nav(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "Progress —" in response.text and "Funnel" in response.text
        assert "<nav>" in response.text
        assert "http-equiv='refresh'" in response.text


class TestSessions:
    def test_search_sessions_group_their_papers(
        self, cli_runner: CliRunner, initialized_project: Path, patched_arxiv: None
    ) -> None:
        from contextlib import closing

        from fastapi.testclient import TestClient

        from researchforge.server.app import create_app
        from researchforge.storage.db import open_project_db
        from researchforge.storage.paper_repository import list_search_runs, papers_for_search_run

        assert (
            cli_runner.invoke(cli_app, ["research", "search", "-q", "all:routing"]).exit_code == 0
        )

        with closing(open_project_db()) as conn:
            runs = list_search_runs(conn)
            assert len(runs) == 1
            linked = papers_for_search_run(conn, str(runs[0]["run_id"]))
        assert linked  # the search recorded which papers it selected

        response = TestClient(create_app()).get("/research")
        assert response.status_code == 200
        text = response.text
        assert "Research sessions (1)" in text
        assert "Search session #1" in text
        assert "all:routing" in text  # the session's query shown
        assert re.search(r"<details class='session' id='[^']+' open>", text)
        assert linked[0] in text  # a session paper rendered inside
        assert "All stored papers" in text

    def test_legacy_search_run_shows_fallback(
        self, cli_runner: CliRunner, initialized_project: Path, patched_arxiv: None
    ) -> None:
        import sqlite3

        from fastapi.testclient import TestClient

        from researchforge.server.app import create_app

        assert (
            cli_runner.invoke(cli_app, ["research", "search", "-q", "all:routing"]).exit_code == 0
        )
        conn = sqlite3.connect(initialized_project / ".researchforge" / "researchforge.db")
        conn.execute("DELETE FROM search_run_papers")  # simulate a pre-v6 recording
        conn.commit()
        conn.close()

        response = TestClient(create_app()).get("/research")
        assert "not attributed" in response.text
        assert "earlier ResearchForge version" in response.text

    def test_landscape_and_hypothesis_full_detail(
        self, cli_runner: CliRunner, initialized_project: Path, patched_arxiv: None
    ) -> None:
        import shutil

        from fastapi.testclient import TestClient

        from researchforge.server.app import create_app

        artifacts = Path(__file__).parent.parent / "fixtures" / "artifacts"
        assert (
            cli_runner.invoke(cli_app, ["research", "search", "-q", "all:routing"]).exit_code == 0
        )
        for fixture, command in (
            ("landscape_valid.yaml", ["research", "landscape", "--import"]),
            ("hypotheses_valid.yaml", ["hypotheses", "import"]),
        ):
            target = initialized_project / fixture
            shutil.copy(artifacts / fixture, target)
            assert cli_runner.invoke(cli_app, [*command, str(target)]).exit_code == 0

        text = TestClient(create_app()).get("/research").text

        # Directions render every recorded facet, including the untouched zones.
        assert "Underexplored aspects" in text
        assert "Joint cost and latency constraints appear underexplored" in text
        assert "Established findings" in text
        assert "Contradictions" in text
        assert "Evidence claims" in text
        # Hypotheses render the full record.
        assert "Rationale:" in text
        assert "Proposed experiment:" in text
        assert "feasibility: high" in text
        assert "Supporting papers" in text

    def test_session_drilldown_page(
        self, cli_runner: CliRunner, initialized_project: Path, patched_arxiv: None
    ) -> None:
        import shutil
        from contextlib import closing

        from fastapi.testclient import TestClient

        from researchforge.server.app import create_app
        from researchforge.storage.db import open_project_db
        from researchforge.storage.paper_repository import list_search_runs

        artifacts = Path(__file__).parent.parent / "fixtures" / "artifacts"
        assert (
            cli_runner.invoke(cli_app, ["research", "search", "-q", "all:routing"]).exit_code == 0
        )
        for fixture, command in (
            ("landscape_valid.yaml", ["research", "landscape", "--import"]),
            ("hypotheses_valid.yaml", ["hypotheses", "import"]),
        ):
            target = initialized_project / fixture
            shutil.copy(artifacts / fixture, target)
            assert cli_runner.invoke(cli_app, [*command, str(target)]).exit_code == 0

        with closing(open_project_db()) as conn:
            run_id = str(list_search_runs(conn)[0]["run_id"])

        client = TestClient(create_app())
        response = client.get(f"/sessions/{run_id}")
        assert response.status_code == 200
        text = response.text
        assert "Search session #1" in text
        assert "work locations" in text
        assert str(initialized_project.resolve()) in text
        assert "Papers this session selected" in text
        # The fixture landscape/hypotheses cite the searched papers.
        assert "Directions citing this session's papers" in text
        assert "Hypotheses citing this session's papers" in text
        assert "Underexplored aspects" in text  # full direction detail reused

        assert client.get("/sessions/nope").status_code == 404

        # The research page links into the session.
        research = client.get("/research").text
        assert f"/sessions/{run_id}" in research

    def test_state_keeper_script_and_ids(self, client) -> None:  # type: ignore[no-untyped-def]
        import re

        response = client.get("/experiments")
        assert "sessionStorage" in response.text  # inline state-keeper present
        details = re.findall(r"<details class='session'[^>]*>", response.text)
        assert details and all("id='" in d for d in details)
        assert "content='30'" in response.text  # idle refresh slowed to 30s

    def test_experiment_runs_are_collapsible_sessions(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/experiments")
        text = response.text
        assert text.count("<details class='session'") >= 1
        assert re.search(r"<details class='session' id='[^']+' open>", text)  # latest expanded
        assert "full history" in text

    def test_overview_counts_sessions(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/")
        assert "research sessions" in response.text


class TestRunHistory:
    def test_run_detail_timeline(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/runs/run-001")
        assert response.status_code == 200
        text = response.text
        assert "Execution timeline" in text
        # All funnel stages appear, including validation attempts.
        for stage in ("screening", "full", "validation"):
            assert stage in text
        assert "exp-001" in text and "exp-002" in text
        assert "0.85" in text  # winner's measured value
        assert "violated: p95_latency_ms" in text  # the loser's constraint result
        assert "charts for this run" in text

    def test_unknown_run_404(self, client) -> None:  # type: ignore[no-untyped-def]
        assert client.get("/runs/run-999").status_code == 404
        assert client.get("/dashboard?run=run-999").status_code == 404

    def test_dashboard_run_selection(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/dashboard?run=run-001")
        assert response.status_code == 200
        assert "run-001" in response.text

    def test_experiments_page_links_to_history(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/experiments")
        assert "/runs/run-001" in response.text

    def test_empty_states_show_next_action(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        from fastapi.testclient import TestClient

        from researchforge.server.app import create_app

        fresh = TestClient(create_app())
        experiments = fresh.get("/experiments")
        assert "nothing here yet" in experiments.text
        assert "experiment plan" in experiments.text  # the actual next action

        # Dashboard exists (contract+baseline) so it renders its own empty state.
        dashboard = fresh.get("/dashboard")
        assert dashboard.status_code == 200
        assert "No experiment runs recorded yet" in dashboard.text


class TestApi:
    def test_state_snapshot(self, client) -> None:  # type: ignore[no-untyped-def]
        payload = client.get("/api/state").json()
        assert payload["project"]["status"]
        assert len(payload["experiments"]) == 2
        assert payload["next_action"]

    def test_run_manifest(self, client) -> None:  # type: ignore[no-untyped-def]
        payload = client.get("/api/runs/run-001").json()
        parsed = [ExperimentExecution.model_validate(item) for item in payload]
        assert parsed and all(e.run_id == "run-001" for e in parsed)
        assert client.get("/api/runs/run-999").status_code == 404

    def test_read_only_and_get_only(self, client, isolated_project_dir: Path) -> None:  # type: ignore[no-untyped-def]
        db_file = isolated_project_dir / ".researchforge" / "researchforge.db"
        before = hashlib.sha256(db_file.read_bytes()).hexdigest()

        for route in ("/", "/research", "/experiments", "/dashboard", "/api/state"):
            assert client.get(route).status_code == 200

        assert hashlib.sha256(db_file.read_bytes()).hexdigest() == before

        from fastapi.routing import APIRoute

        from researchforge.server.app import create_app

        methods = {m for r in create_app().routes if isinstance(r, APIRoute) for m in r.methods}
        assert methods <= {"GET", "HEAD"}


class TestRefreshAndState:
    def test_fast_refresh_while_run_in_progress(self, client, isolated_project_dir: Path) -> None:  # type: ignore[no-untyped-def]
        import sqlite3

        db_file = isolated_project_dir / ".researchforge" / "researchforge.db"
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        record = json.loads(conn.execute("SELECT record FROM experiment_runs").fetchone()["record"])
        record["status"] = "in_progress"
        conn.execute(
            "UPDATE experiment_runs SET record = ?, status = 'in_progress'",
            (json.dumps(record),),
        )
        conn.commit()
        conn.close()

        response = client.get("/")
        assert "content='3'" in response.text
        assert "in progress" in response.text

    def test_read_state_on_baseline_only_project(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        state = read_state()
        assert state.baseline is not None
        assert state.runs == []
        assert "experiment plan" in state.next_action


class TestServeCli:
    def test_uninitialized_directory_refused(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(cli_app, ["serve"])
        assert result.exit_code == 1
        assert "Not an initialized" in result.output

    def test_missing_extra_prints_install_hint(
        self,
        cli_runner: CliRunner,
        validated_project: Path,
        isolated_project_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def no_uvicorn(name: str, *args: object, **kwargs: object) -> object:
            if name == "uvicorn":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", no_uvicorn)
        result = cli_runner.invoke(cli_app, ["serve"])
        assert result.exit_code == 1
        assert "researchforge[serve]" in result.output

    def test_serve_runs_uvicorn_and_warns_on_public_host(
        self,
        cli_runner: CliRunner,
        validated_project: Path,
        isolated_project_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import uvicorn

        calls: list[dict[str, object]] = []
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))

        result = cli_runner.invoke(cli_app, ["serve"])
        assert result.exit_code == 0, result.output
        assert calls[-1]["host"] == "127.0.0.1"
        assert "WARNING" not in result.output

        result = cli_runner.invoke(cli_app, ["serve", "--host", "0.0.0.0"])
        assert result.exit_code == 0
        assert "WARNING" in result.output


class TestExperimentDrilldown:
    def test_experiment_page(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/experiments/exp-001")
        assert response.status_code == 200
        text = response.text
        assert "exp-001" in text and "score" in text
        assert "vs baseline" in text
        assert "Executions" in text and "validation" in text
        assert "Change" in text and "src/algo.py" in text
        assert "Artifacts on disk" in text
        assert client.get("/experiments/exp-999").status_code == 404

    def test_dashboard_graph_links_to_drilldown(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/dashboard")
        assert "href='/experiments/exp-001'" in response.text

    def test_overview_locations(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/")
        assert "Locations" in response.text
        assert "worktrees" in response.text
        assert "project root" in response.text
        # Paths shown are absolute, not confusing relative ones.
        assert str(Path.cwd().resolve()) in response.text

    def test_run_page_shows_work_locations(self, client) -> None:  # type: ignore[no-untyped-def]
        text = client.get("/runs/run-001").text
        assert "work locations" in text
        expected = Path.cwd().resolve() / ".researchforge" / "artifacts" / "experiments" / "run-001"
        assert str(expected) in text


class TestLiveExperimentGraph:
    def test_the_experiments_page_renders_the_graph(self, client) -> None:  # type: ignore[no-untyped-def]
        text = client.get("/experiments").text
        assert "Experiment graph" in text
        assert "data-graph-node='baseline'" in text
        assert "data-graph-node='exp-001'" in text

    def test_every_card_is_a_link_to_its_experiment(self, client) -> None:  # type: ignore[no-untyped-def]
        text = client.get("/experiments").text
        assert "href='/experiments/exp-001'" in text

    def test_the_graph_explains_its_own_notation(self, client) -> None:  # type: ignore[no-untyped-def]
        text = client.get("/experiments").text
        assert "LEGEND" in text
        assert "merge (multi-parent)" in text

    def test_a_wide_graph_scrolls_instead_of_shrinking(self, client) -> None:  # type: ignore[no-untyped-def]
        text = client.get("/experiments").text
        assert ".graph { overflow-x: auto" in text

    def test_a_project_without_a_baseline_shows_no_graph(
        self, cli_runner: CliRunner, contracted_project: Path, isolated_project_dir: Path
    ) -> None:
        from researchforge.server.pages import project_graph

        assert project_graph(read_state()) == ""


class TestRoundsByExperiment:
    def test_experiments_are_mapped_to_the_round_that_ran_them(self) -> None:
        from datetime import UTC, datetime

        from researchforge.autorun.state import (
            AutorunState,
            RoundRecord,
            rounds_by_experiment,
        )

        now = datetime.now(UTC)
        state = AutorunState(
            started_at=now,
            updated_at=now,
            rounds=[
                RoundRecord(
                    round_num=1,
                    promising=["exp-001"],
                    rejected=["exp-002"],
                    completed_at=now,
                ),
                RoundRecord(
                    round_num=2,
                    promising=["exp-003"],
                    failed=["exp-004"],
                    completed_at=now,
                ),
            ],
        )

        assert rounds_by_experiment(state) == {
            "exp-001": 1,
            "exp-002": 1,
            "exp-003": 2,
            "exp-004": 2,
        }

    def test_a_project_that_never_ran_a_loop_has_no_rounds(self) -> None:
        from researchforge.autorun.state import rounds_by_experiment

        assert rounds_by_experiment(None) == {}


class TestPathsCli:
    def test_paths_json(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(cli_app, ["paths", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        for key in (
            "repo_root",
            "state_dir",
            "database",
            "worktrees",
            "artifacts",
            "reports",
            "research_output",
        ):
            assert key in payload

    def test_paths_respects_configured_research_output_dir(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        config = isolated_project_dir / ".researchforge" / "config.json"
        config.write_text(json.dumps({"research_output_dir": "out/bundle"}), encoding="utf-8")
        result = cli_runner.invoke(cli_app, ["paths", "--json"])
        payload = json.loads(result.output)
        assert payload["research_output"].endswith("out/bundle")

    def test_uninitialized_refused(self, cli_runner: CliRunner, isolated_project_dir: Path) -> None:
        result = cli_runner.invoke(cli_app, ["paths"])
        assert result.exit_code == 1

    def test_paths_are_absolute(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(cli_app, ["paths", "--json"])
        payload = json.loads(result.output)
        assert all(Path(value).is_absolute() for value in payload.values())


class TestProjectDirOption:
    def test_dir_option_targets_another_directory(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        """`researchforge -C <folder> init` creates the project in that folder."""
        elsewhere = isolated_project_dir / "some_new_folder"
        elsewhere.mkdir()
        result = cli_runner.invoke(cli_app, ["-C", str(elsewhere), "init"])
        assert result.exit_code == 0, result.output
        assert (elsewhere / ".researchforge").is_dir()
        assert not (isolated_project_dir / ".researchforge").exists()

        paths = cli_runner.invoke(cli_app, ["--dir", str(elsewhere), "paths", "--json"])
        assert paths.exit_code == 0, paths.output
        assert json.loads(paths.output)["repo_root"] == str(elsewhere.resolve())

    def test_dir_option_rejects_missing_directory(
        self, cli_runner: CliRunner, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(cli_app, ["-C", str(isolated_project_dir / "nope"), "status"])
        assert result.exit_code != 0
