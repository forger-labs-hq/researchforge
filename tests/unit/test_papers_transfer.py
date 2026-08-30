"""Moving stored papers between machines: `papers export` and `papers import`."""

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.domain.paper import EvidenceStrength, Paper
from researchforge.research.cli import PAPERS_EXPORT_VERSION, read_papers_export
from researchforge.storage.db import open_project_db
from researchforge.storage.paper_repository import get_paper, list_papers, upsert_paper
from researchforge.storage.project_repository import get_project


def _paper(paper_id: str, title: str, score: float = 0.9) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=["A. Author"],
        published_at=datetime(2025, 1, 15, tzinfo=UTC),
        abstract=f"An abstract about {title}.",
        source_url=f"https://arxiv.org/abs/{paper_id}",
        categories=["cs.LG"],
        relevance_score=score,
    )


def _store(papers: list[Paper]) -> None:
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        assert project is not None
        for paper in papers:
            upsert_paper(conn, project.id, paper)


class TestExport:
    def test_it_writes_every_stored_paper(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        _store([_paper("arxiv:2501.00001", "Routing"), _paper("arxiv:2501.00002", "Gating")])
        destination = tmp_path / "papers.json"
        result = cli_runner.invoke(app, ["papers", "export", str(destination)])
        assert result.exit_code == 0, result.output

        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["schema_version"] == PAPERS_EXPORT_VERSION
        assert {p["paper_id"] for p in payload["papers"]} == {
            "arxiv:2501.00001",
            "arxiv:2501.00002",
        }

    def test_an_empty_project_has_nothing_to_export(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        result = cli_runner.invoke(app, ["papers", "export", str(tmp_path / "papers.json")])
        assert result.exit_code == 1
        assert "No papers stored" in result.output

    def test_it_creates_missing_parent_directories(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        _store([_paper("arxiv:2501.00001", "Routing")])
        destination = tmp_path / "transfer" / "papers.json"
        assert cli_runner.invoke(app, ["papers", "export", str(destination)]).exit_code == 0
        assert destination.is_file()

    def test_json_output_reports_the_path_and_count(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        _store([_paper("arxiv:2501.00001", "Routing")])
        destination = tmp_path / "papers.json"
        result = cli_runner.invoke(
            app, ["papers", "export", str(destination), "--json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["path"] == str(destination)
        assert payload["paper_count"] == 1


class TestReadPapersExport:
    def test_a_valid_file_round_trips(self, tmp_path: Path) -> None:
        source = _paper("arxiv:2501.00001", "Routing")
        file = tmp_path / "papers.json"
        file.write_text(
            json.dumps(
                {
                    "schema_version": PAPERS_EXPORT_VERSION,
                    "papers": [source.model_dump(mode="json")],
                }
            ),
            encoding="utf-8",
        )
        papers = read_papers_export(file)
        assert len(papers) == 1
        assert papers[0].paper_id == "arxiv:2501.00001"
        assert papers[0].title == "Routing"

    def test_a_non_object_payload_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "papers.json"
        file.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            read_papers_export(file)

    def test_an_unknown_schema_version_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "papers.json"
        file.write_text(json.dumps({"schema_version": 99, "papers": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            read_papers_export(file)

    def test_a_missing_papers_key_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "papers.json"
        file.write_text(
            json.dumps({"schema_version": PAPERS_EXPORT_VERSION}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="non-empty list"):
            read_papers_export(file)

    def test_an_empty_paper_list_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "papers.json"
        file.write_text(
            json.dumps({"schema_version": PAPERS_EXPORT_VERSION, "papers": []}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-empty list"):
            read_papers_export(file)

    def test_an_invalid_paper_names_its_position(self, tmp_path: Path) -> None:
        good = _paper("arxiv:2501.00001", "Routing").model_dump(mode="json")
        file = tmp_path / "papers.json"
        file.write_text(
            json.dumps(
                {
                    "schema_version": PAPERS_EXPORT_VERSION,
                    "papers": [good, {"paper_id": "arxiv:2501.00002"}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"papers\[1\]"):
            read_papers_export(file)

    def test_malformed_json_is_reported(self, tmp_path: Path) -> None:
        file = tmp_path / "papers.json"
        file.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            read_papers_export(file)


class TestImport:
    def test_papers_move_between_projects(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        _store([_paper("arxiv:2501.00001", "Routing"), _paper("arxiv:2501.00002", "Gating")])
        transfer = tmp_path / "papers.json"
        assert cli_runner.invoke(app, ["papers", "export", str(transfer)]).exit_code == 0

        with closing(open_project_db()) as conn:
            conn.execute("DELETE FROM papers")
            conn.commit()

        result = cli_runner.invoke(app, ["papers", "import", str(transfer)])
        assert result.exit_code == 0, result.output
        with closing(open_project_db()) as conn:
            restored = list_papers(conn)
        assert {p.paper_id for p in restored} == {"arxiv:2501.00001", "arxiv:2501.00002"}

    def test_the_scored_fields_survive_the_trip(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        annotated = _paper("arxiv:2501.00001", "Routing", score=0.77).model_copy(
            update={
                "evidence_strength": EvidenceStrength.HIGH,
                "method_summary": "Trains a router.",
                "limitations": ["single dataset"],
            }
        )
        _store([annotated])
        transfer = tmp_path / "papers.json"
        assert cli_runner.invoke(app, ["papers", "export", str(transfer)]).exit_code == 0
        with closing(open_project_db()) as conn:
            conn.execute("DELETE FROM papers")
            conn.commit()
        assert cli_runner.invoke(app, ["papers", "import", str(transfer)]).exit_code == 0

        with closing(open_project_db()) as conn:
            restored = get_paper(conn, "arxiv:2501.00001")
        assert restored is not None
        assert restored.relevance_score == 0.77
        assert restored.evidence_strength is EvidenceStrength.HIGH
        assert restored.method_summary == "Trains a router."
        assert restored.limitations == ["single dataset"]

    def test_re_importing_the_same_file_does_not_double_the_set(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        _store([_paper("arxiv:2501.00001", "Routing")])
        transfer = tmp_path / "papers.json"
        assert cli_runner.invoke(app, ["papers", "export", str(transfer)]).exit_code == 0

        assert cli_runner.invoke(app, ["papers", "import", str(transfer)]).exit_code == 0
        assert cli_runner.invoke(app, ["papers", "import", str(transfer)]).exit_code == 0

        with closing(open_project_db()) as conn:
            assert len(list_papers(conn)) == 1

    def test_it_reports_how_many_were_new(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        _store([_paper("arxiv:2501.00001", "Routing")])
        transfer = tmp_path / "papers.json"
        assert cli_runner.invoke(app, ["papers", "export", str(transfer)]).exit_code == 0

        result = cli_runner.invoke(app, ["papers", "import", str(transfer), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["imported"] == 1
        assert payload["added"] == 0
        assert payload["replaced"] == 1

    def test_a_bad_file_imports_nothing(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        good = _paper("arxiv:2501.00001", "Routing").model_dump(mode="json")
        transfer = tmp_path / "papers.json"
        transfer.write_text(
            json.dumps(
                {
                    "schema_version": PAPERS_EXPORT_VERSION,
                    "papers": [good, {"paper_id": "arxiv:2501.00002"}],
                }
            ),
            encoding="utf-8",
        )
        result = cli_runner.invoke(app, ["papers", "import", str(transfer)])
        assert result.exit_code == 1
        with closing(open_project_db()) as conn:
            assert list_papers(conn) == []

    def test_a_missing_file_is_reported(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        result = cli_runner.invoke(app, ["papers", "import", str(tmp_path / "absent.json")])
        assert result.exit_code == 1
