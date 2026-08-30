"""Results-grounded re-synthesis: what the AI is told, and what gets stored."""

import shutil
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.ai.service import build_results_context
from researchforge.cli import app
from researchforge.config.settings import load_settings
from researchforge.domain.contract import MetricDirection
from researchforge.domain.experiment import (
    Decision,
    DecisionOutcome,
    Experiment,
    ExperimentStatus,
)
from researchforge.research.importers import import_additional_hypotheses
from researchforge.research.research_log import build_measured_summary, results_instructions
from researchforge.storage.db import open_project_db
from researchforge.storage.hypothesis_repository import list_hypotheses, next_hypothesis_ids
from researchforge.storage.project_repository import get_project

ARTIFACTS = Path(__file__).parent.parent / "fixtures" / "artifacts"

# hyp-001 in the fixture is entropy-threshold routing for inference cost. The
# first entry here says the same thing in different words; the second is a new
# idea about batching.
RESYNTH_ARTIFACT = """\
hypotheses:
  - hypothesis_id: hyp-001
    title: Entropy-threshold routing reduces inference cost
    claim: >
      Routing queries by an entropy threshold can reduce average inference cost
      without raising P95 latency, under the conditions reviewed.
    rationale: Re-proposed from the same literature.
    supporting_paper_ids:
      - arxiv:2401.12345
    feasibility: high
    estimated_effort: medium
    novelty_confidence: unknown
    proposed_experiment: Implement entropy-threshold routing and measure cost.
  - hypothesis_id: hyp-002
    title: Continuous batching raises throughput
    claim: Scheduling requests continuously increases tokens served per second.
    rationale: Serving-systems literature reports queue-level gains.
    supporting_paper_ids:
      - arxiv:2312.00001
    feasibility: medium
    estimated_effort: low
    novelty_confidence: unknown
    proposed_experiment: Enable continuous batching and measure throughput.
"""


@pytest.fixture
def project_with_hypotheses(
    cli_runner: CliRunner,
    initialized_project: Path,
    patched_arxiv: None,
) -> Path:
    """A synthesized project: papers stored and hyp-001..003 imported."""
    assert cli_runner.invoke(app, ["research", "search", "-q", "all:routing"]).exit_code == 0
    artifact = initialized_project / "hypotheses_valid.yaml"
    shutil.copy(ARTIFACTS / "hypotheses_valid.yaml", artifact)
    assert cli_runner.invoke(app, ["hypotheses", "import", str(artifact)]).exit_code == 0
    return initialized_project


def _write_resynth(base: Path) -> Path:
    path = base / "resynthesized.yaml"
    path.write_text(RESYNTH_ARTIFACT, encoding="utf-8")
    return path


def _experiment(
    experiment_id: str, summary: str, status: ExperimentStatus, reason: str | None = None
) -> Experiment:
    now = datetime.now(UTC)
    return Experiment(
        experiment_id=experiment_id,
        plan_id="plan-001",
        hypothesis_id="hyp-001",
        title=f"Variant {experiment_id}",
        change_summary=summary,
        patch_text="p",
        patch_sha256="0" * 64,
        status=status,
        decision=(
            Decision(outcome=DecisionOutcome.REJECT, reason=reason)
            if reason is not None
            else None
        ),
        created_at=now,
        updated_at=now,
    )


class TestNextHypothesisIds:
    def test_continues_after_the_highest_stored(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        with closing(open_project_db()) as conn:
            assert next_hypothesis_ids(conn, 2) == ["hyp-004", "hyp-005"]

    def test_an_empty_project_starts_at_one(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        with closing(open_project_db()) as conn:
            assert next_hypothesis_ids(conn, 1) == ["hyp-001"]


class TestAdditiveImport:
    def test_stored_hypotheses_survive(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        """A later round must not delete the hypotheses whose plans are on record."""
        path = _write_resynth(project_with_hypotheses)

        with closing(open_project_db()) as conn:
            project = get_project(conn)
            assert project is not None
            import_additional_hypotheses(conn, path, project.id, load_settings())
            stored = {h.hypothesis_id for h in list_hypotheses(conn)}

        assert {"hyp-001", "hyp-002", "hyp-003"} <= stored

    def test_the_new_idea_is_added_under_a_fresh_id(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        path = _write_resynth(project_with_hypotheses)

        with closing(open_project_db()) as conn:
            project = get_project(conn)
            assert project is not None
            additions = import_additional_hypotheses(conn, path, project.id, load_settings())

        assert additions.added == ["hyp-004"]

    def test_the_restated_idea_is_skipped_and_named(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        path = _write_resynth(project_with_hypotheses)

        with closing(open_project_db()) as conn:
            project = get_project(conn)
            assert project is not None
            additions = import_additional_hypotheses(conn, path, project.id, load_settings())

        assert additions.restated == ["hyp-001"]
        assert any("restates hyp-001" in w for w in additions.result.warnings)

    def test_the_renumbered_hypothesis_keeps_its_content(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        path = _write_resynth(project_with_hypotheses)

        with closing(open_project_db()) as conn:
            project = get_project(conn)
            assert project is not None
            import_additional_hypotheses(conn, path, project.id, load_settings())
            added = next(h for h in list_hypotheses(conn) if h.hypothesis_id == "hyp-004")

        assert added.title == "Continuous batching raises throughput"
        assert added.supporting_paper_ids == ["arxiv:2312.00001"]

    def test_importing_the_same_batch_twice_adds_nothing(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        path = _write_resynth(project_with_hypotheses)

        with closing(open_project_db()) as conn:
            project = get_project(conn)
            assert project is not None
            import_additional_hypotheses(conn, path, project.id, load_settings())
            second = import_additional_hypotheses(conn, path, project.id, load_settings())
            stored = list_hypotheses(conn)

        assert second.added == []
        assert len(stored) == 4
        assert any("nothing was added" in w for w in second.result.warnings)

    def test_a_malformed_artifact_changes_nothing(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        path = project_with_hypotheses / "broken.yaml"
        path.write_text("hypotheses: [{hypothesis_id: hyp-009}]\n", encoding="utf-8")

        with closing(open_project_db()) as conn:
            project = get_project(conn)
            assert project is not None
            additions = import_additional_hypotheses(conn, path, project.id, load_settings())
            stored = list_hypotheses(conn)

        assert not additions.ok
        assert len(stored) == 3

    def test_an_unknown_paper_reference_is_refused(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        path = project_with_hypotheses / "ghost.yaml"
        path.write_text(
            RESYNTH_ARTIFACT.replace("arxiv:2312.00001", "arxiv:9999.99999"), encoding="utf-8"
        )

        with closing(open_project_db()) as conn:
            project = get_project(conn)
            assert project is not None
            additions = import_additional_hypotheses(conn, path, project.id, load_settings())

        assert not additions.ok
        assert any("arxiv:9999.99999" in error for error in additions.result.errors)


class TestMeasuredSummary:
    def test_names_the_baseline_and_its_direction(self) -> None:
        summary = build_measured_summary([], {}, 0.80, "f1", MetricDirection.MAXIMIZE)
        assert "Baseline **f1** = 0.8000" in summary
        assert "higher is better" in summary

    def test_a_minimized_metric_says_lower_is_better(self) -> None:
        summary = build_measured_summary([], {}, 120.0, "latency_ms", MetricDirection.MINIMIZE)
        assert "lower is better" in summary

    def test_an_empty_project_says_so(self) -> None:
        summary = build_measured_summary([], {}, 0.80, "f1", MetricDirection.MAXIMIZE)
        assert "No experiment has completed a full benchmark yet." in summary

    def test_a_measured_experiment_reports_its_value_and_delta(self) -> None:
        experiments = [_experiment("exp-001", "Raise the knob", ExperimentStatus.PROMISING)]
        summary = build_measured_summary(
            experiments, {"exp-001": 0.88}, 0.80, "f1", MetricDirection.MAXIMIZE
        )
        assert "**exp-001** f1 = 0.8800 (+10.0% vs baseline): Raise the knob" in summary

    def test_an_observation_is_quoted_under_its_experiment(self) -> None:
        experiment = _experiment("exp-001", "Raise the knob", ExperimentStatus.PROMISING)
        summary = build_measured_summary(
            [experiment.model_copy(update={"observation": "Loss plateaued at epoch 3."})],
            {"exp-001": 0.88},
            0.80,
            "f1",
            MetricDirection.MAXIMIZE,
        )
        assert "  > Loss plateaued at epoch 3." in summary

    def test_an_unmeasured_experiment_still_carries_its_observation(self) -> None:
        experiment = _experiment(
            "exp-002", "Broken knob", ExperimentStatus.FAILED_EXECUTION, "crashed"
        )
        summary = build_measured_summary(
            [experiment.model_copy(update={"observation": "The run raised a KeyError."})],
            {},
            0.80,
            "f1",
            MetricDirection.MAXIMIZE,
        )
        assert "Tried without a usable measurement" in summary
        assert "  > The run raised a KeyError." in summary

    def test_the_best_result_comes_first(self) -> None:
        experiments = [
            _experiment("exp-001", "Small win", ExperimentStatus.PROMISING),
            _experiment("exp-002", "Big win", ExperimentStatus.PROMISING),
        ]
        summary = build_measured_summary(
            experiments,
            {"exp-001": 0.82, "exp-002": 0.90},
            0.80,
            "f1",
            MetricDirection.MAXIMIZE,
        )
        assert summary.index("exp-002") < summary.index("exp-001")

    def test_a_minimized_metric_ranks_the_lowest_first(self) -> None:
        experiments = [
            _experiment("exp-001", "Slower", ExperimentStatus.PROMISING),
            _experiment("exp-002", "Faster", ExperimentStatus.PROMISING),
        ]
        summary = build_measured_summary(
            experiments,
            {"exp-001": 150.0, "exp-002": 90.0},
            120.0,
            "latency_ms",
            MetricDirection.MINIMIZE,
        )
        assert summary.index("exp-002") < summary.index("exp-001")

    def test_an_unmeasured_experiment_is_listed_with_its_reason(self) -> None:
        experiments = [
            _experiment(
                "exp-002",
                "Latency violator",
                ExperimentStatus.REJECTED,
                reason="p95_latency_ms 250 > 200",
            )
        ]
        summary = build_measured_summary([*experiments], {}, 0.80, "f1", MetricDirection.MAXIMIZE)
        assert "Tried without a usable measurement" in summary
        assert "p95_latency_ms 250 > 200" in summary


class TestResultsInstructions:
    def test_asks_for_something_new(self) -> None:
        lines = results_instructions("everything measured")
        assert any("materially different" in line for line in lines)

    def test_carries_the_context(self) -> None:
        assert "everything measured" in results_instructions("everything measured")

    def test_a_huge_context_is_truncated(self) -> None:
        lines = results_instructions("z" * 20_000)
        assert max(len(line) for line in lines) == 6000


class TestResultsContext:
    def test_a_measured_project_reports_its_outcomes(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        context = build_results_context()

        assert "exp-001" in context
        assert "Baseline **f1**" in context

    def test_a_rejection_is_included_as_a_negative_result(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        context = build_results_context()

        assert "exp-002" in context

    def test_scoping_to_a_run_is_accepted(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        context = build_results_context("run-001")

        assert "exp-001" in context

    def test_an_unknown_run_is_refused(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="Unknown run"):
            build_results_context("run-404")

    def test_a_project_without_experiments_has_no_context(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        assert build_results_context() == ""


class TestFromResultsCli:
    def test_without_results_it_refuses_and_says_why(
        self, cli_runner: CliRunner, project_with_hypotheses: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        result = cli_runner.invoke(app, ["research", "synthesize", "--from-results"])

        assert result.exit_code == 1
        assert "No results to synthesize from yet" in result.output

    def test_run_without_from_results_is_refused(
        self, cli_runner: CliRunner, project_with_hypotheses: Path
    ) -> None:
        result = cli_runner.invoke(
            app, ["research", "synthesize", "--run", "run-001"]
        )

        assert result.exit_code == 1
        assert "--run only applies with --from-results" in result.output
