"""The research log: what the AI reads before re-synthesizing."""

from datetime import UTC, datetime
from pathlib import Path

from researchforge.domain.baseline import (
    BaselineRun,
    BaselineStatus,
    EnvironmentFingerprint,
)
from researchforge.domain.contract import MetricDirection
from researchforge.domain.environment import ExecutionEngine
from researchforge.domain.experiment import Experiment, ExperimentStatus
from researchforge.execution.metrics import MetricResult, MetricValue
from researchforge.research.research_log import (
    beats_baseline,
    build_initial_log,
    build_resynth_context,
    improvement_pct,
    render_best_section,
    update_log_after_round,
)

MAXIMIZE = MetricDirection.MAXIMIZE
MINIMIZE = MetricDirection.MINIMIZE


def _baseline(value: float, name: str = "f1") -> BaselineRun:
    now = datetime.now(UTC)
    return BaselineRun(
        baseline_id="base-001",
        contract_id="contract-001",
        contract_version=1,
        commit_sha="a" * 40,
        execution_mode=ExecutionEngine.VENV,
        command="python benchmarks/evaluate.py",
        status=BaselineStatus.SUCCEEDED,
        metrics=MetricResult(
            schema_version=1,
            primary_metric=MetricValue(name=name, value=value),
        ),
        fingerprint=EnvironmentFingerprint(
            platform="test",
            execution_mode=ExecutionEngine.VENV,
            contract_id="contract-001",
            contract_version=1,
            commit_sha="a" * 40,
        ),
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        started_at=now,
        completed_at=now,
        duration_seconds=1.0,
    )


def _experiment(experiment_id: str, summary: str, status: ExperimentStatus) -> Experiment:
    now = datetime.now(UTC)
    return Experiment(
        experiment_id=experiment_id,
        plan_id="plan-001",
        hypothesis_id="hyp-001",
        title=f"Variant {experiment_id}",
        change_summary=summary,
        patch_text="",
        patch_sha256="0" * 64,
        status=status,
        created_at=now,
        updated_at=now,
    )


class TestImprovementMath:
    def test_maximize_improvement_is_positive_when_value_rises(self) -> None:
        assert improvement_pct(0.77, 0.70, MAXIMIZE) > 0

    def test_maximize_improvement_is_negative_when_value_falls(self) -> None:
        assert improvement_pct(0.63, 0.70, MAXIMIZE) < 0

    def test_minimize_improvement_is_positive_when_value_falls(self) -> None:
        assert improvement_pct(9.0, 10.0, MINIMIZE) > 0

    def test_minimize_improvement_is_negative_when_value_rises(self) -> None:
        assert improvement_pct(11.0, 10.0, MINIMIZE) < 0

    def test_zero_baseline_reports_no_percentage_rather_than_dividing(self) -> None:
        assert improvement_pct(1.0, 0.0, MAXIMIZE) == 0.0

    def test_lower_beats_baseline_only_when_minimizing(self) -> None:
        assert beats_baseline(9.0, 10.0, MINIMIZE) is True
        assert beats_baseline(9.0, 10.0, MAXIMIZE) is False


class TestInitialLog:
    def test_objective_and_baseline_are_recorded(self) -> None:
        text = build_initial_log("Improve F1 without hurting latency", _baseline(0.7125))
        assert "Improve F1 without hurting latency" in text
        assert "0.7125" in text
        assert "## Current Best" in text
        assert "_No experiments run yet._" in text


class TestBestSection:
    def test_baseline_is_the_best_when_nothing_has_won(self) -> None:
        section = render_best_section(None, 0.70, 0.70, "f1", MAXIMIZE)
        assert "**Baseline**" in section
        assert "nothing has beaten it yet" in section

    def test_winning_experiment_is_named_with_its_delta(self) -> None:
        experiment = _experiment("exp-004", "CONF + LR schedule", ExperimentStatus.PROMISING)
        section = render_best_section(experiment, 0.77, 0.70, "f1", MAXIMIZE)
        assert "exp-004" in section
        assert "+10.0%" in section
        assert "CONF + LR schedule" in section

    def test_lower_value_wins_for_a_minimize_metric(self) -> None:
        experiment = _experiment("exp-002", "Smaller model", ExperimentStatus.PROMISING)
        section = render_best_section(experiment, 9.0, 10.0, "inference_ms", MINIMIZE)
        assert "exp-002" in section
        assert "+10.0%" in section

    def test_worse_value_does_not_become_the_best_for_a_minimize_metric(self) -> None:
        experiment = _experiment("exp-003", "Bigger model", ExperimentStatus.PROMISING)
        section = render_best_section(experiment, 12.0, 10.0, "inference_ms", MINIMIZE)
        assert "**Baseline**" in section


class TestRoundUpdate:
    def test_round_outcomes_are_appended(self, tmp_path: Path) -> None:
        log = tmp_path / "research-log.md"
        log.write_text(build_initial_log("Improve F1", _baseline(0.70)), encoding="utf-8")

        update_log_after_round(
            log,
            1,
            [
                _experiment("exp-001", "CONF=0.001", ExperimentStatus.PROMISING),
                _experiment("exp-002", "yolov5mu", ExperimentStatus.REJECTED),
                _experiment("exp-003", "broken patch", ExperimentStatus.FAILED_SETUP),
            ],
            0.70,
            "f1",
            _experiment("exp-001", "CONF=0.001", ExperimentStatus.PROMISING),
            0.74,
            MAXIMIZE,
        )

        text = log.read_text(encoding="utf-8")
        assert "## Round 1" in text
        assert "[✓] **exp-001**: CONF=0.001" in text
        assert "[✗] **exp-002**: yolov5mu" in text
        assert "[⚠] **exp-003**: broken patch" in text

    def test_current_best_section_is_replaced_not_duplicated(self, tmp_path: Path) -> None:
        log = tmp_path / "research-log.md"
        log.write_text(build_initial_log("Improve F1", _baseline(0.70)), encoding="utf-8")
        winner = _experiment("exp-001", "CONF=0.001", ExperimentStatus.PROMISING)

        update_log_after_round(log, 1, [winner], 0.70, "f1", winner, 0.74, MAXIMIZE)
        text = log.read_text(encoding="utf-8")

        assert text.count("## Current Best") == 1
        assert "**exp-001**" in text.split("## Round 1")[0]

    def test_minimize_metric_reports_the_round_as_an_improvement(self, tmp_path: Path) -> None:
        log = tmp_path / "research-log.md"
        log.write_text(
            build_initial_log("Cut latency", _baseline(10.0, "inference_ms")), encoding="utf-8"
        )
        winner = _experiment("exp-001", "int8 quantization", ExperimentStatus.PROMISING)

        update_log_after_round(log, 1, [winner], 10.0, "inference_ms", winner, 8.0, MINIMIZE)

        text = log.read_text(encoding="utf-8")
        assert "Current best: exp-001" in text
        assert "+20.0% vs baseline" in text

    def test_empty_round_is_recorded_honestly(self, tmp_path: Path) -> None:
        log = tmp_path / "research-log.md"
        log.write_text(build_initial_log("Improve F1", _baseline(0.70)), encoding="utf-8")

        update_log_after_round(log, 2, [], 0.70, "f1", None, 0.70, MAXIMIZE)

        text = log.read_text(encoding="utf-8")
        assert "_No experiment completed this round._" in text

    def test_observations_are_attached_to_their_experiment(self, tmp_path: Path) -> None:
        log = tmp_path / "research-log.md"
        log.write_text(build_initial_log("Improve F1", _baseline(0.70)), encoding="utf-8")
        winner = _experiment("exp-001", "CONF=0.001", ExperimentStatus.PROMISING)

        update_log_after_round(
            log,
            1,
            [winner],
            0.70,
            "f1",
            winner,
            0.74,
            MAXIMIZE,
            {"exp-001": "Loss plateaued at epoch 3 — warmup too short."},
        )

        assert "> Loss plateaued at epoch 3" in log.read_text(encoding="utf-8")


class TestResynthContext:
    def test_missing_log_yields_empty_context(self, tmp_path: Path) -> None:
        assert build_resynth_context(tmp_path / "absent.md") == ""

    def test_existing_log_is_returned_verbatim(self, tmp_path: Path) -> None:
        log = tmp_path / "research-log.md"
        log.write_text("# Log\ncontent", encoding="utf-8")
        assert build_resynth_context(log) == "# Log\ncontent"
