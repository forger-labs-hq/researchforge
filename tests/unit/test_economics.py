"""Time economics: what was spent, and what the loop declined to spend.

The avoided-work arithmetic is the easy place to ship a plausible wrong number,
so most of what follows pins down the cases where the honest answer is "no
figure" rather than a confident zero.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from researchforge.domain.baseline import BaselineRun, EnvironmentFingerprint
from researchforge.domain.contract import ConstraintOperator
from researchforge.domain.environment import ExecutionEngine
from researchforge.domain.experiment import (
    BenchmarkStage,
    ConstraintResult,
    ExecutionArtifacts,
    ExecutionRecordStatus,
    Experiment,
    ExperimentExecution,
    ExperimentStatus,
)
from researchforge.execution.metrics import MetricResult
from researchforge.reporting.economics import (
    Avoided,
    _avoided,
    _caught,
    _record,
    _stage_time,
    _time_by_outcome,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)

FINGERPRINT = EnvironmentFingerprint(
    platform="darwin",
    execution_mode=ExecutionEngine.VENV,
    python_version="3.12.0",
    contract_id="contract-001",
    contract_version=1,
    commit_sha="abc123",
)


def _execution(
    experiment_id: str,
    stage: BenchmarkStage,
    seconds: float,
    *,
    value: float | None = None,
    violated: bool = False,
) -> ExperimentExecution:
    return ExperimentExecution(
        execution_id=f"{experiment_id}-{stage.value}",
        experiment_id=experiment_id,
        run_id="run-001",
        hypothesis_id="hyp-001",
        baseline_commit="abc123",
        execution_mode=ExecutionEngine.VENV,
        benchmark_stage=stage,
        attempt=1,
        change_summary="a change",
        started_at=NOW,
        completed_at=NOW,
        status=ExecutionRecordStatus.SUCCEEDED,
        metrics=(
            None
            if value is None
            else MetricResult(
                schema_version=1,
                primary_metric={"name": "f1", "value": value},
            )
        ),
        constraints=(
            [
                ConstraintResult(
                    name="p95_latency_ms",
                    operator=ConstraintOperator.LE,
                    threshold=200.0,
                    observed=291.0,
                    passed=False,
                )
            ]
            if violated
            else []
        ),
        artifacts=ExecutionArtifacts(diff_path="d", stdout_path="o", stderr_path="e"),
        fingerprint=FINGERPRINT,
        duration_seconds=seconds,
    )


def _experiment(
    experiment_id: str,
    status: ExperimentStatus,
    parents: list[str] | None = None,
) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        plan_id="plan-001",
        hypothesis_id="hyp-001",
        parent_experiment_ids=parents or [],
        title="A variant",
        change_summary="a change",
        patch_text="",
        patch_sha256="0" * 64,
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


class TestTimeUsed:
    def test_each_stage_is_totalled_separately(self) -> None:
        stages = _stage_time(
            [
                _execution("exp-001", BenchmarkStage.SCREENING, 30.0),
                _execution("exp-001", BenchmarkStage.FULL, 600.0),
                _execution("exp-001", BenchmarkStage.VALIDATION, 1800.0),
            ],
            [],
        )

        assert (stages.screening, stages.full, stages.validation) == (30.0, 600.0, 1800.0)

    def test_the_baseline_counts_as_compute_too(self) -> None:
        baseline = BaselineRun(
            baseline_id="base-001",
            contract_id="contract-001",
            contract_version=1,
            commit_sha="abc123",
            execution_mode=ExecutionEngine.VENV,
            command="pytest",
            status="succeeded",
            fingerprint=FINGERPRINT,
            stdout_path="o",
            stderr_path="e",
            started_at=NOW,
            completed_at=NOW,
            duration_seconds=283.0,
        )

        assert _stage_time([], [baseline]).total == 283.0

    def test_time_is_grouped_by_how_the_experiment_ended(self) -> None:
        """What the losers cost is invisible in a ranking of the winners."""
        experiments = [
            _experiment("exp-001", ExperimentStatus.VALIDATED),
            _experiment("exp-002", ExperimentStatus.REJECTED),
            _experiment("exp-003", ExperimentStatus.FAILED_EXECUTION),
        ]
        executions = [
            _execution("exp-001", BenchmarkStage.FULL, 100.0),
            _execution("exp-002", BenchmarkStage.FULL, 200.0),
            _execution("exp-003", BenchmarkStage.FULL, 5.0),
        ]

        assert _time_by_outcome(experiments, executions) == {
            "failed": 5.0,
            "kept": 100.0,
            "rejected": 200.0,
        }


class TestWorkAvoided:
    def test_screening_that_stopped_a_full_run_is_counted(self) -> None:
        avoided = _avoided(
            [_experiment("exp-001", ExperimentStatus.REJECTED)],
            [
                _execution("exp-001", BenchmarkStage.SCREENING, 30.0),
                _execution("exp-002", BenchmarkStage.FULL, 600.0),
            ],
        )

        assert avoided.screened_out == 1
        assert avoided.seconds == pytest.approx(600.0)

    def test_screening_that_led_to_a_full_run_avoided_nothing(self) -> None:
        avoided = _avoided(
            [_experiment("exp-001", ExperimentStatus.REJECTED)],
            [
                _execution("exp-001", BenchmarkStage.SCREENING, 30.0),
                _execution("exp-001", BenchmarkStage.FULL, 600.0),
            ],
        )

        assert avoided.screened_out == 0

    def test_the_stall_rule_gets_its_own_count(self) -> None:
        avoided = _avoided(
            [_experiment("exp-002", ExperimentStatus.CANCELLED)],
            [_execution("exp-001", BenchmarkStage.FULL, 600.0)],
        )

        assert avoided.cancelled == 1
        assert avoided.runs == 1

    def test_a_cancelled_experiment_is_not_also_counted_as_screened_out(self) -> None:
        """One skipped run, one reason — otherwise the total double-counts."""
        avoided = _avoided(
            [_experiment("exp-002", ExperimentStatus.CANCELLED)],
            [
                _execution("exp-001", BenchmarkStage.FULL, 600.0),
                _execution("exp-002", BenchmarkStage.SCREENING, 30.0),
            ],
        )

        assert (avoided.screened_out, avoided.cancelled) == (0, 1)
        assert avoided.runs == 1

    def test_without_a_full_run_to_average_there_is_no_figure(self) -> None:
        """No measured cost means no claim — not a claim of zero."""
        avoided = _avoided(
            [_experiment("exp-001", ExperimentStatus.CANCELLED)],
            [_execution("exp-001", BenchmarkStage.SCREENING, 30.0)],
        )

        assert avoided.mean_full_seconds is None
        assert avoided.seconds is None

    def test_avoiding_nothing_is_zero_rather_than_unknown(self) -> None:
        assert Avoided(mean_full_seconds=600.0).seconds is None
        assert Avoided(screened_out=0, cancelled=0, mean_full_seconds=None).runs == 0

    def test_the_average_comes_from_this_project(self) -> None:
        avoided = _avoided(
            [_experiment("exp-003", ExperimentStatus.CANCELLED)],
            [
                _execution("exp-001", BenchmarkStage.FULL, 400.0),
                _execution("exp-002", BenchmarkStage.FULL, 600.0),
            ],
        )

        assert avoided.mean_full_seconds == pytest.approx(500.0)
        assert avoided.seconds == pytest.approx(500.0)


class TestTheRecord:
    def test_failures_are_preserved_rather_than_dropped(self) -> None:
        record = _record(
            [
                _experiment("exp-001", ExperimentStatus.VALIDATED),
                _experiment("exp-002", ExperimentStatus.REJECTED),
                _experiment("exp-003", ExperimentStatus.FAILED_EXECUTION),
                _experiment("exp-004", ExperimentStatus.CANCELLED),
            ]
        )

        assert record.experiments == 4
        assert (record.kept, record.rejected, record.failed, record.cancelled) == (1, 1, 1, 1)

    def test_lineage_is_counted(self) -> None:
        record = _record(
            [
                _experiment("exp-001", ExperimentStatus.VALIDATED),
                _experiment("exp-002", ExperimentStatus.REJECTED, parents=["exp-001"]),
            ]
        )

        assert record.with_lineage == 1


class TestMistakesCaught:
    def test_a_broken_hard_limit_is_reported(self) -> None:
        caught = _caught(
            [_experiment("exp-001", ExperimentStatus.REJECTED)],
            [_execution("exp-001", BenchmarkStage.FULL, 600.0, value=0.77, violated=True)],
            baseline_value=0.74,
        )

        assert caught.constraint_violations == ["exp-001"]

    def test_a_limit_broken_at_screening_still_counts(self) -> None:
        """Caught early is the best case, and has no full-benchmark value to key on."""
        caught = _caught(
            [_experiment("exp-001", ExperimentStatus.REJECTED)],
            [_execution("exp-001", BenchmarkStage.SCREENING, 30.0, value=0.77, violated=True)],
            baseline_value=0.74,
        )

        assert caught.constraint_violations == ["exp-001"]

    def test_a_change_that_moved_nothing_is_reported_as_a_no_op(self) -> None:
        """It inherits its parent's gain while contributing none of its own."""
        caught = _caught(
            [
                _experiment("exp-001", ExperimentStatus.VALIDATED),
                _experiment("exp-002", ExperimentStatus.PROMISING, parents=["exp-001"]),
            ],
            [
                _execution("exp-001", BenchmarkStage.FULL, 600.0, value=0.838),
                _execution("exp-002", BenchmarkStage.FULL, 600.0, value=0.838),
            ],
            baseline_value=0.829,
        )

        assert caught.no_ops == ["exp-002"]

    def test_a_child_that_moved_the_metric_is_not_a_no_op(self) -> None:
        caught = _caught(
            [
                _experiment("exp-001", ExperimentStatus.VALIDATED),
                _experiment("exp-002", ExperimentStatus.PROMISING, parents=["exp-001"]),
            ],
            [
                _execution("exp-001", BenchmarkStage.FULL, 600.0, value=0.838),
                _execution("exp-002", BenchmarkStage.FULL, 600.0, value=0.841),
            ],
            baseline_value=0.829,
        )

        assert caught.no_ops == []

    def test_a_root_that_matched_the_baseline_moved_nothing(self) -> None:
        caught = _caught(
            [_experiment("exp-001", ExperimentStatus.REJECTED)],
            [_execution("exp-001", BenchmarkStage.FULL, 600.0, value=0.829)],
            baseline_value=0.829,
        )

        assert caught.no_ops == ["exp-001"]

    def test_an_unmeasured_experiment_is_neither(self) -> None:
        caught = _caught(
            [_experiment("exp-001", ExperimentStatus.FAILED_EXECUTION)],
            [_execution("exp-001", BenchmarkStage.SCREENING, 5.0)],
            baseline_value=0.829,
        )

        assert caught.no_ops == []
        assert caught.constraint_violations == []

    def test_an_execution_for_an_unknown_experiment_is_ignored(self) -> None:
        """Stale rows must not invent a catch that has nothing to point at."""
        caught = _caught(
            [],
            [_execution("exp-404", BenchmarkStage.FULL, 600.0, value=0.77, violated=True)],
            baseline_value=0.74,
        )

        assert caught.constraint_violations == []
