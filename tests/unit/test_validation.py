"""Validation aggregation (pure) + real-venv Stage 3 end to end."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.domain.baseline import BaselineRun, BaselineStatus, EnvironmentFingerprint
from researchforge.domain.contract import ConstraintOperator, MetricDirection
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
from researchforge.execution.validation import summarize_validation


def _knob_patch(improvement: int, latency: float) -> str:
    return f"""\
diff --git a/src/algo.py b/src/algo.py
new file mode 100644
--- /dev/null
+++ b/src/algo.py
@@ -0,0 +1,2 @@
+IMPROVEMENT = {improvement}
+LATENCY = {latency}
"""


def _stage_plan(base: Path, entries: list[tuple[str, str]]) -> Path:
    staging = base / ".researchforge" / "experiments"
    patches = staging / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    lines = [
        "hypothesis_id: hyp-001",
        "approach_summary: Knob variants.",
        "experiments:",
    ]
    for key, patch_text in entries:
        (patches / f"{key}.patch").write_text(patch_text, encoding="utf-8")
        lines += [
            f"  - key: {key}",
            f"    title: Variant {key}",
            f"    change_summary: Set knobs for {key}.",
            f"    patch_file: patches/{key}.patch",
        ]
    plan = staging / "plan.yaml"
    plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def _metrics(value: float) -> MetricResult:
    return MetricResult.model_validate(
        {"schema_version": 1, "primary_metric": {"name": "f1", "value": value}}
    )


def _baseline(value: float = 0.80) -> BaselineRun:
    now = datetime.now(UTC)
    return BaselineRun(
        baseline_id="b1",
        contract_id="c1",
        contract_version=1,
        commit_sha="a" * 40,
        execution_mode=ExecutionEngine.VENV,
        command="eval",
        status=BaselineStatus.SUCCEEDED,
        metrics=_metrics(value),
        fingerprint=EnvironmentFingerprint(
            platform="test",
            execution_mode=ExecutionEngine.VENV,
            contract_id="c1",
            contract_version=1,
            commit_sha="a" * 40,
        ),
        stdout_path="s",
        stderr_path="e",
        started_at=now,
        completed_at=now,
        duration_seconds=1.0,
    )


def _experiment() -> Experiment:
    now = datetime.now(UTC)
    return Experiment(
        experiment_id="exp-001",
        plan_id="plan-001",
        hypothesis_id="hyp-001",
        title="t",
        change_summary="c",
        patch_text="p",
        patch_sha256="0" * 64,
        status=ExperimentStatus.VALIDATING,
        created_at=now,
        updated_at=now,
    )


def _attempt(
    value: float | None,
    attempt: int,
    status: ExecutionRecordStatus = ExecutionRecordStatus.SUCCEEDED,
    constraint_passed: bool | None = True,
) -> ExperimentExecution:
    now = datetime.now(UTC)
    constraints = []
    if constraint_passed is not None:
        constraints = [
            ConstraintResult(
                name="p95_latency_ms",
                operator=ConstraintOperator.LE,
                threshold=200.0,
                observed=150.0 if constraint_passed else 250.0,
                passed=constraint_passed,
            )
        ]
    return ExperimentExecution(
        execution_id=uuid4().hex,
        experiment_id="exp-001",
        run_id="run-001",
        hypothesis_id="hyp-001",
        baseline_commit="a" * 40,
        execution_mode=ExecutionEngine.VENV,
        benchmark_stage=BenchmarkStage.VALIDATION,
        attempt=attempt,
        change_summary="c",
        started_at=now,
        completed_at=now,
        status=status,
        metrics=_metrics(value) if value is not None else None,
        constraints=constraints,
        artifacts=ExecutionArtifacts(diff_path="d", stdout_path="o", stderr_path="e"),
        fingerprint=EnvironmentFingerprint(
            platform="test",
            execution_mode=ExecutionEngine.VENV,
            contract_id="c1",
            contract_version=1,
            commit_sha="a" * 40,
        ),
    )


class TestSummarizeValidation:
    def test_all_confirm_becomes_validated(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.85, 1), _attempt(0.851, 2)],
            _baseline(),
            MetricDirection.MAXIMIZE,
        )
        assert summary.outcome is ExperimentStatus.VALIDATED
        assert summary.mean is not None and abs(summary.mean - 0.8505) < 1e-9
        assert summary.stdev is not None and summary.stdev > 0
        assert summary.improvement_confirmed_in_all

    def test_aggregation_math(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.84, 1), _attempt(0.86, 2)],
            _baseline(),
            MetricDirection.MAXIMIZE,
        )
        assert summary.mean == 0.85
        # sample stdev (n-1): sqrt(((0.01)^2 + (0.01)^2)/1)
        assert summary.stdev is not None and abs(summary.stdev - 0.0141421356) < 1e-6
        assert summary.min_value == 0.84
        assert summary.max_value == 0.86
        assert summary.coefficient_of_variation is not None

    def test_single_attempt_has_no_stdev(self) -> None:
        summary = summarize_validation(
            _experiment(), [_attempt(0.85, 1)], _baseline(), MetricDirection.MAXIMIZE
        )
        assert summary.stdev is None
        assert summary.outcome is ExperimentStatus.VALIDATED  # + the Stage-2 run = 2 measurements

    def test_no_spread_limit_means_no_spread_check(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.84, 1), _attempt(0.96, 2)],  # wildly noisy, but nothing asked
            _baseline(),
            MetricDirection.MAXIMIZE,
        )
        assert summary.stdev_max is None
        assert not summary.stdev_exceeded
        assert summary.outcome is ExperimentStatus.VALIDATED

    def test_spread_above_the_limit_is_rejected_despite_improving(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.84, 1), _attempt(0.96, 2)],  # stdev ~0.0849, both beat 0.80
            _baseline(),
            MetricDirection.MAXIMIZE,
            stdev_max=0.01,
        )
        assert summary.improvement_confirmed_in_all
        assert summary.all_constraints_passed
        assert summary.stdev_exceeded
        assert summary.outcome is ExperimentStatus.REJECTED

    def test_spread_within_the_limit_still_validates(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.850, 1), _attempt(0.851, 2)],
            _baseline(),
            MetricDirection.MAXIMIZE,
            stdev_max=0.01,
        )
        assert summary.stdev is not None and summary.stdev < 0.01
        assert not summary.stdev_exceeded
        assert summary.outcome is ExperimentStatus.VALIDATED

    def test_the_limit_is_recorded_so_a_reader_knows_what_was_applied(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.85, 1), _attempt(0.851, 2)],
            _baseline(),
            MetricDirection.MAXIMIZE,
            stdev_max=0.25,
        )
        assert summary.stdev_max == 0.25

    def test_one_attempt_cannot_exceed_a_spread_limit(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.85, 1)],
            _baseline(),
            MetricDirection.MAXIMIZE,
            stdev_max=0.0001,
        )
        assert summary.stdev is None
        assert not summary.stdev_exceeded
        assert summary.outcome is ExperimentStatus.VALIDATED

    def test_constraint_violation_in_any_attempt_rejects(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.85, 1), _attempt(0.86, 2, constraint_passed=False)],
            _baseline(),
            MetricDirection.MAXIMIZE,
        )
        assert summary.outcome is ExperimentStatus.REJECTED

    def test_failed_attempt_rejects(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [
                _attempt(0.85, 1),
                _attempt(None, 2, status=ExecutionRecordStatus.FAILED_EXECUTION),
            ],
            _baseline(),
            MetricDirection.MAXIMIZE,
        )
        assert summary.outcome is ExperimentStatus.REJECTED

    def test_unconfirmed_improvement_returns_to_promising(self) -> None:
        summary = summarize_validation(
            _experiment(),
            [_attempt(0.85, 1), _attempt(0.79, 2)],  # second run below baseline
            _baseline(),
            MetricDirection.MAXIMIZE,
        )
        assert summary.outcome is ExperimentStatus.PROMISING
        assert not summary.improvement_confirmed_in_all


class TestValidateEndToEnd:
    """Real venv Stage 3 on the funnel fixture (reuses PR3's batch helpers)."""

    def test_validate_confirms_winner(
        self,
        cli_runner: CliRunner,
        funnel_project: Path,
        isolated_project_dir: Path,
    ) -> None:
        plan = _stage_plan(isolated_project_dir, [("improve", _knob_patch(5, 150.0))])
        assert cli_runner.invoke(app, ["experiment", "import", str(plan)]).exit_code == 0
        assert cli_runner.invoke(app, ["experiment", "approve", "plan-001", "--yes"]).exit_code == 0
        assert cli_runner.invoke(app, ["experiment", "run", "plan-001"]).exit_code == 0

        result = cli_runner.invoke(app, ["validate", "run-001", "--yes", "--json"])

        assert result.exit_code == 0, result.output
        summaries = json.loads(result.output)
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary["outcome"] == "validated"
        assert summary["attempts"] == 2  # repeat_finalists: 2
        assert summary["values"] == [0.85, 0.85]
        assert summary["improvement_confirmed_in_all"] is True

        # Status + results reflect validation.
        status = json.loads(cli_runner.invoke(app, ["status", "--json"]).output)
        assert status["status"] == "validated"
        assert "ship branch" in status["next_action"]

        shown = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-001", "--json"]).output
        )
        assert shown["status"] == "validated"
        assert "validated across 2 repeated runs" in shown["decision"]["reason"]

        results = cli_runner.invoke(app, ["results", "show", "run-001"])
        assert results.exit_code == 0
        assert "[validated]" in results.output
        assert "Winner: exp-001" in results.output

        # Validation artifacts persisted, one fresh env per attempt.
        validation_dir = (
            funnel_project / ".researchforge" / "artifacts" / "experiments" / "run-001" / "exp-001"
        )
        assert (validation_dir / "validation-a1" / "manifest.json").is_file()
        assert (validation_dir / "validation-a2" / "manifest.json").is_file()
        summary_file = (
            funnel_project
            / ".researchforge"
            / "artifacts"
            / "experiments"
            / "run-001"
            / "validation_summary.json"
        )
        assert summary_file.is_file()

    def test_n_overrides_the_contracts_repeat_count(
        self,
        cli_runner: CliRunner,
        funnel_project: Path,
        isolated_project_dir: Path,
    ) -> None:
        plan = _stage_plan(isolated_project_dir, [("improve", _knob_patch(5, 150.0))])
        assert cli_runner.invoke(app, ["experiment", "import", str(plan)]).exit_code == 0
        assert cli_runner.invoke(app, ["experiment", "approve", "plan-001", "--yes"]).exit_code == 0
        assert cli_runner.invoke(app, ["experiment", "run", "plan-001"]).exit_code == 0

        result = cli_runner.invoke(app, ["validate", "run-001", "--n", "3", "--yes", "--json"])
        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)[0]
        assert summary["attempts"] == 3, "the contract says 2; --n asked for 3"
        assert summary["outcome"] == "validated"

    def test_n_does_not_change_the_stored_contract(
        self,
        cli_runner: CliRunner,
        funnel_project: Path,
        isolated_project_dir: Path,
    ) -> None:
        from contextlib import closing

        from researchforge.storage.contract_repository import get_active_contract
        from researchforge.storage.db import open_project_db

        plan = _stage_plan(isolated_project_dir, [("improve", _knob_patch(5, 150.0))])
        assert cli_runner.invoke(app, ["experiment", "import", str(plan)]).exit_code == 0
        assert cli_runner.invoke(app, ["experiment", "approve", "plan-001", "--yes"]).exit_code == 0
        assert cli_runner.invoke(app, ["experiment", "run", "plan-001"]).exit_code == 0
        assert cli_runner.invoke(
            app, ["validate", "run-001", "--n", "3", "--yes", "--json"]
        ).exit_code == 0

        with closing(open_project_db()) as conn:
            contract = get_active_contract(conn)
        assert contract is not None
        assert contract.spec.validation.repeat_finalists == 2

    def test_stdev_max_rejects_a_result_it_cannot_resolve(
        self,
        cli_runner: CliRunner,
        funnel_project: Path,
        isolated_project_dir: Path,
    ) -> None:
        plan = _stage_plan(isolated_project_dir, [("improve", _knob_patch(5, 150.0))])
        assert cli_runner.invoke(app, ["experiment", "import", str(plan)]).exit_code == 0
        assert cli_runner.invoke(app, ["experiment", "approve", "plan-001", "--yes"]).exit_code == 0
        assert cli_runner.invoke(app, ["experiment", "run", "plan-001"]).exit_code == 0

        # The fixture evaluator is deterministic, so its repeats have stdev 0 and
        # no limit can reject them. A negative-width limit is unreachable, so
        # this asserts the flag is accepted and recorded rather than ignored.
        result = cli_runner.invoke(
            app, ["validate", "run-001", "--stdev-max", "0.5", "--yes", "--json"]
        )
        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)[0]
        assert summary["stdev_max"] == 0.5
        assert summary["outcome"] == "validated"

    def test_validate_without_promising_is_blocked(
        self,
        cli_runner: CliRunner,
        funnel_project: Path,
        isolated_project_dir: Path,
    ) -> None:
        # Only a constraint-violating variant -> nothing promising.
        plan = _stage_plan(isolated_project_dir, [("hot", _knob_patch(6, 250.0))])
        assert cli_runner.invoke(app, ["experiment", "import", str(plan)]).exit_code == 0
        cli_runner.invoke(app, ["experiment", "approve", "plan-001", "--yes"])
        cli_runner.invoke(app, ["experiment", "run", "plan-001"])

        result = cli_runner.invoke(app, ["validate", "run-001", "--yes"])

        assert result.exit_code == 1
        assert "no promising experiments" in result.output
