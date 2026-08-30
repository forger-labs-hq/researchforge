"""Stage 3: repeated validation of promising finalists.

Each attempt is a fresh worktree + environment rebuild of the full benchmark.
An experiment becomes `validated` only when every attempt succeeds, passes
all hard constraints, and confirms the improvement direction — on top of the
Stage-2 run, that is always at least two independent measurements, so a
one-off result can never be validated.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

from researchforge.config.paths import experiment_artifacts_dir
from researchforge.domain.baseline import BaselineRun
from researchforge.domain.contract import MetricDirection
from researchforge.domain.environment import DockerProbe
from researchforge.domain.experiment import (
    BenchmarkStage,
    Decision,
    DecisionOutcome,
    ExecutionRecordStatus,
    Experiment,
    ExperimentExecution,
    ExperimentStatus,
    ValidationSummary,
)
from researchforge.domain.project import ProjectStatus
from researchforge.execution.constraints import constraints_ok
from researchforge.execution.experiments import (
    ExperimentBlockedError,
    RunPreparation,
    execute_stage,
    prepare_run,
)
from researchforge.execution.runner import CommandRunner, SubprocessRunner
from researchforge.project.service import touch_project_status
from researchforge.storage.experiment_repository import (
    get_run,
    list_executions,
    list_experiments,
    update_experiment,
)


def _improved(baseline: float, candidate: float, direction: MetricDirection) -> bool:
    return candidate > baseline if direction is MetricDirection.MAXIMIZE else candidate < baseline


def summarize_validation(
    experiment: Experiment,
    attempts: list[ExperimentExecution],
    baseline: BaselineRun,
    direction: MetricDirection,
    stdev_max: float | None = None,
) -> ValidationSummary:
    """Aggregate validation attempts into a summary and final outcome.

    `stdev_max` refuses to validate a result whose repeats disagree by more than
    that much, however well it improved on average. A wide spread means the
    benchmark cannot resolve the change being claimed, so the average sits
    inside the noise and improving on the baseline was partly the draw.
    """
    assert baseline.metrics is not None
    base_value = baseline.metrics.primary_metric.value

    succeeded = [a for a in attempts if a.status is ExecutionRecordStatus.SUCCEEDED]
    values = [a.metrics.primary_metric.value for a in succeeded if a.metrics is not None]

    mean = sum(values) / len(values) if values else None
    stdev = None
    if len(values) >= 2 and mean is not None:
        stdev = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    cv = stdev / abs(mean) if stdev is not None and mean not in (None, 0) else None

    all_succeeded = len(succeeded) == len(attempts) and bool(attempts)
    all_constraints = all_succeeded and all(constraints_ok(a.constraints) for a in succeeded)
    any_violation = any(not constraints_ok(a.constraints) for a in succeeded)
    improvement_in_all = (
        all_succeeded
        and bool(values)
        and all(_improved(base_value, value, direction) for value in values)
    )

    too_noisy = stdev_max is not None and stdev is not None and stdev > stdev_max

    if too_noisy:
        outcome = ExperimentStatus.REJECTED
    elif all_succeeded and all_constraints and improvement_in_all:
        outcome = ExperimentStatus.VALIDATED
    elif any_violation or not all_succeeded:
        outcome = ExperimentStatus.REJECTED
    else:
        outcome = ExperimentStatus.PROMISING  # succeeded but improvement unconfirmed

    return ValidationSummary(
        experiment_id=experiment.experiment_id,
        attempts=len(attempts),
        succeeded_attempts=len(succeeded),
        values=values,
        mean=mean,
        stdev=stdev,
        coefficient_of_variation=cv,
        min_value=min(values) if values else None,
        max_value=max(values) if values else None,
        all_constraints_passed=all_constraints,
        improvement_confirmed_in_all=improvement_in_all,
        stdev_max=stdev_max,
        outcome=outcome,
    )


@dataclass
class ValidationRun:
    run_id: str
    summaries: list[ValidationSummary]


def validate_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    experiment_ids: list[str] | None = None,
    runner: CommandRunner | None = None,
    docker: DockerProbe | None = None,
    repeats: int | None = None,
    stdev_max: float | None = None,
) -> ValidationRun:
    """Run Stage 3 for the run's promising experiments (or a subset).

    `repeats` overrides the contract's `validation.repeat_finalists` for this
    invocation only; the contract is not modified, so what the project committed
    to stays the record and this stays a choice about one validation.
    """
    run = get_run(conn, run_id)
    if run is None:
        raise ExperimentBlockedError(f"Unknown run: {run_id}.")
    prep: RunPreparation = prepare_run(conn, run.plan_id, docker, allow_completed=True)
    active_runner: CommandRunner = runner or SubprocessRunner()

    experiments = list_experiments(conn, run.plan_id)
    if experiment_ids:
        targets = [e for e in experiments if e.experiment_id in experiment_ids]
        unknown = set(experiment_ids) - {e.experiment_id for e in targets}
        if unknown:
            raise ExperimentBlockedError(
                f"Unknown experiment id(s) for {run_id}: {', '.join(sorted(unknown))}."
            )
        not_promising = [
            e.experiment_id for e in targets if e.status is not ExperimentStatus.PROMISING
        ]
        if not_promising:
            raise ExperimentBlockedError(
                f"Only promising experiments can be validated; not promising: "
                f"{', '.join(not_promising)}."
            )
    else:
        targets = [e for e in experiments if e.status is ExperimentStatus.PROMISING]
    if not targets:
        raise ExperimentBlockedError(
            f"{run_id} has no promising experiments to validate — see "
            f"`researchforge results show {run_id}`."
        )

    attempts_each = repeats or prep.contract.spec.validation.repeat_finalists
    direction = prep.contract.spec.objective.primary_metric.direction
    summaries: list[ValidationSummary] = []

    for experiment in targets:
        experiment = experiment.model_copy(update={"status": ExperimentStatus.VALIDATING})
        update_experiment(conn, experiment)

        previous_validation_attempts = [
            e
            for e in list_executions(conn, run_id=run_id, experiment_id=experiment.experiment_id)
            if e.benchmark_stage is BenchmarkStage.VALIDATION
        ]
        start_attempt = max((e.attempt for e in previous_validation_attempts), default=0) + 1

        attempts: list[ExperimentExecution] = []
        for offset in range(attempts_each):
            attempt = start_attempt + offset
            execution = execute_stage(
                conn,
                prep,
                run,
                experiment,
                BenchmarkStage.VALIDATION,
                attempt,
                active_runner,
                extra_env={"RF_VALIDATION_ATTEMPT": str(attempt)},
            )
            attempts.append(execution)

        summary = summarize_validation(
            experiment, attempts, prep.baseline, direction, stdev_max
        )
        decision_reason = (
            f"validation: {summary.succeeded_attempts}/{summary.attempts} attempts "
            f"succeeded; values={summary.values}; mean={summary.mean}; "
            f"stdev={summary.stdev}"
        )
        if summary.outcome is ExperimentStatus.VALIDATED:
            decision = Decision(
                outcome=DecisionOutcome.KEEP,
                reason=f"validated across {summary.attempts} repeated runs — {decision_reason}",
            )
        elif summary.stdev_exceeded:
            decision = Decision(
                outcome=DecisionOutcome.REJECT,
                reason=(
                    f"spread too wide to validate: stdev {summary.stdev:.4g} exceeds the "
                    f"{stdev_max:.4g} limit — {decision_reason}"
                ),
            )
        elif summary.outcome is ExperimentStatus.REJECTED:
            decision = Decision(
                outcome=DecisionOutcome.REJECT,
                reason=f"validation failed — {decision_reason}",
            )
        else:
            decision = Decision(
                outcome=DecisionOutcome.INVESTIGATE,
                reason=f"improvement not confirmed in every attempt — {decision_reason}",
            )
        experiment = experiment.model_copy(update={"status": summary.outcome, "decision": decision})
        update_experiment(conn, experiment)
        summaries.append(summary)

    if any(s.outcome is ExperimentStatus.VALIDATED for s in summaries):
        touch_project_status(conn, ProjectStatus.VALIDATED)

    summary_path = experiment_artifacts_dir(prep.repo_root) / run_id / "validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import json as json_module

    summary_path.write_text(
        json_module.dumps([s.model_dump(mode="json") for s in summaries], indent=2),
        encoding="utf-8",
    )
    return ValidationRun(run_id=run_id, summaries=summaries)
