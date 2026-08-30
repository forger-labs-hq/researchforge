"""Result ranking and trade-off analysis (pure).

Viability is a status, not one magic score (spec §15.4): rejected and failed
experiments are always shown with their reasons, viable candidates are ranked
by the approved primary objective, and a Pareto frontier is identified when
several candidates are valid. Secondary-metric directions are inferred only
from hard-constraint operators — anything else is displayed but excluded
from dominance math (inferring more would be dishonest).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchforge.domain.baseline import BaselineRun
from researchforge.domain.contract import ConstraintOperator, ContractSpec, MetricDirection
from researchforge.domain.experiment import (
    BenchmarkStage,
    ConstraintResult,
    Decision,
    Experiment,
    ExperimentExecution,
    ExperimentStatus,
    ValidationSummary,
)
from researchforge.experiments.measurements import latest_full_execution

VIABLE_STATUSES = frozenset(
    {
        ExperimentStatus.PROMISING,
        ExperimentStatus.VALIDATING,
        ExperimentStatus.VALIDATED,
        ExperimentStatus.IMPLEMENTATION_READY,
    }
)

_CONFIDENCE_BY_STATUS: dict[ExperimentStatus, str] = {
    ExperimentStatus.PROMISING: "promising",
    ExperimentStatus.VALIDATING: "promising",
    ExperimentStatus.VALIDATED: "validated",
    ExperimentStatus.IMPLEMENTATION_READY: "implementation_ready",
}

ONE_OFF_CAVEAT = (
    "Promising results are one-off controlled runs — never treated as validated. "
    "Run `researchforge validate <run-id>` to confirm finalists."
)


class CandidateRow(BaseModel):
    experiment_id: str
    title: str
    status: ExperimentStatus
    confidence: str  # speculative | promising | validated | implementation_ready
    primary_value: float | None = None
    primary_delta: float | None = None
    primary_delta_pct: float | None = None
    secondary_values: dict[str, float] = Field(default_factory=dict)
    constraints: list[ConstraintResult] = Field(default_factory=list)
    decision: Decision | None = None
    benchmark_stage: BenchmarkStage | None = None
    validation: ValidationSummary | None = None


class RankingReport(BaseModel):
    run_id: str
    baseline_row: CandidateRow
    candidates: list[CandidateRow] = Field(default_factory=list)  # viable, ranked
    rejected: list[CandidateRow] = Field(default_factory=list)  # negative results (§4.5)
    pareto_ids: list[str] = Field(default_factory=list)
    single_winner: str | None = None
    trade_off_notes: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


def signed_improvement(baseline: float, candidate: float, direction: MetricDirection) -> float:
    """Positive = better, regardless of optimization direction."""
    delta = candidate - baseline
    return delta if direction is MetricDirection.MAXIMIZE else -delta


def relative_improvement_pct(
    baseline: float, candidate: float, direction: MetricDirection
) -> float | None:
    if baseline == 0:
        return None
    return signed_improvement(baseline, candidate, direction) / abs(baseline) * 100.0


def inferred_directions(spec: ContractSpec) -> dict[str, MetricDirection]:
    """Directions derivable from the contract: primary + constraint operators."""
    directions = {spec.objective.primary_metric.name: spec.objective.primary_metric.direction}
    for constraint in spec.objective.hard_constraints:
        if constraint.name in directions:
            continue
        if constraint.operator in (ConstraintOperator.LE, ConstraintOperator.LT):
            directions[constraint.name] = MetricDirection.MINIMIZE
        elif constraint.operator in (ConstraintOperator.GE, ConstraintOperator.GT):
            directions[constraint.name] = MetricDirection.MAXIMIZE
        # == constraints imply no direction.
    return directions


def dominates(
    a: dict[str, float], b: dict[str, float], directions: dict[str, MetricDirection]
) -> bool:
    """True when `a` is at least as good on every directed metric and strictly
    better on at least one (metrics missing from either side are skipped)."""
    at_least_as_good = True
    strictly_better = False
    compared = False
    for name, direction in directions.items():
        if name not in a or name not in b:
            continue
        compared = True
        improvement = signed_improvement(b[name], a[name], direction)
        if improvement < 0:
            at_least_as_good = False
            break
        if improvement > 0:
            strictly_better = True
    return compared and at_least_as_good and strictly_better


def _directed_values(
    row: CandidateRow, directions: dict[str, MetricDirection], primary_name: str
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in directions:
        if name == primary_name:
            if row.primary_value is not None:
                values[name] = row.primary_value
        elif name in row.secondary_values:
            values[name] = row.secondary_values[name]
    return values


def pareto_frontier(
    rows: list[CandidateRow],
    directions: dict[str, MetricDirection],
    primary_name: str,
) -> list[str]:
    frontier: list[str] = []
    for row in rows:
        row_values = _directed_values(row, directions, primary_name)
        dominated = any(
            dominates(_directed_values(other, directions, primary_name), row_values, directions)
            for other in rows
            if other.experiment_id != row.experiment_id
        )
        if not dominated:
            frontier.append(row.experiment_id)
    return frontier


def _candidate_row(
    experiment: Experiment,
    executions: list[ExperimentExecution],
    baseline: BaselineRun,
    spec: ContractSpec,
    validation: ValidationSummary | None,
) -> CandidateRow:
    direction = spec.objective.primary_metric.direction
    assert baseline.metrics is not None
    base_value = baseline.metrics.primary_metric.value

    full = latest_full_execution(executions, experiment.experiment_id)
    metrics = full.metrics if full is not None else None
    primary_value = metrics.primary_metric.value if metrics is not None else None
    return CandidateRow(
        experiment_id=experiment.experiment_id,
        title=experiment.title,
        status=experiment.status,
        confidence=_CONFIDENCE_BY_STATUS.get(experiment.status, "speculative"),
        primary_value=primary_value,
        primary_delta=(primary_value - base_value) if primary_value is not None else None,
        primary_delta_pct=(
            relative_improvement_pct(base_value, primary_value, direction)
            if primary_value is not None
            else None
        ),
        secondary_values=dict(metrics.secondary_metrics) if metrics is not None else {},
        constraints=full.constraints if full is not None else [],
        decision=experiment.decision,
        benchmark_stage=full.benchmark_stage if full is not None else None,
        validation=validation,
    )


def build_ranking_report(
    run_id: str,
    baseline: BaselineRun,
    experiments: list[Experiment],
    executions: list[ExperimentExecution],
    spec: ContractSpec,
    *,
    tradeoff_material_pct: float,
    validations: dict[str, ValidationSummary] | None = None,
) -> RankingReport:
    assert baseline.metrics is not None
    directions = inferred_directions(spec)
    validations = validations or {}

    rows = [
        _candidate_row(
            experiment,
            executions,
            baseline,
            spec,
            validations.get(experiment.experiment_id),
        )
        for experiment in experiments
    ]
    viable = [row for row in rows if row.status in VIABLE_STATUSES]
    rejected = [row for row in rows if row.status not in VIABLE_STATUSES]

    direction = spec.objective.primary_metric.direction
    viable.sort(
        key=lambda row: (
            -signed_improvement(
                baseline.metrics.primary_metric.value,  # type: ignore[union-attr]
                row.primary_value if row.primary_value is not None else float("-inf"),
                direction,
            ),
            row.experiment_id,
        )
    )

    primary_name = spec.objective.primary_metric.name
    pareto_ids = (
        pareto_frontier(viable, directions, primary_name)
        if len(viable) > 1
        else [row.experiment_id for row in viable]
    )

    # Trade-off notes for frontier members that differ materially on a
    # direction-inferable secondary metric.
    notes: list[str] = []
    frontier_rows = [row for row in viable if row.experiment_id in pareto_ids]
    for i, first in enumerate(frontier_rows):
        for second in frontier_rows[i + 1 :]:
            for name, metric_direction in directions.items():
                if name == spec.objective.primary_metric.name:
                    continue
                a_value = first.secondary_values.get(name)
                b_value = second.secondary_values.get(name)
                if a_value is None or b_value is None or b_value == 0:
                    continue
                diff_pct = abs(a_value - b_value) / abs(b_value) * 100.0
                if diff_pct >= tradeoff_material_pct:
                    better = (
                        first.experiment_id
                        if signed_improvement(b_value, a_value, metric_direction) > 0
                        else second.experiment_id
                    )
                    notes.append(
                        f"{first.experiment_id} vs {second.experiment_id}: {name} differs "
                        f"by {diff_pct:.1f}% ({better} is better on {name})"
                    )

    single_winner = None
    if len(pareto_ids) == 1 and viable:
        single_winner = pareto_ids[0]

    caveats = []
    undirected = [name for name in spec.objective.secondary_metrics if name not in directions]
    if undirected:
        caveats.append(
            f"Secondary metrics without a declared direction are shown but not ranked: "
            f"{', '.join(undirected)}."
        )
    if any(row.confidence == "promising" for row in viable):
        caveats.append(ONE_OFF_CAVEAT)

    baseline_row = CandidateRow(
        experiment_id="baseline",
        title="Baseline (frozen)",
        status=ExperimentStatus.VALIDATED,  # placeholder status for display
        confidence="baseline",
        primary_value=baseline.metrics.primary_metric.value,
        secondary_values=dict(baseline.metrics.secondary_metrics),
    )

    return RankingReport(
        run_id=run_id,
        baseline_row=baseline_row,
        candidates=viable,
        rejected=rejected,
        pareto_ids=pareto_ids,
        single_winner=single_winner,
        trade_off_notes=notes,
        caveats=caveats,
    )
