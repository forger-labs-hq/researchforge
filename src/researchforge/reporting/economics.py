"""What a project cost in time, and what the loop declined to spend.

Every figure here is either measured or arithmetic on measured quantities. There
is deliberately no estimate of what a human would have spent instead: that
number cannot be checked, and one unfalsifiable claim discredits the careful
ones standing next to it.

The distinction that matters most is between **time used** and **time avoided**.
Time used is a sum of recorded durations. Time avoided is a count of runs that
never happened multiplied by what a run of that kind actually took *in this
project* — so the multiplication is shown rather than the product asserted, and
a project with nothing to average over reports no figure at all.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from researchforge.ai.usage import AiCall
from researchforge.config.settings import ModelPrice
from researchforge.domain.baseline import BaselineRun
from researchforge.domain.experiment import (
    BenchmarkStage,
    Experiment,
    ExperimentExecution,
    ExperimentStatus,
)
from researchforge.experiments.measurements import measured_values

BASELINE_NODE = "baseline"

#: Statuses that mean the experiment was carried forward rather than discarded.
KEPT_STATUSES = frozenset(
    {
        ExperimentStatus.PROMISING,
        ExperimentStatus.VALIDATING,
        ExperimentStatus.VALIDATED,
        ExperimentStatus.IMPLEMENTATION_READY,
    }
)

FAILED_STATUSES = frozenset({ExperimentStatus.FAILED_SETUP, ExperimentStatus.FAILED_EXECUTION})


@dataclass(frozen=True)
class StageTime:
    """Recorded compute, split by what the time was spent doing."""

    baseline: float = 0.0
    screening: float = 0.0
    full: float = 0.0
    validation: float = 0.0

    @property
    def total(self) -> float:
        return self.baseline + self.screening + self.full + self.validation


@dataclass(frozen=True)
class Avoided:
    """Runs the system declined to start, and what one would have cost.

    `mean_full_seconds` is this project's own measured average, so the claim is
    "at what a full benchmark took here" rather than a guess. When nothing has
    completed a full benchmark there is no average, and `seconds` is None rather
    than zero — an unknown must never render as "saved nothing".
    """

    screened_out: int = 0
    cancelled: int = 0
    mean_full_seconds: float | None = None

    @property
    def runs(self) -> int:
        return self.screened_out + self.cancelled

    @property
    def seconds(self) -> float | None:
        if self.mean_full_seconds is None or self.runs == 0:
            return None
        return self.runs * self.mean_full_seconds


@dataclass(frozen=True)
class Record:
    """What survives: every attempt, including the ones that did not work."""

    experiments: int = 0
    failed: int = 0
    cancelled: int = 0
    rejected: int = 0
    kept: int = 0
    with_lineage: int = 0
    """Experiments that name at least one measured ancestor."""


@dataclass(frozen=True)
class Caught:
    """Changes that would have shipped and should not have."""

    constraint_violations: list[str] = field(default_factory=list)
    """Broke a hard limit at some stage and were stopped by it."""

    no_ops: list[str] = field(default_factory=list)
    """Measured exactly what everything they build on measured."""


@dataclass(frozen=True)
class TokenSpend:
    """What the model calls cost, in tokens and — where priced — in dollars.

    `unpriced_models` is not a footnote. A model with no rate contributes zero
    dollars, so without naming it the total reads as complete when it is not.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    by_purpose: dict[str, int] = field(default_factory=dict)
    """Total tokens per activity — planning, synthesis, observation."""

    unpriced_models: list[str] = field(default_factory=list)

    estimated_tokens: int = 0
    """Tokens sized from an IDE handshake rather than reported by a provider.

    Held apart from the metered total and never priced: these were spent inside
    someone's IDE subscription, on a model and a tokenizer we do not know.
    """

    estimated_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_estimates(self) -> bool:
        return self.estimated_calls > 0

    @property
    def fully_priced(self) -> bool:
        return not self.unpriced_models


@dataclass(frozen=True)
class Economics:
    stages: StageTime
    by_outcome: dict[str, float]
    avoided: Avoided
    record: Record
    caught: Caught
    tokens: TokenSpend = field(default_factory=TokenSpend)
    compute_usd: float | None = None
    """Benchmark compute priced at the configured hourly rate. None when unset."""

    @property
    def measured_seconds(self) -> float:
        return self.stages.total

    @property
    def total_usd(self) -> float | None:
        """Everything the run cost, when enough of it has a price."""
        if self.compute_usd is None:
            return self.tokens.usd if self.tokens.total_tokens else None
        return self.compute_usd + self.tokens.usd


def humanize_seconds(seconds: float | None) -> str:
    """A duration at the precision a reader can act on.

    Two significant units at most: "4h 12m" is a number someone can hold, and
    "4h 12m 07s" is a number they have to parse first.
    """
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _stage_time(
    executions: Sequence[ExperimentExecution], baselines: Sequence[BaselineRun]
) -> StageTime:
    per_stage = dict.fromkeys(BenchmarkStage, 0.0)
    for execution in executions:
        per_stage[execution.benchmark_stage] += execution.duration_seconds
    return StageTime(
        baseline=sum(run.duration_seconds for run in baselines),
        screening=per_stage[BenchmarkStage.SCREENING],
        full=per_stage[BenchmarkStage.FULL],
        validation=per_stage[BenchmarkStage.VALIDATION],
    )


def _outcome_of(experiment: Experiment) -> str:
    if experiment.status in KEPT_STATUSES:
        return "kept"
    if experiment.status in FAILED_STATUSES:
        return "failed"
    if experiment.status is ExperimentStatus.CANCELLED:
        return "cancelled"
    if experiment.status is ExperimentStatus.REJECTED:
        return "rejected"
    return "in progress"


def _time_by_outcome(
    experiments: Sequence[Experiment], executions: Sequence[ExperimentExecution]
) -> dict[str, float]:
    """Where the compute went, grouped by how each experiment ended.

    Answers "what did the ones that did not work cost me?", which is invisible
    when the only view is a ranking of the ones that did.
    """
    outcome_of = {e.experiment_id: _outcome_of(e) for e in experiments}
    totals: dict[str, float] = {}
    for execution in executions:
        outcome = outcome_of.get(execution.experiment_id)
        if outcome is None:
            continue
        totals[outcome] = totals.get(outcome, 0.0) + execution.duration_seconds
    return dict(sorted(totals.items()))


def _mean_full_seconds(executions: Sequence[ExperimentExecution]) -> float | None:
    durations = [
        e.duration_seconds
        for e in executions
        if e.benchmark_stage is BenchmarkStage.FULL and e.duration_seconds > 0
    ]
    if not durations:
        return None
    return sum(durations) / len(durations)


def _avoided(
    experiments: Sequence[Experiment], executions: Sequence[ExperimentExecution]
) -> Avoided:
    """Full benchmarks that screening and the stall rule kept from happening."""
    screened: set[str] = set()
    ran_full: set[str] = set()
    for execution in executions:
        if execution.benchmark_stage is BenchmarkStage.SCREENING:
            screened.add(execution.experiment_id)
        elif execution.benchmark_stage is BenchmarkStage.FULL:
            ran_full.add(execution.experiment_id)

    cancelled = {e.experiment_id for e in experiments if e.status is ExperimentStatus.CANCELLED}
    # A cancelled experiment is counted once, under the rule that stopped it.
    return Avoided(
        screened_out=len(screened - ran_full - cancelled),
        cancelled=len(cancelled - ran_full),
        mean_full_seconds=_mean_full_seconds(executions),
    )


def _record(experiments: Sequence[Experiment]) -> Record:
    outcomes = [_outcome_of(e) for e in experiments]
    return Record(
        experiments=len(experiments),
        failed=outcomes.count("failed"),
        cancelled=outcomes.count("cancelled"),
        rejected=outcomes.count("rejected"),
        kept=outcomes.count("kept"),
        with_lineage=sum(1 for e in experiments if e.parent_experiment_ids),
    )


def _caught(
    experiments: Sequence[Experiment],
    executions: Sequence[ExperimentExecution],
    baseline_value: float | None,
) -> Caught:
    """Changes a hard limit stopped, and changes that moved nothing.

    A constraint is checked at screening as well as at the full benchmark, and a
    violation caught early is the most valuable kind — so this counts a
    violation at any stage rather than only among experiments that got far
    enough to have a full-benchmark value.
    """
    known = {e.experiment_id for e in experiments}
    violations = {
        execution.experiment_id
        for execution in executions
        if execution.experiment_id in known
        and any(c.passed is False for c in execution.constraints)
    }

    values = measured_values(list(executions))
    no_ops: list[str] = []
    for experiment in experiments:
        value = values.get(experiment.experiment_id)
        if value is None:
            continue
        references = [values.get(p) for p in experiment.parent_experiment_ids] or [baseline_value]
        if all(r is not None and value == r for r in references):
            no_ops.append(experiment.experiment_id)

    return Caught(constraint_violations=sorted(violations), no_ops=sorted(no_ops))


def price_of(model: str, prices: Mapping[str, ModelPrice]) -> ModelPrice | None:
    """The rate for a model, matched by longest prefix.

    Longest wins so a specific entry beats a general one: with both "gpt-4o" and
    "gpt-4o-mini" configured, the mini model must not be billed at the full rate.
    """
    matches = [key for key in prices if model.startswith(key)]
    if not matches:
        return None
    return prices[max(matches, key=len)]


def token_spend(calls: Sequence[AiCall], prices: Mapping[str, ModelPrice]) -> TokenSpend:
    """Total up model calls and price them where a rate is known."""
    usd = 0.0
    input_tokens = 0
    output_tokens = 0
    estimated_tokens = 0
    estimated_calls = 0
    by_purpose: dict[str, int] = {}
    unpriced: set[str] = set()

    for call in calls:
        if call.usage.total:
            by_purpose[call.purpose] = by_purpose.get(call.purpose, 0) + call.usage.total

        # Estimates are counted and shown, but never mixed into the metered
        # total and never priced — the whole value of the split is that a
        # reader can tell which number is which.
        if call.estimated:
            estimated_tokens += call.usage.total
            estimated_calls += 1
            continue

        input_tokens += call.usage.input_tokens
        output_tokens += call.usage.output_tokens

        price = price_of(call.model, prices)
        if price is None:
            if call.usage.total:
                unpriced.add(call.model)
            continue
        usd += call.usage.input_tokens / 1_000_000 * price.input
        usd += call.usage.output_tokens / 1_000_000 * price.output

    return TokenSpend(
        calls=len(calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=usd,
        by_purpose=dict(sorted(by_purpose.items(), key=lambda kv: -kv[1])),
        unpriced_models=sorted(unpriced),
        estimated_tokens=estimated_tokens,
        estimated_calls=estimated_calls,
    )


def compute_economics(
    experiments: Sequence[Experiment],
    executions: Sequence[ExperimentExecution],
    baselines: Sequence[BaselineRun],
    baseline_value: float | None,
    calls: Sequence[AiCall] = (),
    prices: Mapping[str, ModelPrice] | None = None,
    usd_per_hour: float = 0.0,
) -> Economics:
    """Total up what a project's records cost, from records already in hand.

    Kept separate from the database read so the hub and the monitor — which hold
    a snapshot rather than a connection — report the same arithmetic as the
    dashboard instead of each growing its own version of it.
    """
    stages = _stage_time(executions, baselines)
    return Economics(
        stages=stages,
        by_outcome=_time_by_outcome(experiments, executions),
        avoided=_avoided(experiments, executions),
        record=_record(experiments),
        caught=_caught(experiments, executions, baseline_value),
        tokens=token_spend(calls, prices if prices is not None else {}),
        compute_usd=(stages.total / 3600 * usd_per_hour if usd_per_hour > 0 else None),
    )


def build_economics(conn: sqlite3.Connection) -> Economics:
    """Read the project's records and total up what they cost."""
    from researchforge.config.settings import load_settings
    from researchforge.storage.ai_call_repository import list_ai_calls
    from researchforge.storage.baseline_repository import (
        get_latest_successful_baseline,
        list_baseline_runs,
    )
    from researchforge.storage.experiment_repository import list_executions, list_experiments

    frozen = get_latest_successful_baseline(conn)
    settings = load_settings()
    return compute_economics(
        list_experiments(conn),
        list_executions(conn),
        list_baseline_runs(conn),
        frozen.metrics.primary_metric.value
        if frozen is not None and frozen.metrics is not None
        else None,
        calls=list_ai_calls(conn),
        prices=settings.model_prices,
        usd_per_hour=settings.local_compute_usd_per_hour,
    )
