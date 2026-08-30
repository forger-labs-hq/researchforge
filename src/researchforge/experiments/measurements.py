"""What an experiment actually measured, read back from its execution records.

An experiment can have several execution records — a screening attempt, a full
benchmark, validation repeats — and only the full benchmark is comparable with
the frozen baseline.  Everything that ranks, selects or summarizes experiments
needs the same answer to "what did this score", so it is computed here once.
"""

from __future__ import annotations

from researchforge.domain.experiment import (
    BenchmarkStage,
    ExecutionRecordStatus,
    ExperimentExecution,
)


def latest_full_execution(
    executions: list[ExperimentExecution], experiment_id: str
) -> ExperimentExecution | None:
    """The most recent full-benchmark attempt for one experiment, if any."""
    attempts = [
        execution
        for execution in executions
        if execution.experiment_id == experiment_id
        and execution.benchmark_stage is BenchmarkStage.FULL
    ]
    return attempts[-1] if attempts else None


def measured_values(executions: list[ExperimentExecution]) -> dict[str, float]:
    """Primary-metric value per experiment, from its successful full benchmark.

    Later records win, so a re-run replaces an earlier measurement. Experiments
    that never completed a full benchmark are absent rather than zero.
    """
    values: dict[str, float] = {}
    for execution in executions:
        if execution.benchmark_stage is not BenchmarkStage.FULL:
            continue
        if execution.status is not ExecutionRecordStatus.SUCCEEDED:
            continue
        if execution.metrics is None:
            continue
        values[execution.experiment_id] = execution.metrics.primary_metric.value
    return values
