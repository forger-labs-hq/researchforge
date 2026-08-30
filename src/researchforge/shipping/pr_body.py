"""Draft PR title and body builders (pure, from stored records only).

Sections mirror the spec's draft-PR contents list: objective, baseline,
motivating papers, experiments attempted, rejected approaches, validated
metrics, constraints, changed files, reproduction command, risks and
limitations, and the report path.
"""

from __future__ import annotations

from researchforge.domain.baseline import BaselineRun
from researchforge.domain.contract import ExperimentContract
from researchforge.domain.experiment import (
    Experiment,
    ExperimentExecution,
    ExperimentStatus,
    ValidationSummary,
)
from researchforge.domain.hypothesis import Hypothesis
from researchforge.domain.paper import Paper


def build_pr_title(hypothesis: Hypothesis, winner: Experiment, contract: ExperimentContract) -> str:
    objective = contract.spec.objective.description.strip().rstrip(".")
    title = f"{objective} ({winner.experiment_id})"
    return title if len(title) <= 90 else f"{winner.title} ({winner.experiment_id})"


def _tests_note(test_command: str | None) -> str:
    if test_command:
        return (
            f"Existing tests were executed via the contract test_command (`{test_command}`) "
            "during pre-ship confirmation; no new tests were authored — test authoring is a "
            "Claude-assisted step (Phase 1E)."
        )
    return (
        "No test command is configured in the contract; no new tests were authored — "
        "test authoring is a Claude-assisted step (Phase 1E)."
    )


def build_pr_body(
    *,
    contract: ExperimentContract,
    hypothesis: Hypothesis,
    papers: list[Paper],
    experiments: list[Experiment],
    executions: list[ExperimentExecution],
    winner: Experiment,
    baseline: BaselineRun,
    validation: ValidationSummary | None,
    preship: ExperimentExecution | None,
    report_path: str | None,
    provenance: str | None = None,
) -> str:
    spec = contract.spec
    primary = spec.objective.primary_metric.name
    assert baseline.metrics is not None
    lines: list[str] = []

    lines += ["## Objective", "", spec.objective.description.strip(), ""]

    lines += [
        "## Current baseline",
        "",
        f"- Commit: `{contract.baseline_commit}`",
        f"- {primary}: {baseline.metrics.primary_metric.value}",
    ]
    for name, metric_value in baseline.metrics.secondary_metrics.items():
        lines.append(f"- {name}: {metric_value}")
    lines += [f"- Environment: {baseline.execution_mode.value}", ""]

    lines += ["## Motivating papers", ""]
    papers_by_id = {p.paper_id: p for p in papers}
    cited = [papers_by_id[pid] for pid in hypothesis.supporting_paper_ids if pid in papers_by_id]
    if cited:
        for paper in cited:
            lines.append(
                f"- **{paper.paper_id}** — {', '.join(paper.authors)}. *{paper.title}* "
                f"({paper.published_at.year}). <{paper.pdf_url or paper.source_url}>"
            )
    else:
        lines.append("None recorded for this hypothesis.")
    lines.append("")

    lines += [
        "## Experiments attempted",
        "",
        "| Experiment | Title | Status | Result |",
        "|---|---|---|---|",
    ]
    full_by_experiment: dict[str, ExperimentExecution] = {}
    for execution in executions:
        if execution.benchmark_stage.value == "full":
            full_by_experiment[execution.experiment_id] = execution
    for experiment in experiments:
        full = full_by_experiment.get(experiment.experiment_id)
        value = (
            f"{primary}={full.metrics.primary_metric.value}"
            if full is not None and full.metrics is not None
            else "—"
        )
        lines.append(
            f"| {experiment.experiment_id} | {experiment.title} | "
            f"{experiment.status.value} | {value} |"
        )
    lines.append("")

    rejected = [
        e
        for e in experiments
        if e.status
        in (
            ExperimentStatus.REJECTED,
            ExperimentStatus.FAILED_SETUP,
            ExperimentStatus.FAILED_EXECUTION,
        )
    ]
    lines += ["## Rejected approaches", ""]
    if rejected:
        for experiment in rejected:
            reason = experiment.decision.reason if experiment.decision else experiment.status.value
            lines.append(f"- **{experiment.experiment_id}** ({experiment.title}): {reason}")
    else:
        lines.append("None — every attempted variant survived.")
    lines.append("")

    lines += ["## Validated metrics", ""]
    if validation is not None and validation.mean is not None:
        stdev = f"{validation.stdev:.4g}" if validation.stdev is not None else "n/a"
        lines.append(
            f"- {primary}: {baseline.metrics.primary_metric.value} → mean "
            f"{validation.mean:.4g} across {validation.attempts} validation run(s) "
            f"(stdev {stdev})"
        )
    if preship is not None and preship.metrics is not None:
        lines.append(f"- Pre-ship confirmation: {primary}={preship.metrics.primary_metric.value}")
    lines.append("")

    lines += ["## Constraints", ""]
    constraint_rows = preship.constraints if preship is not None else []
    if constraint_rows:
        for constraint in constraint_rows:
            status = "pass" if constraint.passed else "FAIL"
            lines.append(
                f"- {constraint.name} {constraint.operator.value} {constraint.threshold}: "
                f"{status} (observed {constraint.observed})"
            )
    else:
        lines.append("No hard constraints in the contract.")
    lines.append("")

    lines += ["## Changed files", ""]
    lines += [f"- `{path}`" for path in winner.changed_files]
    lines.append("")
    if provenance:
        lines += [provenance, ""]

    lines += [
        "## Reproduction",
        "",
        "```bash",
        "researchforge baseline run",
        f"researchforge experiment plan {hypothesis.hypothesis_id}",
        "researchforge experiment import .researchforge/experiments/plan.yaml",
        f"researchforge experiment approve {winner.plan_id} --yes",
        f"researchforge experiment run {winner.plan_id}",
        "researchforge validate <run-id>",
        "```",
        "",
        f"Evaluation command: `{spec.execution.full_command}` "
        f"(result file: `{spec.execution.result_file}`)",
        "",
    ]

    lines += ["## Risks and limitations", ""]
    for limitation in hypothesis.limitations:
        lines.append(f"- {limitation}")
    lines.append(
        f"- Results were measured on a single machine "
        f"({baseline.fingerprint.platform}) in {baseline.execution_mode.value} mode; "
        "they may not generalize beyond the tested conditions."
    )
    if baseline.execution_mode.value == "venv":
        lines.append(
            "- venv execution isolates dependencies, not the machine — see the "
            "ResearchForge docs for the trust model."
        )
    lines.append(f"- {_tests_note(spec.execution.test_command)}")
    lines.append("")

    lines += ["## ResearchForge report", ""]
    if report_path:
        lines.append(f"`{report_path}`")
    else:
        lines.append("Run `researchforge report build` for the full engineering report.")
    lines += [
        "",
        "---",
        "",
        "Draft PR generated by ResearchForge from recorded experiment data.",
        "",
    ]
    return "\n".join(lines)
