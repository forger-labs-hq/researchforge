"""The living research log that accumulates knowledge across autorun rounds.

`.researchforge/research-log.md` is what the AI reads before re-synthesizing.
It records which hypotheses were tried and how they measured, the observations
drawn from the runs, the current best experiment, and which directions are
exhausted — so a later round proposes something new instead of repeating a
result that is already on record.

Everything written here is derived from recorded measurements; the log never
asserts anything the database cannot back up.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from researchforge.config.paths import research_log_path
from researchforge.domain.baseline import BaselineRun
from researchforge.domain.contract import MetricDirection
from researchforge.domain.experiment import Experiment, ExperimentStatus

BEST_SECTION_PATTERN = r"## Current Best\n.*?(?=\n## |\Z)"

STATUS_ICONS = {
    ExperimentStatus.PROMISING: "✓",
    ExperimentStatus.VALIDATED: "✓✓",
    ExperimentStatus.IMPLEMENTATION_READY: "✓✓✓",
    ExperimentStatus.REJECTED: "✗",
    ExperimentStatus.FAILED_SETUP: "⚠",
    ExperimentStatus.FAILED_EXECUTION: "⚠",
    ExperimentStatus.CANCELLED: "·",
}


def log_path(base: Path | None = None) -> Path:
    return research_log_path(base)


def improvement_pct(value: float, baseline: float, direction: MetricDirection) -> float:
    """Signed improvement of `value` over `baseline`, as a percentage."""
    if baseline == 0:
        return 0.0
    raw = (value - baseline) / abs(baseline) * 100
    return raw if direction is MetricDirection.MAXIMIZE else -raw


def beats_baseline(value: float, baseline: float, direction: MetricDirection) -> bool:
    if direction is MetricDirection.MAXIMIZE:
        return value > baseline
    return value < baseline


def build_initial_log(objective: str, baseline: BaselineRun) -> str:
    """The log as it exists before any experiment has run."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    metric_name = baseline.metrics.primary_metric.name if baseline.metrics else "metric"
    baseline_value = baseline.metrics.primary_metric.value if baseline.metrics else 0.0
    return f"""# ResearchForge Research Log
_Started: {now}_

## Objective
{objective}

## Baseline
- **{metric_name}** = {baseline_value:.4f}  (frozen reference — every improvement is measured here)

## Current Best
- **Baseline** ({metric_name} = {baseline_value:.4f})

## Experiment History
_No experiments run yet._

## Directions Exhausted
_None yet._

## Open Questions
_To be filled as experiments run._
"""


def render_best_section(
    best_experiment: Experiment | None,
    best_value: float,
    baseline_value: float,
    metric_name: str,
    direction: MetricDirection,
) -> str:
    if best_experiment is None or not beats_baseline(best_value, baseline_value, direction):
        return (
            "## Current Best\n"
            f"- **Baseline** ({metric_name} = {baseline_value:.4f}) — nothing has beaten it yet\n"
        )
    delta = improvement_pct(best_value, baseline_value, direction)
    return (
        "## Current Best\n"
        f"- **{best_experiment.experiment_id}** ({metric_name} = {best_value:.4f}, "
        f"{delta:+.1f}% vs baseline)\n"
        f"- Change: {best_experiment.change_summary}\n"
    )


def update_log_after_round(
    log_file: Path,
    round_num: int,
    experiments: list[Experiment],
    baseline_value: float,
    metric_name: str,
    best_experiment: Experiment | None,
    best_value: float,
    direction: MetricDirection = MetricDirection.MAXIMIZE,
    observations: dict[str, str] | None = None,
) -> None:
    """Refresh the current-best section and append this round's outcomes."""
    existing = log_file.read_text(encoding="utf-8") if log_file.is_file() else ""
    observations = observations or {}

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"\n## Round {round_num} — {timestamp}\n"]

    if best_experiment is not None and beats_baseline(best_value, baseline_value, direction):
        delta = improvement_pct(best_value, baseline_value, direction)
        lines.append(
            f"**Current best: {best_experiment.experiment_id}** "
            f"({metric_name} = {best_value:.4f}, {delta:+.1f}% vs baseline)\n"
        )
    else:
        lines.append(f"**Current best: baseline** ({metric_name} = {baseline_value:.4f})\n")

    lines.append("### Experiments this round\n")
    if not experiments:
        lines.append("_No experiment completed this round._\n")
    for experiment in experiments:
        icon = STATUS_ICONS.get(experiment.status, "·")
        decision = f" — {experiment.decision.reason}" if experiment.decision else ""
        observation = observations.get(experiment.experiment_id, "")
        note = f"\n  > {observation}" if observation else ""
        lines.append(
            f"- [{icon}] **{experiment.experiment_id}**: {experiment.change_summary}"
            f"{decision}{note}\n"
        )

    best_section = render_best_section(
        best_experiment, best_value, baseline_value, metric_name, direction
    )
    if re.search(BEST_SECTION_PATTERN, existing, re.DOTALL):
        updated = re.sub(BEST_SECTION_PATTERN, best_section.rstrip(), existing, flags=re.DOTALL)
    else:
        updated = existing + "\n" + best_section
    log_file.write_text(updated + "\n".join(lines), encoding="utf-8")


def build_resynth_context(log_file: Path) -> str:
    """The log content to include in a re-synthesis prompt."""
    if not log_file.is_file():
        return ""
    return log_file.read_text(encoding="utf-8")


def build_measured_summary(
    experiments: list[Experiment],
    values: dict[str, float],
    baseline_value: float,
    metric_name: str,
    direction: MetricDirection,
) -> str:
    """The outcomes on record, for a project whose log does not cover them.

    The log is written by the autorun loop, so a project driven by hand has
    nothing accumulated.  This renders the same facts straight from the stored
    experiments: what was tried and how it measured, and nothing else.
    """
    lines = [
        "# Experiment Outcomes On Record",
        "",
        f"Baseline **{metric_name}** = {baseline_value:.4f} "
        f"({'higher' if direction is MetricDirection.MAXIMIZE else 'lower'} is better)",
        "",
    ]
    measured = [e for e in experiments if e.experiment_id in values]
    unmeasured = [e for e in experiments if e.experiment_id not in values]

    if not measured:
        lines.append("_No experiment has completed a full benchmark yet._")
    for experiment in sorted(
        measured,
        key=lambda e: improvement_pct(values[e.experiment_id], baseline_value, direction),
        reverse=True,
    ):
        value = values[experiment.experiment_id]
        delta = improvement_pct(value, baseline_value, direction)
        icon = STATUS_ICONS.get(experiment.status, "·")
        decision = f" — {experiment.decision.reason}" if experiment.decision else ""
        lines.append(
            f"- [{icon}] **{experiment.experiment_id}** {metric_name} = {value:.4f} "
            f"({delta:+.1f}% vs baseline): {experiment.change_summary}{decision}"
        )
        if experiment.observation:
            lines.append(f"  > {experiment.observation}")

    if unmeasured:
        lines.append("")
        lines.append("## Tried without a usable measurement")
        for experiment in unmeasured:
            icon = STATUS_ICONS.get(experiment.status, "·")
            reason = f" — {experiment.decision.reason}" if experiment.decision else ""
            lines.append(
                f"- [{icon}] **{experiment.experiment_id}** "
                f"({experiment.status.value}): {experiment.change_summary}{reason}"
            )
            if experiment.observation:
                lines.append(f"  > {experiment.observation}")
    return "\n".join(lines) + "\n"


MAX_RESULTS_CONTEXT_CHARS = 6000


def results_instructions(results_context: str) -> list[str]:
    """Prompt lines that put measured outcomes in front of the synthesizer.

    Shared by the autorun loop and `research synthesize --from-results` so both
    ask for the same thing: ideas that are new and that follow the evidence.
    """
    return [
        "RESULTS SO FAR — measured outcomes from this project. Use them to avoid "
        "re-proposing ideas that were already tried and to build on what "
        "measurably worked:",
        results_context[:MAX_RESULTS_CONTEXT_CHARS],
        "Generate hypotheses that are materially different from everything already "
        "tried above, and that follow the directions the measurements support.",
    ]
