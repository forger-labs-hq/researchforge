"""Shared synthesis service called by both `research synthesize` and `hypotheses generate`.

Extracts the core logic so both CLI commands delegate here rather than
duplicating code.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from researchforge.ai.providers import AiProvider, get_provider
from researchforge.ai.synthesis import synthesize as _ai_synthesize
from researchforge.ai.synthesis import write_artifacts
from researchforge.config.paths import research_log_path
from researchforge.config.settings import load_settings
from researchforge.experiments.measurements import measured_values
from researchforge.research.context_export import SynthesisContext, build_context
from researchforge.research.importers import (
    ImportResult,
    import_additional_hypotheses,
    import_hypotheses,
    import_landscape,
)
from researchforge.research.research_log import (
    build_measured_summary,
    build_resynth_context,
    results_instructions,
)
from researchforge.storage.baseline_repository import get_latest_baseline
from researchforge.storage.contract_repository import get_active_contract
from researchforge.storage.db import open_project_db
from researchforge.storage.experiment_repository import (
    get_run,
    list_executions,
    list_experiments,
)
from researchforge.storage.project_repository import get_project
from researchforge.storage.scan_repository import get_latest_scan


@dataclass
class SynthesisOutcome:
    landscape_path: Path
    hypotheses_path: Path
    landscape_result: ImportResult
    hypotheses_result: ImportResult
    added_hypotheses: list[str] = field(default_factory=list)
    """Ids stored by a results-grounded run, which adds instead of replacing."""

    restated_hypotheses: list[str] = field(default_factory=list)
    """Stored ids whose ideas were proposed again and therefore skipped."""

    @property
    def ok(self) -> bool:
        return self.landscape_result.ok and self.hypotheses_result.ok


def resolve_provider(
    provider_hint: str | None = None,
    model_hint: str | None = None,
) -> AiProvider:
    """Resolve the AI provider or raise RuntimeError with a clear message."""
    provider = get_provider(provider_hint=provider_hint, model_hint=model_hint)
    if provider is None:
        raise RuntimeError(
            "No AI provider configured.\n"
            "Set one of:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  GEMINI_API_KEY=...\n"
            "  OPENAI_API_KEY=sk-...\n"
            "Or pass --provider anthropic|google|openai"
        )
    return provider


def build_results_context(run_id: str | None = None) -> str:
    """Everything this project has measured, as text for the synthesizer.

    The autorun loop's research log comes first when it exists, since it
    accumulates observations across rounds.  The stored experiments are then
    summarized from the database, so a project driven by hand — which has no
    log — still gets its outcomes in front of the AI.  Returns "" when there
    is nothing measured to report.
    """
    sections: list[str] = []
    log = build_resynth_context(research_log_path())
    if log:
        sections.append(log)

    with closing(open_project_db()) as conn:
        baseline = get_latest_baseline(conn)
        contract = get_active_contract(conn)
        if run_id is None:
            experiments = list_experiments(conn)
            executions = list_executions(conn)
        else:
            run = get_run(conn, run_id)
            if run is None:
                raise RuntimeError(f"Unknown run: {run_id}.")
            experiments = list_experiments(conn, run.plan_id)
            executions = list_executions(conn, run_id=run_id)

        if not experiments or contract is None or baseline is None or baseline.metrics is None:
            return "\n\n".join(sections)

        metric = contract.spec.objective.primary_metric
        sections.append(
            build_measured_summary(
                experiments,
                measured_values(executions),
                baseline.metrics.primary_metric.value,
                metric.name,
                metric.direction,
            )
        )
    return "\n\n".join(sections)


def _with_results(bundle: SynthesisContext, results_context: str) -> SynthesisContext:
    """The same bundle, with measured outcomes appended to its instructions."""
    return bundle.model_copy(
        update={"instructions": [*bundle.instructions, *results_instructions(results_context)]}
    )


def run_synthesis(
    provider: AiProvider,
    *,
    do_import: bool = True,
    results_context: str | None = None,
) -> SynthesisOutcome:
    """Build context, call AI, write artifacts, optionally import them.

    Opens the project database automatically.  Raises RuntimeError on
    configuration problems (no project, no papers).

    With `results_context` the synthesis is grounded in what the project has
    already measured, and the hypotheses are ADDED to the stored set rather
    than replacing it — a later round must not delete the hypotheses whose
    experiments are on record.
    """
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None or project.objective is None:
            raise RuntimeError(
                "No project configured. Run `researchforge project create` first."
            )
        scan = get_latest_scan(conn)
        settings = load_settings()
        bundle = build_context(conn, project, scan, settings)

        if not bundle.papers:
            raise RuntimeError(
                "No papers stored. Run `researchforge research search` first."
            )

    if results_context:
        bundle = _with_results(bundle, results_context)

    # AI synthesis — may raise ValueError on malformed output
    landscape_yaml, hypotheses_yaml = _ai_synthesize(bundle, provider)

    # Write artifacts
    landscape_path = Path(bundle.expected_artifacts.landscape_path)
    hypotheses_path = Path(bundle.expected_artifacts.hypotheses_path)
    write_artifacts(landscape_yaml, hypotheses_yaml, landscape_path, hypotheses_path)

    if not do_import:
        # Return dummy OK results (no import performed)
        dummy = ImportResult()
        return SynthesisOutcome(
            landscape_path=landscape_path,
            hypotheses_path=hypotheses_path,
            landscape_result=dummy,
            hypotheses_result=dummy,
        )

    # Import both artifacts
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None:
            raise RuntimeError("Project not found after synthesis.")

        landscape_result = import_landscape(conn, landscape_path, project.id)
        if results_context:
            additions = import_additional_hypotheses(
                conn, hypotheses_path, project.id, settings
            )
            return SynthesisOutcome(
                landscape_path=landscape_path,
                hypotheses_path=hypotheses_path,
                landscape_result=landscape_result,
                hypotheses_result=additions.result,
                added_hypotheses=additions.added,
                restated_hypotheses=additions.restated,
            )
        hypotheses_result = import_hypotheses(conn, hypotheses_path, project.id, settings)

    return SynthesisOutcome(
        landscape_path=landscape_path,
        hypotheses_path=hypotheses_path,
        landscape_result=landscape_result,
        hypotheses_result=hypotheses_result,
    )
