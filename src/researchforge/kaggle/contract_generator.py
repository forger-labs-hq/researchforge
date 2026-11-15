"""Auto-generate ResearchForge contracts for Kaggle competitions.

All metric information comes from CompetitionMetadata (read from the Kaggle
page and/or AI extraction) — nothing is hardcoded per competition slug.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from researchforge.domain.contract import (
    ContractExecutionMode,
    ContractSpec,
    ExecutionSection,
    MetricDirection,
    ObjectiveSection,
    PermissionsSection,
    PrimaryMetric,
    ProjectSection,
    RepositorySection,
)
from researchforge.domain.project import ProjectMode

if TYPE_CHECKING:
    from researchforge.kaggle.competition_reader import CompetitionMetadata


def generate_contract_for_kaggle(
    metadata: CompetitionMetadata,
    baseline_cv_score: float,
    artifacts_dir: Path,
    target_improvement_pct: float = 0.03,
) -> ContractSpec:
    """Generate a ResearchForge contract from competition metadata.

    Args:
        metadata: Fully populated CompetitionMetadata (from competition_reader).
        baseline_cv_score: Measured baseline score.
        artifacts_dir: Where to store contract artefacts.
        target_improvement_pct: Desired relative improvement, e.g. 0.03 = 3%.
            Applied in the correct direction (higher or lower) automatically.

    Returns:
        ContractSpec ready for user approval.
    """
    direction = (
        MetricDirection.MAXIMIZE
        if metadata.metric_direction == "maximize"
        else MetricDirection.MINIMIZE
    )

    if direction == MetricDirection.MAXIMIZE:
        target_value = baseline_cv_score * (1.0 + target_improvement_pct)
    else:
        target_value = baseline_cv_score * (1.0 - target_improvement_pct)

    primary_metric = PrimaryMetric(
        name=metadata.metric_name,
        direction=direction,
        target_value=target_value,
    )

    objective = ObjectiveSection(
        description=(
            f"Improve {metadata.title or metadata.slug} — "
            f"{metadata.task_type} · {metadata.metric_name}"
        ),
        primary_metric=primary_metric,
    )

    project = ProjectSection(
        name=f"kaggle-{metadata.slug}",
        mode=ProjectMode.IMPROVE_REPOSITORY,
    )

    execution = ExecutionSection(
        mode=ContractExecutionMode.AUTO,
        full_command="python benchmark.py",
    )

    return ContractSpec(
        version=1,
        objective=objective,
        project=project,
        repository=RepositorySection(),
        execution=execution,
        permissions=PermissionsSection(),
    )
