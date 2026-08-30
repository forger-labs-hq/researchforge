"""Contract draft generation from the stored project and repository scan."""

from __future__ import annotations

import re
from pathlib import Path

from researchforge.domain.contract import (
    ContractSpec,
    ExecutionSection,
    MetricDirection,
    ObjectiveSection,
    PermissionsSection,
    PrimaryMetric,
    ProjectSection,
    RepositorySection,
)
from researchforge.domain.project import Project, ProjectMode
from researchforge.domain.repo_scan import RepoScan

FULL_COMMAND_PLACEHOLDER = "# TODO: command that writes the result file"

# Keyword → (metric name, direction). First match in the objective text wins.
_MINIMIZE_KEYWORDS = (
    ("latency", "latency_ms"),
    ("cost", "average_cost_usd"),
    ("loss", "loss"),
    ("perplexity", "perplexity"),
    ("memory", "memory_mb"),
    ("error", "error_rate"),
    ("time", "runtime_seconds"),
)
_MAXIMIZE_KEYWORDS = (
    ("map50", "map50"),   # object detection — check before generic "map"
    ("map@0", "map50"),
    ("mAP", "map50"),
    ("map", "map50"),
    ("f1", "f1"),
    ("accuracy", "accuracy"),
    ("rouge", "rouge_l"),
    ("bleu", "bleu"),
    ("recall", "recall"),
    ("precision", "precision"),
    ("throughput", "throughput"),
    ("quality", "quality_score"),
    ("score", "score"),
)


def guess_primary_metric(objective_text: str) -> PrimaryMetric:
    lowered = objective_text.lower()
    # Maximize keywords first: objectives like "improve F1 without increasing
    # latency" name the optimization target before the constraint.
    for keyword, name in _MAXIMIZE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return PrimaryMetric(name=name, direction=MetricDirection.MAXIMIZE)
    for keyword, name in _MINIMIZE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return PrimaryMetric(name=name, direction=MetricDirection.MINIMIZE)
    return PrimaryMetric(name="primary_metric", direction=MetricDirection.MAXIMIZE)


def _has_real_build_backend(pyproject_path: Path) -> bool:
    """Return True only if pyproject.toml has a real [build-system] with build-backend.

    Many repos (YOLOv5, ultralytics, etc.) have pyproject.toml only for
    tool configuration ([tool.ruff], [tool.pytest], etc.) — not as a build
    backend.  Using ``pip install -e .`` on those fails with
    "Multiple top-level packages discovered".
    """
    try:
        text = pyproject_path.read_text(encoding="utf-8", errors="replace")
        in_build_system = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "[build-system]":
                in_build_system = True
                continue
            if in_build_system:
                if stripped.startswith("[") and not stripped.startswith("[build-system"):
                    break  # left the section
                if stripped.startswith("build-backend"):
                    return True
        return False
    except OSError:
        return False


# Standard benchmark locations checked in priority order
_BENCHMARK_PATTERNS = [
    (
        "benchmarks/evaluate.py",
        "python benchmarks/evaluate.py --quick",
        "python benchmarks/evaluate.py",
    ),
    ("benchmarks/eval.py", "python benchmarks/eval.py --quick", "python benchmarks/eval.py"),
    ("evaluate.py", None, "python evaluate.py"),
    ("eval.py", None, "python eval.py"),
]


def guess_commands(
    scan: RepoScan, repo_root: Path | None = None
) -> tuple[str | None, str | None, str]:
    """Return (setup_command, screening_command_or_None, full_command_or_placeholder)."""
    setup: str | None = None

    # Setup command: prefer requirements.txt unless pyproject.toml has a real
    # build backend (i.e. not just tool config like YOLOv5/ultralytics).
    if scan.python.requirements_files:
        # requirements.txt is the most reliable install method for ML repos
        req = scan.python.requirements_files[0]
        setup = f"pip install --upgrade pip setuptools wheel && pip install -r {req}"
    elif scan.python.has_pyproject:
        if repo_root is not None:
            pyproject_path = repo_root / "pyproject.toml"
            if _has_real_build_backend(pyproject_path):
                setup = "pip install --upgrade pip setuptools wheel && pip install -e ."
            # else: tool-config only pyproject.toml — no pip install -e . needed
        else:
            # Can't verify the build backend without repo_root;
            # fall back to the safe default (works for properly packaged repos)
            setup = "python -m pip install -e ."
    elif scan.python.has_setup_py:
        setup = "pip install --upgrade pip setuptools wheel && pip install -e ."

    # 1. Filesystem check for well-known benchmark paths
    if repo_root is not None:
        for rel_path, screening, full in _BENCHMARK_PATTERNS:
            if (repo_root / rel_path).is_file():
                return setup, screening, full

    # 2. Fall back to stored scan candidates
    for candidate in scan.benchmark_candidates:
        if candidate.endswith(".py"):
            return setup, None, f"python {candidate}"

    return setup, None, FULL_COMMAND_PLACEHOLDER


def build_draft_spec(
    project: Project,
    scan: RepoScan,
    repo_root: Path | None = None,
    target_value: float | None = None,
) -> ContractSpec:
    if project.mode is None or project.objective is None:
        raise ValueError("Project mode and objective must be set before generating a contract.")

    effective_root = repo_root or (Path(scan.repo_path) if scan.repo_path else None)
    setup_command, screening_command, full_command = guess_commands(scan, effective_root)
    metric = guess_primary_metric(project.objective)
    return ContractSpec(
        version=1,
        project=ProjectSection(name=project.name, mode=ProjectMode(project.mode)),
        objective=ObjectiveSection(
            description=project.objective,
            primary_metric=metric.model_copy(update={"target_value": target_value}),
        ),
        repository=RepositorySection(baseline_ref=scan.git.branch or "main"),
        execution=ExecutionSection(
            setup_command=setup_command,
            screening_command=screening_command,
            full_command=full_command,
            trusted_repository=True,
        ),
        permissions=PermissionsSection(
            editable_paths=list(scan.suggested_editable_paths) or ["src/"],
            protected_paths=list(scan.suggested_protected_paths),
        ),
    )
