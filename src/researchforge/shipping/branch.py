"""Ship a validated experiment as a clean branch (spec: repository outcome).

The branch is reconstructed from the frozen baseline commit plus the winning
experiment's immutable patch — one commit, no experiment history. The user's
checkout is never touched: the commit is built in a temporary detached
worktree and materialized as a ref with `git branch`. Nothing is ever pushed
by this module.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from researchforge import __version__
from researchforge.config.paths import experiment_artifacts_dir
from researchforge.domain.contract import MetricDirection
from researchforge.domain.deliverable import Deliverable, DeliverableKind
from researchforge.domain.environment import DockerProbe
from researchforge.domain.experiment import (
    BenchmarkStage,
    ExecutionRecordStatus,
    Experiment,
    ExperimentExecution,
    ExperimentRunGroup,
    ExperimentStatus,
    ValidationSummary,
    advance,
)
from researchforge.domain.hypothesis import Hypothesis
from researchforge.execution.constraints import constraints_ok
from researchforge.execution.experiments import (
    ExperimentBlockedError,
    RunPreparation,
    execute_stage,
    prepare_run,
)
from researchforge.execution.runner import CommandRunner, SubprocessRunner
from researchforge.execution.validation import summarize_validation
from researchforge.execution.worktrees import WorktreeManager
from researchforge.experiments.graph import ancestor_order
from researchforge.storage.deliverable_repository import (
    get_branch_deliverable,
    insert_deliverable,
)
from researchforge.storage.experiment_repository import (
    get_experiment,
    list_executions,
    list_experiments,
    list_runs,
    patch_ancestry,
    update_experiment,
)
from researchforge.storage.hypothesis_repository import get_hypothesis

BRANCH_PREFIX = "researchforge/"
MAX_SLUG_LENGTH = 40


class ShipBlockedError(Exception):
    """Shipping cannot proceed; message is user-facing."""


@dataclass
class ShipPreparation:
    prep: RunPreparation
    experiment: Experiment
    hypothesis: Hypothesis
    run: ExperimentRunGroup
    validation_summary: ValidationSummary | None
    ancestor_patches: list[str] = field(default_factory=list)  # root-first chain


class ShipResult(BaseModel):
    experiment_id: str
    branch: str
    commit_sha: str
    baseline_commit: str
    preship_execution_id: str
    preship_primary_value: float
    changed_files: list[str]
    deliverable_id: str
    next_action: str


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:MAX_SLUG_LENGTH].rstrip("-") or "experiment"


def derive_branch_name(
    title: str, taken: Callable[[str], bool], override: str | None = None
) -> str:
    """Branch name from the hypothesis title, avoiding existing refs."""
    if override is not None:
        if taken(override):
            raise ShipBlockedError(f"Branch {override!r} already exists — choose another name.")
        return override
    base = f"{BRANCH_PREFIX}{_slugify(title)}"
    candidate = base
    suffix = 2
    while taken(candidate):
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _tests_line(test_command: str | None) -> str:
    if test_command:
        return (
            f"existing tests executed via contract test_command ({test_command}); "
            "no new tests authored (Claude-assisted test authoring arrives in Phase 1E)"
        )
    return (
        "no test command configured in the contract; no new tests authored "
        "(Claude-assisted test authoring arrives in Phase 1E)"
    )


def build_commit_message(ship: ShipPreparation, preship: ExperimentExecution) -> str:
    """Explanatory commit built entirely from stored records."""
    spec = ship.prep.contract.spec
    primary = spec.objective.primary_metric.name
    baseline = ship.prep.baseline
    assert baseline.metrics is not None
    assert preship.metrics is not None

    subject = f"{spec.objective.description.strip().rstrip('.')} ({ship.experiment.experiment_id})"
    if len(subject) > 72:
        subject = subject[:69] + "..."

    validated_line = "single confirmation run"
    if ship.validation_summary is not None and ship.validation_summary.mean is not None:
        summary = ship.validation_summary
        stdev = f"{summary.stdev:.4g}" if summary.stdev is not None else "n/a"
        validated_line = (
            f"{primary} {baseline.metrics.primary_metric.value} -> mean {summary.mean:.4g} "
            f"(n={summary.attempts}, stdev={stdev})"
        )

    constraint_parts = [
        f"{c.name} {c.operator.value} {c.threshold}: {'pass' if c.passed else 'FAIL'}"
        for c in preship.constraints
    ]
    lines = [
        subject,
        "",
        f"Hypothesis:  {ship.hypothesis.hypothesis_id} — {ship.hypothesis.title}",
        f"Experiment:  {ship.experiment.experiment_id} — {ship.experiment.title}: "
        f"{ship.experiment.change_summary}",
        f"Contract:    {ship.prep.contract.contract_id} v{ship.prep.contract.contract_version}",
        f"Baseline:    {ship.prep.plan.baseline_commit[:12]} "
        f"({primary}={baseline.metrics.primary_metric.value})",
        f"Validated:   {validated_line}",
        f"Pre-ship:    {primary}={preship.metrics.primary_metric.value}"
        + (f"; constraints: {', '.join(constraint_parts)}" if constraint_parts else ""),
        f"Tests:       {_tests_line(spec.execution.test_command)}",
        f"Changed:     {', '.join(ship.experiment.changed_files)}",
        "",
        f"Generated by ResearchForge {__version__} from experiment records.",
    ]
    return "\n".join(lines) + "\n"


def prepare_ship(
    conn: sqlite3.Connection,
    experiment_id: str | None,
    docker: DockerProbe | None = None,
) -> ShipPreparation:
    """Resolve the winner and run every shipping gate."""
    if experiment_id is not None:
        experiment = get_experiment(conn, experiment_id)
        if experiment is None:
            raise ShipBlockedError(f"Unknown experiment: {experiment_id}.")
    else:
        validated = [e for e in list_experiments(conn) if e.status is ExperimentStatus.VALIDATED]
        if not validated:
            shipped = [
                e
                for e in list_experiments(conn)
                if e.status is ExperimentStatus.IMPLEMENTATION_READY
            ]
            if shipped:
                deliverable = get_branch_deliverable(conn, shipped[-1].experiment_id)
                location = deliverable.location if deliverable else "unknown"
                raise ShipBlockedError(
                    f"{shipped[-1].experiment_id} was already shipped as branch "
                    f"{location!r}. Validate another experiment to ship again."
                )
            raise ShipBlockedError(
                "No validated experiment to ship — run `researchforge validate <run-id>` first."
            )
        if len(validated) > 1:
            ids = ", ".join(e.experiment_id for e in validated)
            raise ShipBlockedError(
                f"Multiple validated experiments ({ids}) — pass one explicitly: "
                "`researchforge ship branch <experiment-id>`."
            )
        experiment = validated[0]

    if experiment.status is ExperimentStatus.IMPLEMENTATION_READY:
        deliverable = get_branch_deliverable(conn, experiment.experiment_id)
        location = deliverable.location if deliverable else "unknown"
        raise ShipBlockedError(
            f"{experiment.experiment_id} was already shipped as branch {location!r}."
        )
    if experiment.status is not ExperimentStatus.VALIDATED:
        raise ShipBlockedError(
            f"{experiment.experiment_id} is {experiment.status.value} — only validated "
            "experiments can be shipped."
        )

    try:
        prep = prepare_run(conn, experiment.plan_id, docker, allow_completed=True)
    except ExperimentBlockedError as exc:
        raise ShipBlockedError(str(exc)) from None

    if not prep.contract.spec.shipping.allow_branch_creation:
        raise ShipBlockedError(
            "shipping.allow_branch_creation is false in the approved contract — edit "
            "researchforge.yaml and re-approve to allow branch creation."
        )

    hypothesis = get_hypothesis(conn, experiment.hypothesis_id)
    if hypothesis is None:
        raise ShipBlockedError(f"Hypothesis {experiment.hypothesis_id} not found.")

    runs = [r for r in list_runs(conn) if r.plan_id == experiment.plan_id]
    if not runs:
        raise ShipBlockedError(f"No run group found for {experiment.plan_id}.")
    run = runs[-1]

    validation_attempts = [
        e
        for e in list_executions(conn, run_id=run.run_id, experiment_id=experiment.experiment_id)
        if e.benchmark_stage is BenchmarkStage.VALIDATION
    ]
    validation_summary = None
    if validation_attempts:
        validation_summary = summarize_validation(
            experiment,
            validation_attempts,
            prep.baseline,
            prep.contract.spec.objective.primary_metric.direction,
        )

    # The shipped branch is the whole lineage, not just this experiment's own
    # diff: everything it was measured on top of has to land with it. A merge
    # has several parents, so the order comes from a topological walk and a
    # shared ancestor contributes once. A re-authored merge ships as its own
    # patch alone, which is what was measured.
    ancestor_patches: list[str] = []
    for ancestor_id in ancestor_order(experiment.experiment_id, patch_ancestry(conn)):
        ancestor_experiment = get_experiment(conn, ancestor_id)
        assert ancestor_experiment is not None  # import validated the lineage
        if ancestor_experiment.patch_text:
            ancestor_patches.append(ancestor_experiment.patch_text)

    return ShipPreparation(
        prep=prep,
        experiment=experiment,
        hypothesis=hypothesis,
        run=run,
        validation_summary=validation_summary,
        ancestor_patches=ancestor_patches,
    )


def run_preship_validation(
    conn: sqlite3.Connection, ship: ShipPreparation, runner: CommandRunner
) -> ExperimentExecution:
    """One fresh confirmation run of the full benchmark before shipping."""
    attempts = [
        e.attempt
        for e in list_executions(
            conn, run_id=ship.run.run_id, experiment_id=ship.experiment.experiment_id
        )
        if e.benchmark_stage is BenchmarkStage.VALIDATION
    ]
    attempt = max(attempts, default=0) + 1
    execution = execute_stage(
        conn,
        ship.prep,
        ship.run,
        ship.experiment,
        BenchmarkStage.VALIDATION,
        attempt,
        runner,
        extra_env={"RF_PRESHIP": "1", "RF_VALIDATION_ATTEMPT": str(attempt)},
    )
    if execution.status is not ExecutionRecordStatus.SUCCEEDED:
        raise ShipBlockedError(
            f"Pre-ship confirmation failed ({execution.status.value}: "
            f"{execution.failure_reason}) — not shipping. Artifacts: "
            f"{execution.artifacts.stdout_path.rsplit('/', 1)[0]}"
        )
    if not constraints_ok(execution.constraints):
        raise ShipBlockedError("Pre-ship confirmation violated a hard constraint — not shipping.")
    assert execution.metrics is not None
    baseline_metrics = ship.prep.baseline.metrics
    assert baseline_metrics is not None
    direction = ship.prep.contract.spec.objective.primary_metric.direction
    base_value = baseline_metrics.primary_metric.value
    value = execution.metrics.primary_metric.value
    improved = value > base_value if direction is MetricDirection.MAXIMIZE else value < base_value
    if not improved:
        raise ShipBlockedError(
            f"Pre-ship confirmation did not improve on the baseline "
            f"({base_value} -> {value}) — not shipping."
        )
    return execution


def reconstruct_branch(
    manager: WorktreeManager, ship: ShipPreparation, branch: str, message: str
) -> str:
    """Build the single clean commit and materialize it as a branch ref."""
    worktree_name = f"ship-{ship.experiment.experiment_id}"
    ship_artifacts = (
        experiment_artifacts_dir(ship.prep.repo_root) / "ship" / ship.experiment.experiment_id
    )
    ship_artifacts.mkdir(parents=True, exist_ok=True)
    patch_file = ship_artifacts / "change.patch"
    patch_file.write_text(ship.experiment.patch_text, encoding="utf-8")
    message_file = ship_artifacts / "commit_message.txt"
    message_file.write_text(message, encoding="utf-8")

    # Branched winners ship their full ancestor chain, roots first.
    chain_files: list[Path] = []
    for depth, ancestor_text in enumerate(ship.ancestor_patches):
        chain_file = ship_artifacts / f"chain-{depth}.patch"
        chain_file.write_text(ancestor_text, encoding="utf-8")
        chain_files.append(chain_file)

    worktree = manager.create(worktree_name, ship.prep.plan.baseline_commit, recreate=True)
    try:
        for chain_file in chain_files:
            manager.apply_patch(worktree, chain_file)
        # A merge or env-override winner has no diff of its own: the chain above
        # already is the change, and git refuses an empty patch.
        if ship.experiment.patch_text:
            manager.apply_patch(worktree, patch_file)
        sha = manager.commit_all_in_worktree(worktree, message_file)
    finally:
        manager.remove(worktree_name)

    manager.create_branch(branch, sha)

    # Post-conditions: single clean commit on the frozen baseline with exactly
    # the validated change. Violation means a bug — remove the ref and abort.
    parent = manager.parent_of(sha)
    diff = sorted(manager.diff_names(ship.prep.plan.baseline_commit, sha))
    if parent != ship.prep.plan.baseline_commit or diff != sorted(ship.experiment.changed_files):
        manager.delete_branch(branch)
        raise ShipBlockedError(
            "Reconstructed branch did not match the validated change "
            f"(parent {parent[:12]}, files {diff}) — branch removed, nothing shipped."
        )
    return sha


def ship_branch(
    conn: sqlite3.Connection,
    experiment_id: str | None = None,
    *,
    branch: str | None = None,
    runner: CommandRunner | None = None,
    docker: DockerProbe | None = None,
) -> ShipResult:
    ship = prepare_ship(conn, experiment_id, docker)
    manager = WorktreeManager(ship.prep.repo_root)

    if branch is not None and not manager.check_branch_name(branch):
        raise ShipBlockedError(f"{branch!r} is not a valid git branch name.")
    branch_name = derive_branch_name(ship.hypothesis.title, manager.branch_exists, override=branch)

    active_runner: CommandRunner = runner or SubprocessRunner()
    preship = run_preship_validation(conn, ship, active_runner)

    message = build_commit_message(ship, preship)
    sha = reconstruct_branch(manager, ship, branch_name, message)

    updated = ship.experiment.model_copy(
        update={
            "status": advance(ship.experiment.status, ExperimentStatus.IMPLEMENTATION_READY),
        }
    )
    update_experiment(conn, updated)

    deliverable = Deliverable(
        deliverable_id=uuid4().hex,
        kind=DeliverableKind.BRANCH,
        experiment_id=ship.experiment.experiment_id,
        location=branch_name,
        commit_sha=sha,
        details={"baseline_commit": ship.prep.plan.baseline_commit},
        created_at=datetime.now(UTC),
    )
    insert_deliverable(conn, ship.prep.project.id, deliverable)

    assert preship.metrics is not None
    return ShipResult(
        experiment_id=ship.experiment.experiment_id,
        branch=branch_name,
        commit_sha=sha,
        baseline_commit=ship.prep.plan.baseline_commit,
        preship_execution_id=preship.execution_id,
        preship_primary_value=preship.metrics.primary_metric.value,
        changed_files=ship.experiment.changed_files,
        deliverable_id=deliverable.deliverable_id,
        next_action="researchforge report build  # then optionally: researchforge ship pr",
    )
