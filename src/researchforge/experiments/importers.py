"""Experiment-plan import: validation and persistence (Claude -> CLI).

The enforcement boundary for experiment variants. Mechanical problems
(schema errors, missing/oversized/binary patches, patches that don't apply)
are import errors the author fixes and retries. A patch that touches
protected or non-editable paths is NOT an error: the experiment is persisted
as `rejected` with its violations — a first-class negative result that will
never run (spec: protected-path modification rejected before evaluation).
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from researchforge.config.paths import contract_path, experiment_artifacts_dir, experiments_dir
from researchforge.contract.service import check_contract_drift
from researchforge.domain.experiment import (
    Decision,
    DecisionOutcome,
    Experiment,
    ExperimentPlan,
    ExperimentStatus,
)
from researchforge.execution.baseline import BaselineBlockedError, baseline_gate
from researchforge.execution.path_guard import check_changed_paths
from researchforge.execution.worktrees import WorktreeError, WorktreeManager
from researchforge.experiments.context_export import (
    PATCHES_DIR_NAME,
    ExperimentPlanArtifact,
    PlannedExperimentEntry,
)
from researchforge.experiments.graph import GraphCycleError, ancestor_order
from researchforge.storage.contract_repository import get_active_contract
from researchforge.storage.experiment_repository import (
    get_experiment,
    insert_plan,
    next_experiment_ids,
    next_plan_id,
)
from researchforge.storage.hypothesis_repository import get_hypothesis
from researchforge.storage.project_repository import get_project
from researchforge.utils.artifact_io import ArtifactLoadError, load_artifact

MAX_PATCH_BYTES = 512_000
PLAN_CHECK_WORKTREE = "plan-check"


@dataclass
class ImportResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _format_validation_error(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location or '<root>'}: {error['msg']}")
    return messages


def _patch_candidate(patch_file: str, base: Path | None) -> Path:
    """Where `patch_file` points, resolved against the experiments directory.

    A bare file name is taken to mean the patches directory, since that is the
    only place a patch may live and an author who wrote `foo.patch` meant
    `patches/foo.patch`. Anything with a directory component is resolved as
    written, so an attempt to escape the tree is still caught by the caller.
    """
    root = experiments_dir(base)
    if Path(patch_file).parent == Path("."):
        return (root / PATCHES_DIR_NAME / patch_file).resolve()
    return (root / patch_file).resolve()


def _load_patch(
    entry_key: str, patch_file: str, base: Path | None, errors: list[str]
) -> Path | None:
    """Resolve and sanity-check a patch file; returns its path or records errors."""
    patches_root = (experiments_dir(base) / PATCHES_DIR_NAME).resolve()
    candidate = _patch_candidate(patch_file, base)
    where = f"experiments.{entry_key}.patch_file"
    if not candidate.is_relative_to(patches_root):
        errors.append(f"{where}: must live inside {patches_root} (got {patch_file!r}).")
        return None
    if not candidate.is_file():
        errors.append(f"{where}: file not found at {candidate}.")
        return None
    if candidate.stat().st_size > MAX_PATCH_BYTES:
        errors.append(f"{where}: exceeds {MAX_PATCH_BYTES} bytes.")
        return None
    raw = candidate.read_bytes()
    if b"\0" in raw:
        errors.append(f"{where}: contains NUL bytes; only text diffs are supported.")
        return None
    if b"GIT binary patch" in raw:
        errors.append(f"{where}: binary patches are not supported.")
        return None
    return candidate


_BRANCHABLE_STATUSES = frozenset(
    {
        ExperimentStatus.PROMISING,
        ExperimentStatus.REJECTED,
        ExperimentStatus.VALIDATED,
        ExperimentStatus.IMPLEMENTATION_READY,
    }
)

MERGE_CONFLICT_PREFIX = "merge conflict"


@dataclass
class ParentResolution:
    """Per-entry ancestry, resolved and validated before anything runs."""

    chains: dict[str, list[str]] = field(default_factory=dict)
    """Ancestor patch texts in dependency order (roots first), own patch excluded."""

    declared: dict[str, list[str]] = field(default_factory=dict)
    """Ancestor references as the author wrote them: plan keys or exp-NNN ids."""


def _read_entry_patch(entry: PlannedExperimentEntry, base: Path | None) -> str:
    """An entry's own patch text, or "" when it has none.

    Patch files are validated properly in Layer 3; here we only need the text
    for chain composition, so a missing or out-of-tree file reads as empty and
    the later layer reports it.
    """
    if not entry.patch_file:
        return ""
    patches_root = (experiments_dir(base) / PATCHES_DIR_NAME).resolve()
    candidate = _patch_candidate(entry.patch_file, base)
    if not candidate.is_relative_to(patches_root):
        return ""
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError:
        return ""


def _resolve_parents(
    conn: sqlite3.Connection,
    artifact: ExperimentPlanArtifact,
    result: ImportResult,
    base: Path | None = None,
) -> ParentResolution:
    """Validate every declared ancestor and order the patches they contribute.

    Ancestors may be keys in this same plan or `exp-NNN` ids already stored, and
    an entry may declare several of them (a merge).  Stored ancestors must be
    measured experiments: a plan cannot branch from something that never ran.

    A node marked `patch_includes_parents` contributes no chain: its own patch
    is the combined state, so its parents are lineage rather than something to
    apply.  Chains stop there for its descendants as well.
    """
    entries = {entry.key: entry for entry in artifact.experiments}
    resolution = ParentResolution()
    stored: dict[str, Experiment] = {}

    def stored_experiment(experiment_id: str, where: str) -> Experiment | None:
        if experiment_id in stored:
            return stored[experiment_id]
        experiment = get_experiment(conn, experiment_id)
        if experiment is None:
            result.errors.append(f"{where}: unknown parent experiment {experiment_id!r}.")
            return None
        if experiment.status not in _BRANCHABLE_STATUSES:
            result.errors.append(
                f"{where}: parent {experiment_id} is {experiment.status.value} — only "
                "measured experiments (promising/rejected/validated/"
                "implementation_ready) can be branched on."
            )
            return None
        stored[experiment_id] = experiment
        return experiment

    def parents_of(node: str) -> list[str]:
        entry = entries.get(node)
        if entry is not None:
            return [] if entry.patch_includes_parents else entry.declared_parents
        experiment = stored.get(node)
        if experiment is None or experiment.patch_includes_parents:
            return []
        return list(experiment.parent_experiment_ids)

    def patch_text_of(node: str) -> str:
        entry = entries.get(node)
        if entry is not None:
            return _read_entry_patch(entry, base)
        experiment = stored.get(node)
        return experiment.patch_text if experiment else ""

    # Load the stored side of the graph first, following each declared ancestor
    # all the way to its roots: an ancestor's own ancestors contribute patches
    # too, so a lineage that is only half-loaded would compose a partial state.
    for entry in artifact.experiments:
        where = f"experiments.{entry.key}.parent"
        pending = [ref for ref in entry.declared_parents if ref not in entries]
        seen: set[str] = set()
        while pending:
            reference = pending.pop()
            if reference in stored or reference in seen:
                continue
            seen.add(reference)
            experiment = stored_experiment(reference, where)
            if experiment is not None:
                pending.extend(experiment.parent_experiment_ids)
    if result.errors:
        return resolution

    for entry in artifact.experiments:
        resolution.declared[entry.key] = entry.declared_parents
        try:
            ancestors = ancestor_order(entry.key, parents_of)
        except GraphCycleError as exc:
            result.errors.append(f"experiments.{entry.key}: parent graph has a cycle ({exc}).")
            continue
        resolution.chains[entry.key] = [
            text for text in (patch_text_of(node) for node in ancestors) if text
        ]
    return resolution


def _check_patches(
    repo_root: Path,
    baseline_commit: str,
    artifact: ExperimentPlanArtifact,
    patch_paths: dict[str, Path],
    parents: ParentResolution,
    result: ImportResult,
) -> dict[str, list[str]]:
    """Apply every entry in a scratch worktree; return its changed files.

    Each entry that has ancestors is checked on a freshly composed worktree so
    conflicts between merged branches surface here rather than mid-run.
    """
    needs_check = [
        entry for entry in artifact.experiments if entry.patch_file or parents.chains.get(entry.key)
    ]
    if not needs_check:
        return {}

    manager = WorktreeManager(repo_root)
    changed_by_key: dict[str, list[str]] = {}
    try:
        scratch = manager.create(PLAN_CHECK_WORKTREE, baseline_commit, recreate=True)
        for entry in needs_check:
            chain = parents.chains.get(entry.key, [])
            if chain:
                scratch = manager.create(PLAN_CHECK_WORKTREE, baseline_commit, recreate=True)
            chain_files = _apply_chain(manager, scratch, entry.key, chain, result)
            if chain_files is None:
                continue

            patch = patch_paths.get(entry.key)
            if patch is None:
                changed_by_key[entry.key] = sorted(set(chain_files))
                continue

            applies, message = manager.apply_patch_check(scratch, patch)
            if not applies:
                where = (
                    "on top of its ancestor chain"
                    if chain
                    else (f"at baseline {baseline_commit[:12]}")
                )
                result.errors.append(
                    f"experiments.{entry.key}: patch does not apply {where} — {message}"
                )
                continue
            own_files = manager.patch_numstat(scratch, patch)
            changed_by_key[entry.key] = sorted(set(chain_files) | set(own_files))
    except WorktreeError as exc:
        result.errors.append(f"Could not prepare the patch-check worktree: {exc}")
    finally:
        with contextlib.suppress(WorktreeError):
            manager.remove(PLAN_CHECK_WORKTREE)
    return changed_by_key


def _apply_chain(
    manager: WorktreeManager,
    scratch: Path,
    key: str,
    chain: list[str],
    result: ImportResult,
) -> list[str] | None:
    """Apply ancestor patches in order; None means the chain does not compose."""
    chain_files: list[str] = []
    for depth, ancestor_text in enumerate(chain):
        ancestor_patch = scratch / f".rf-ancestor-{depth}.patch"
        ancestor_patch.write_text(ancestor_text, encoding="utf-8")
        try:
            chain_files.extend(manager.patch_numstat(scratch, ancestor_patch))
            manager.apply_patch(scratch, ancestor_patch)
        except WorktreeError as exc:
            detail = (
                f"{MERGE_CONFLICT_PREFIX}: ancestor patch #{depth + 1} of "
                f"{len(chain)} does not apply on the combined state — {exc}"
                if len(chain) > 1
                else f"ancestor patch no longer applies — {exc}"
            )
            result.errors.append(f"experiments.{key}: {detail}")
            return None
        finally:
            ancestor_patch.unlink(missing_ok=True)
    return chain_files


def import_experiment_plan(
    conn: sqlite3.Connection, path: Path, *, base: Path | None = None
) -> tuple[ImportResult, ExperimentPlan | None]:
    result = ImportResult()

    # Layer 1: parse + schema.
    try:
        raw = load_artifact(path)
    except ArtifactLoadError as exc:
        result.errors.append(str(exc))
        return result, None
    try:
        artifact = ExperimentPlanArtifact.model_validate(raw)
    except ValidationError as exc:
        result.errors.extend(_format_validation_error(exc))
        return result, None

    # Layer 2: gates.
    project = get_project(conn)
    if project is None:
        result.errors.append("No project found. Run `researchforge project create` first.")
        return result, None
    contract = get_active_contract(conn)
    if contract is None:
        result.errors.append("No approved contract. Run `researchforge contract approve`.")
        return result, None
    repo_root = Path(project.repository.path) if project.repository.path else Path.cwd()
    if check_contract_drift(conn, contract_path(repo_root)):
        result.errors.append(
            "researchforge.yaml changed since approval — re-approve before planning experiments."
        )
        return result, None
    try:
        baseline = baseline_gate(conn)
    except BaselineBlockedError as exc:
        result.errors.append(str(exc))
        return result, None
    hypothesis = get_hypothesis(conn, artifact.hypothesis_id)
    if hypothesis is None:
        result.errors.append(
            f"hypothesis_id: unknown hypothesis {artifact.hypothesis_id!r} — "
            "see `researchforge hypotheses list`."
        )
        return result, None
    if not hypothesis.is_plannable:
        result.errors.append(
            f"hypothesis_id: {artifact.hypothesis_id} was rejected in review — run "
            f"`researchforge hypotheses approve {artifact.hypothesis_id}` to plan it."
        )
        return result, None

    max_experiments = contract.spec.execution.max_experiments
    if len(artifact.experiments) > max_experiments:
        result.errors.append(
            f"experiments: {len(artifact.experiments)} provided; the contract allows "
            f"at most {max_experiments} (execution.max_experiments)."
        )
    keys = [entry.key for entry in artifact.experiments]
    if len(keys) != len(set(keys)):
        result.errors.append("experiments: duplicate keys.")
    if result.errors:
        return result, None

    # Layer 2b: resolve `parent:` / `parents:` references (same-plan keys or
    # stored exp-NNN ids) into ordered ancestor patch chains, refusing cycles
    # and unmeasured parents.
    parents = _resolve_parents(conn, artifact, result, base)
    if result.errors:
        return result, None

    # Layer 3: patch files (skipped for env-only entries and pure merges).
    patch_paths: dict[str, Path] = {}
    for entry in artifact.experiments:
        has_patch = bool(entry.patch_file)
        has_env = bool(entry.env_overrides)
        is_merge = len(parents.declared.get(entry.key, [])) > 1
        if has_patch and has_env:
            result.errors.append(
                f"experiments.{entry.key}: set either patch_file OR env_overrides, not both."
            )
            continue
        if not has_patch and not has_env and not is_merge:
            result.errors.append(
                f"experiments.{entry.key}: must provide either patch_file or env_overrides "
                "(only a merge of several parents may have neither)."
            )
            continue
        if has_patch:
            candidate = _load_patch(entry.key, entry.patch_file, base, result.errors)  # type: ignore[arg-type]
            if candidate is not None:
                patch_paths[entry.key] = candidate
        # env-only: no patch file to load
    if result.errors:
        return result, None

    # Layer 4: apply-check + changed-path extraction in a scratch worktree.
    # An entry with ancestors is checked in a worktree where the whole ancestor
    # chain has actually been applied, so the diff is verified against the state
    # it was written for and a merge that cannot compose is caught before
    # anything runs. Entries with neither a patch nor ancestors change no files.
    changed_by_key = _check_patches(
        repo_root, baseline.commit_sha, artifact, patch_paths, parents, result
    )
    if result.errors:
        return result, None

    # Layer 5: path guard per experiment (violations => rejected record, not error).
    now = datetime.now(UTC)
    plan_id = next_plan_id(conn)
    experiment_ids = next_experiment_ids(conn, len(artifact.experiments))
    experiments: list[Experiment] = []
    seen_hashes: dict[str, str] = {}
    runnable = 0
    id_by_key = {
        entry.key: experiment_id
        for entry, experiment_id in zip(artifact.experiments, experiment_ids, strict=True)
    }
    for entry, experiment_id in zip(artifact.experiments, experiment_ids, strict=True):
        # No own patch: an env-override variant, or a merge whose whole change
        # is its ancestors' patches. Neither adds file changes of its own, so
        # the path guard has nothing new to judge — the ancestors were guarded
        # when they were imported.
        is_patchless = not entry.patch_file
        if is_patchless:
            patch_text = ""
            digest = hashlib.sha256(b"").hexdigest()
            changed = changed_by_key.get(entry.key, [])
            status = ExperimentStatus.PLANNED
            decision = None
            runnable += 1
        else:
            patch_text = patch_paths[entry.key].read_text(encoding="utf-8")
            digest = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                result.warnings.append(
                    f"experiments.{entry.key}: patch is identical to "
                    f"{seen_hashes[digest]} — duplicate variant?"
                )
            seen_hashes.setdefault(digest, entry.key)

            changed = changed_by_key.get(entry.key, [])
            guard = check_changed_paths(changed, contract.spec.permissions)
            status = ExperimentStatus.PLANNED
            decision = None
            if not guard.allowed:
                status = ExperimentStatus.REJECTED
                details = ", ".join(
                    f"{violation.path} ({violation.rule.value})" for violation in guard.violations
                )
                decision = Decision(
                    outcome=DecisionOutcome.REJECT,
                    reason=f"changes protected or non-editable paths: {details}",
                )
                result.warnings.append(
                    f"experiments.{entry.key} ({experiment_id}): {decision.reason} — "
                    "recorded as rejected; it will not run."
                )
            else:
                runnable += 1

        experiments.append(
            Experiment(
                experiment_id=experiment_id,
                plan_id=plan_id,
                hypothesis_id=artifact.hypothesis_id,
                parent_experiment_ids=[
                    id_by_key.get(reference, reference)
                    for reference in parents.declared.get(entry.key, [])
                ],
                patch_includes_parents=entry.patch_includes_parents,
                title=entry.title,
                change_summary=entry.change_summary,
                patch_text=patch_text,
                patch_sha256=digest,
                changed_files=changed,
                env_overrides=dict(entry.env_overrides or {}),
                path_violations=guard.violations if not is_patchless else [],
                expected_effect=entry.expected_effect,
                status=status,
                decision=decision,
                created_at=now,
                updated_at=now,
            )
        )

    if runnable == 0:
        result.errors.append(
            "experiments: every variant was rejected by the path guard — nothing to run. "
            "Author changes inside editable_paths only."
        )
        return result, None

    # Layer 6: transactional persist + patch copies into artifacts.
    plan = ExperimentPlan(
        plan_id=plan_id,
        hypothesis_id=artifact.hypothesis_id,
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        baseline_id=baseline.baseline_id,
        baseline_commit=baseline.commit_sha,
        approach_summary=artifact.approach_summary,
        source_file=str(path),
        created_at=now,
        updated_at=now,
    )
    insert_plan(conn, project.id, plan, experiments)
    for experiment in experiments:
        target_dir = experiment_artifacts_dir(repo_root) / plan_id / experiment.experiment_id
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "change.patch").write_text(experiment.patch_text, encoding="utf-8")
    return result, plan
