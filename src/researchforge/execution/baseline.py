"""Baseline orchestration: worktree -> environment -> run -> parse -> persist.

Execution always uses the STORED approved contract snapshot, never the disk
yaml (which may have drifted). A failed baseline is persisted and blocks
experimentation via `baseline_gate`.
"""

from __future__ import annotations

import math
import platform as platform_module
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from researchforge.config.paths import artifacts_dir, contract_path
from researchforge.contract.service import check_contract_drift
from researchforge.domain.baseline import (
    BaselineRepeats,
    BaselineRun,
    BaselineStatus,
    EnvironmentFingerprint,
)
from researchforge.domain.contract import ExperimentContract
from researchforge.domain.environment import DockerProbe, EnvironmentResolution
from researchforge.domain.project import Project, ProjectStatus
from researchforge.domain.repo_scan import CompatibilityStatus
from researchforge.execution import venv_exec
from researchforge.execution.environment import probe_docker, resolve_environment
from researchforge.execution.evaluation import EvaluationStatus, run_evaluation
from researchforge.execution.metrics import MetricResult
from researchforge.execution.runner import CommandRunner, SubprocessRunner
from researchforge.execution.worktrees import WorktreeManager
from researchforge.project.service import touch_project_status
from researchforge.storage.baseline_repository import (
    get_latest_baseline,
    insert_baseline_run,
)
from researchforge.storage.contract_repository import get_active_contract
from researchforge.storage.project_repository import get_project
from researchforge.storage.scan_repository import get_latest_scan

BASELINE_WORKTREE = "baseline"

_EVALUATION_TO_BASELINE_STATUS: dict[EvaluationStatus, BaselineStatus] = {
    EvaluationStatus.SUCCEEDED: BaselineStatus.SUCCEEDED,
    EvaluationStatus.FAILED_SETUP: BaselineStatus.FAILED_SETUP,
    # Baselines pass test_command=None, so FAILED_TESTS is defensive only.
    EvaluationStatus.FAILED_TESTS: BaselineStatus.FAILED_EXECUTION,
    EvaluationStatus.FAILED_EXECUTION: BaselineStatus.FAILED_EXECUTION,
    EvaluationStatus.FAILED_TIMEOUT: BaselineStatus.FAILED_TIMEOUT,
    EvaluationStatus.FAILED_INVALID_RESULT: BaselineStatus.FAILED_INVALID_RESULT,
}


class BaselineBlockedError(Exception):
    """Baseline cannot run (or has not succeeded); message is user-facing."""

    def __init__(self, reason: str, resolution: EnvironmentResolution | None = None) -> None:
        self.reason = reason
        self.resolution = resolution
        super().__init__(reason)


@dataclass
class BaselinePreparation:
    project: Project
    contract: ExperimentContract
    repo_root: Path
    resolution: EnvironmentResolution


def prepare_baseline(
    conn: sqlite3.Connection, docker: DockerProbe | None = None
) -> BaselinePreparation:
    """Load contract + scan, check drift, resolve the environment."""
    project = get_project(conn)
    if project is None:
        raise BaselineBlockedError("No project found. Run `researchforge project create` first.")
    contract = get_active_contract(conn)
    if contract is None:
        raise BaselineBlockedError(
            "No approved contract. Run `researchforge contract approve` first."
        )
    repo_root = Path(project.repository.path) if project.repository.path else Path.cwd()
    if check_contract_drift(conn, contract_path(repo_root)):
        raise BaselineBlockedError(
            "researchforge.yaml changed since approval — run `researchforge contract "
            f"approve` to create contract version {contract.contract_version + 1}."
        )
    scan = get_latest_scan(conn)
    if scan is None:
        raise BaselineBlockedError("No repository scan. Run `researchforge repo scan` first.")

    resolution = resolve_environment(contract.spec, scan, docker or probe_docker())
    return BaselinePreparation(
        project=project, contract=contract, repo_root=repo_root, resolution=resolution
    )


def _redact_logs(paths: list[Path], secrets: dict[str, str]) -> None:
    """Replace forwarded secret values with <redacted:NAME> in persisted logs."""
    if not secrets:
        return
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, value in secrets.items():
            if value:
                text = text.replace(value, f"<redacted:{name}>")
        path.write_text(text, encoding="utf-8")


def summarize_repeats(
    requested: int, values: list[float], failed: int
) -> BaselineRepeats | None:
    """The mean and spread of the repeats that produced a number.

    Returns None for a single measurement: calling one run a "mean of 1" adds a
    statistic that says nothing, and the absence of the field is what tells a
    reader this baseline was measured once.
    """
    if requested <= 1 or not values:
        return None
    mean = sum(values) / len(values)
    stdev = None
    if len(values) >= 2:
        stdev = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return BaselineRepeats(
        requested=requested,
        values=values,
        failed=failed,
        mean=mean,
        stdev=stdev,
        coefficient_of_variation=(stdev / abs(mean) if stdev is not None and mean != 0 else None),
    )


def averaged_metrics(outcomes: list[MetricResult]) -> MetricResult:
    """One MetricResult holding the mean of each metric every repeat reported.

    A secondary metric missing from any repeat is dropped rather than averaged
    over the subset that has it, which would be a mean of a different thing than
    the label claims.
    """
    first = outcomes[0]
    shared = set(first.secondary_metrics)
    for outcome in outcomes[1:]:
        shared &= set(outcome.secondary_metrics)
    return first.model_copy(
        update={
            "primary_metric": first.primary_metric.model_copy(
                update={
                    "value": sum(o.primary_metric.value for o in outcomes) / len(outcomes)
                }
            ),
            "secondary_metrics": {
                name: sum(o.secondary_metrics[name] for o in outcomes) / len(outcomes)
                for name in sorted(shared)
            },
        }
    )


def run_baseline(
    conn: sqlite3.Connection,
    *,
    runner: CommandRunner | None = None,
    docker: DockerProbe | None = None,
    on_phase: Callable[[str], None] | None = None,
    n_runs: int = 1,
) -> BaselineRun:
    """Run the baseline; persists a BaselineRun for every terminal status.

    `n_runs` above 1 measures the same commit repeatedly and stores the mean as
    the baseline, with the individual values kept in `repeats`. This is one
    baseline made of several executions, not several baselines: experiments
    compare against a single reference point, and averaging is only meaningful
    if that reference is the average.
    """
    prep = prepare_baseline(conn, docker)
    if prep.resolution.status is not CompatibilityStatus.READY:
        raise BaselineBlockedError("The repository is not ready for execution.", prep.resolution)

    engine = prep.resolution.execution_mode
    spec = prep.contract.spec
    active_runner: CommandRunner = runner or SubprocessRunner()

    if on_phase:
        on_phase("Creating isolated worktree…")
    manager = WorktreeManager(prep.repo_root)
    worktree = manager.create(BASELINE_WORKTREE, spec.repository.baseline_ref, recreate=True)
    commit_sha = manager.resolve_ref(spec.repository.baseline_ref)

    baseline_id = uuid4().hex
    run_artifacts = artifacts_dir(prep.repo_root) / "baseline" / baseline_id
    run_artifacts.mkdir(parents=True, exist_ok=True)

    secrets = venv_exec.forwarded_values(spec.secrets.forward_environment_variables)
    timeout_seconds = spec.execution.timeout_minutes * 60.0
    started_at = datetime.now(UTC)

    warnings: list[str] = []
    if commit_sha != prep.contract.baseline_commit:
        warnings.append(
            f"baseline_ref {spec.repository.baseline_ref!r} moved since approval "
            f"({prep.contract.baseline_commit[:12]} -> {commit_sha[:12]}); the current "
            "commit is recorded as fact."
        )

    fingerprint = EnvironmentFingerprint(
        platform=platform_module.platform(),
        execution_mode=engine,
        contract_id=prep.contract.contract_id,
        contract_version=prep.contract.contract_version,
        commit_sha=commit_sha,
    )

    # Repeat 1 writes into the run's own directory, so a single-run baseline has
    # exactly the layout it always had; later repeats get their own subdirectory.
    outcomes = []
    for repeat in range(1, max(n_runs, 1) + 1):
        repeat_artifacts = run_artifacts if repeat == 1 else run_artifacts / f"repeat-{repeat}"
        repeat_artifacts.mkdir(parents=True, exist_ok=True)
        if on_phase and n_runs > 1:
            on_phase(f"Baseline repeat {repeat} of {n_runs}…")
        outcomes.append(
            run_evaluation(
                spec=spec,
                engine=engine,
                command=spec.execution.full_command,
                worktree=worktree,
                run_artifacts=repeat_artifacts,
                runner=active_runner,
                secrets=secrets,
                timeout_seconds=timeout_seconds,
                fingerprint=fingerprint,
                name_slug=f"bl-{baseline_id[:12]}",
                on_phase=on_phase if n_runs == 1 else None,
            )
        )
        _redact_logs(sorted(repeat_artifacts.glob("*.log")), secrets)

    measured = [o for o in outcomes if o.status is EvaluationStatus.SUCCEEDED and o.metrics]
    # The first repeat stands for the run's identity: its logs are at the paths
    # everything else expects, and its failure is the one reported if none worked.
    outcome = outcomes[0]
    warnings.extend(outcome.warnings)

    metrics: MetricResult | None
    if measured:
        status = BaselineStatus.SUCCEEDED
        metrics = averaged_metrics([o.metrics for o in measured if o.metrics])
        failure_reason = None
    else:
        status = _EVALUATION_TO_BASELINE_STATUS[outcome.status]
        metrics = outcome.metrics
        failure_reason = outcome.failure_reason

    repeats = summarize_repeats(
        max(n_runs, 1),
        [o.metrics.primary_metric.value for o in measured if o.metrics],
        failed=len(outcomes) - len(measured),
    )
    if repeats is not None and repeats.failed:
        warnings.append(
            f"{repeats.failed} of {repeats.requested} baseline repeats did not produce "
            "a measurement; the mean is over the ones that did."
        )

    completed_at = datetime.now(UTC)
    run = BaselineRun(
        baseline_id=baseline_id,
        contract_id=prep.contract.contract_id,
        contract_version=prep.contract.contract_version,
        commit_sha=commit_sha,
        execution_mode=engine,
        command=spec.execution.full_command,
        status=status,
        failure_reason=failure_reason,
        metrics=metrics,
        repeats=repeats,
        warnings=warnings,
        fingerprint=outcome.fingerprint,
        stdout_path=str(run_artifacts / "stdout.log"),
        stderr_path=str(run_artifacts / "stderr.log"),
        results_path=outcome.results_path,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=(completed_at - started_at).total_seconds(),
    )
    (run_artifacts / "baseline_run.json").write_text(run.model_dump_json(indent=2), "utf-8")
    (run_artifacts / "fingerprint.json").write_text(
        outcome.fingerprint.model_dump_json(indent=2), "utf-8"
    )
    insert_baseline_run(conn, prep.project.id, run)
    if status is BaselineStatus.SUCCEEDED:
        touch_project_status(conn, ProjectStatus.BASELINED)
    return run


def baseline_gate(conn: sqlite3.Connection) -> BaselineRun:
    """Phase 1C's entry check: raises unless the latest full baseline succeeded."""
    latest = get_latest_baseline(conn)
    if latest is None:
        raise BaselineBlockedError("No baseline has been run. Run `researchforge baseline run`.")
    if latest.status is not BaselineStatus.SUCCEEDED:
        raise BaselineBlockedError(
            f"The latest baseline failed ({latest.status.value}): {latest.failure_reason} "
            "— experiments are blocked until a baseline succeeds."
        )
    return latest
