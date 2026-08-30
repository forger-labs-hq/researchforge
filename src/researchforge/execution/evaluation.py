"""Shared single-evaluation engine used by the baseline and experiment stages.

Runs (setup ->) (tests ->) evaluation in a prepared worktree using either a
venv or docker, parses the result file, and reports a structured outcome.
Extracted from the Phase 1B baseline runner so every benchmark stage shares
one timeout/fingerprint/artifact implementation.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from researchforge.domain.baseline import EnvironmentFingerprint
from researchforge.domain.contract import ContractSpec, NetworkMode
from researchforge.domain.environment import ExecutionEngine
from researchforge.execution import docker_exec, venv_exec
from researchforge.execution.metrics import MetricParseError, MetricResult, parse_result_file
from researchforge.execution.runner import CommandRunner, shell_argv


class EvaluationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED_SETUP = "failed_setup"
    FAILED_TESTS = "failed_tests"
    FAILED_EXECUTION = "failed_execution"
    FAILED_TIMEOUT = "failed_timeout"
    FAILED_INVALID_RESULT = "failed_invalid_result"


class EvaluationOutcome(BaseModel):
    status: EvaluationStatus
    failure_reason: str | None = None
    metrics: MetricResult | None = None
    warnings: list[str] = Field(default_factory=list)
    results_path: str | None = None
    commands: list[str] = Field(default_factory=list)
    fingerprint: EnvironmentFingerprint

    @property
    def ok(self) -> bool:
        return self.status is EvaluationStatus.SUCCEEDED


def _finalize_result(
    spec: ContractSpec, worktree: Path, run_artifacts: Path
) -> tuple[EvaluationStatus, str | None, MetricResult | None, list[str], str | None]:
    """Copy and parse the result file after a successful evaluation command."""
    source = worktree / spec.execution.result_file
    copied: str | None = None
    if source.is_file():
        target = run_artifacts / "results.json"
        shutil.copyfile(source, target)
        copied = str(target)
    try:
        metrics, warnings = parse_result_file(source, spec)
    except MetricParseError as exc:
        return (
            EvaluationStatus.FAILED_INVALID_RESULT,
            "; ".join(exc.errors),
            None,
            [],
            copied,
        )
    return EvaluationStatus.SUCCEEDED, None, metrics, warnings, copied


def run_evaluation(
    *,
    spec: ContractSpec,
    engine: ExecutionEngine,
    command: str,
    worktree: Path,
    run_artifacts: Path,
    runner: CommandRunner,
    secrets: dict[str, str],
    timeout_seconds: float,
    fingerprint: EnvironmentFingerprint,
    name_slug: str,
    test_command: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> EvaluationOutcome:
    """Run one evaluation in `worktree`; artifacts land in `run_artifacts`."""
    stdout_path = run_artifacts / "stdout.log"
    stderr_path = run_artifacts / "stderr.log"

    if engine is ExecutionEngine.VENV:
        return _run_venv(
            spec=spec,
            command=command,
            worktree=worktree,
            run_artifacts=run_artifacts,
            runner=runner,
            secrets=secrets,
            timeout_seconds=timeout_seconds,
            fingerprint=fingerprint,
            test_command=test_command,
            extra_env=extra_env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            on_phase=on_phase,
        )
    return _run_docker(
        spec=spec,
        command=command,
        worktree=worktree,
        run_artifacts=run_artifacts,
        runner=runner,
        secrets=secrets,
        timeout_seconds=timeout_seconds,
        fingerprint=fingerprint,
        name_slug=name_slug,
        test_command=test_command,
        extra_env=extra_env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        on_phase=on_phase,
    )


def _run_venv(
    *,
    spec: ContractSpec,
    command: str,
    worktree: Path,
    run_artifacts: Path,
    runner: CommandRunner,
    secrets: dict[str, str],
    timeout_seconds: float,
    fingerprint: EnvironmentFingerprint,
    test_command: str | None,
    extra_env: Mapping[str, str] | None,
    stdout_path: Path,
    stderr_path: Path,
    on_phase: Callable[[str], None] | None = None,
) -> EvaluationOutcome:
    def _phase(msg: str) -> None:
        if on_phase:
            on_phase(msg)

    commands: list[str] = []

    _phase("Creating virtual environment…")
    venv_python, outcome = venv_exec.create_venv(
        worktree, runner, timeout_seconds=timeout_seconds, log_dir=run_artifacts
    )
    if not outcome.ok:
        return EvaluationOutcome(
            status=EvaluationStatus.FAILED_SETUP,
            failure_reason="Failed to create the virtual environment; see venv_create_stderr.log.",
            commands=commands,
            fingerprint=fingerprint,
        )

    env = venv_exec.minimal_env(venv_python.parent.parent, secrets)
    if extra_env:
        env.update(extra_env)

    if spec.execution.setup_command:
        _phase(f"Installing dependencies…  ({spec.execution.setup_command[:60]})")
        commands.append(spec.execution.setup_command)
        setup_outcome = runner.run(
            shell_argv(spec.execution.setup_command),
            cwd=worktree,
            env=env,
            timeout_seconds=timeout_seconds,
            stdout_path=run_artifacts / "setup_stdout.log",
            stderr_path=run_artifacts / "setup_stderr.log",
        )
        if not setup_outcome.ok:
            reason = (
                "Setup command timed out."
                if setup_outcome.timed_out
                else f"Setup command exited {setup_outcome.exit_code}; see setup_stderr.log."
            )
            return EvaluationOutcome(
                status=EvaluationStatus.FAILED_SETUP,
                failure_reason=reason,
                commands=commands,
                fingerprint=fingerprint,
            )

    if test_command:
        _phase(f"Running tests…  ({test_command[:60]})")
        commands.append(test_command)
        test_outcome = runner.run(
            shell_argv(test_command),
            cwd=worktree,
            env=env,
            timeout_seconds=timeout_seconds,
            stdout_path=run_artifacts / "tests_stdout.log",
            stderr_path=run_artifacts / "tests_stderr.log",
        )
        if not test_outcome.ok:
            reason = (
                "Test command timed out."
                if test_outcome.timed_out
                else f"Test command exited {test_outcome.exit_code}; see tests_stderr.log."
            )
            return EvaluationOutcome(
                status=EvaluationStatus.FAILED_TESTS,
                failure_reason=reason,
                commands=commands,
                fingerprint=fingerprint,
            )

    _phase(f"Running benchmark…  ({command[:60]})")
    commands.append(command)
    eval_outcome = runner.run(
        shell_argv(command),
        cwd=worktree,
        env=env,
        timeout_seconds=timeout_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    python_version, packages_hash = venv_exec.venv_fingerprint(
        venv_python, runner, log_dir=run_artifacts
    )
    enriched = fingerprint.model_copy(
        update={"python_version": python_version, "venv_packages_hash": packages_hash}
    )
    # Persist artifacts first, then drop the venv (spec §9.4 step 8).
    shutil.rmtree(worktree / venv_exec.VENV_DIR_NAME, ignore_errors=True)

    if eval_outcome.timed_out:
        return EvaluationOutcome(
            status=EvaluationStatus.FAILED_TIMEOUT,
            failure_reason=f"Evaluation timed out after {timeout_seconds:.0f}s.",
            commands=commands,
            fingerprint=enriched,
        )
    if eval_outcome.exit_code != 0:
        return EvaluationOutcome(
            status=EvaluationStatus.FAILED_EXECUTION,
            failure_reason=f"Evaluation command exited {eval_outcome.exit_code}; see stderr.log.",
            commands=commands,
            fingerprint=enriched,
        )

    final_status, final_reason, metrics, metric_warnings, results_path = _finalize_result(
        spec, worktree, run_artifacts
    )
    return EvaluationOutcome(
        status=final_status,
        failure_reason=final_reason,
        metrics=metrics,
        warnings=metric_warnings,
        results_path=results_path,
        commands=commands,
        fingerprint=enriched,
    )


def _run_docker(
    *,
    spec: ContractSpec,
    command: str,
    worktree: Path,
    run_artifacts: Path,
    runner: CommandRunner,
    secrets: dict[str, str],
    timeout_seconds: float,
    fingerprint: EnvironmentFingerprint,
    name_slug: str,
    test_command: str | None,
    extra_env: Mapping[str, str] | None,
    stdout_path: Path,
    stderr_path: Path,
    on_phase: Callable[[str], None] | None = None,
) -> EvaluationOutcome:
    def _phase(msg: str) -> None:
        if on_phase:
            on_phase(msg)

    commands: list[str] = []

    _phase("Building Docker image…")
    tag = f"researchforge-{name_slug}"
    build_outcome = docker_exec.build_image(
        worktree, tag, runner, timeout_seconds=timeout_seconds, log_dir=run_artifacts
    )
    if not build_outcome.ok:
        reason = (
            "Docker build timed out."
            if build_outcome.timed_out
            else f"Docker build exited {build_outcome.exit_code}; see docker_build_stderr.log."
        )
        return EvaluationOutcome(
            status=EvaluationStatus.FAILED_SETUP,
            failure_reason=reason,
            commands=commands,
            fingerprint=fingerprint,
        )

    enriched = fingerprint.model_copy(
        update={"docker_image_id": docker_exec.image_id(tag, runner, log_dir=run_artifacts)}
    )

    # Compose one in-container shell command: setup && tests && evaluation.
    parts: list[str] = []
    if spec.execution.setup_command:
        _phase(f"Installing dependencies…  ({spec.execution.setup_command[:60]})")
        parts.append(spec.execution.setup_command)
    if test_command:
        _phase(f"Running tests…  ({test_command[:60]})")
        parts.append(test_command)
    _phase(f"Running benchmark…  ({command[:60]})")
    parts.append(command)
    container_command = " && ".join(parts)
    commands.extend(parts)

    container_name = f"researchforge-run-{name_slug}"
    env_pairs = dict(secrets)
    if extra_env:
        env_pairs.update(extra_env)
    argv = docker_exec.docker_run_argv(
        image=tag,
        container_name=container_name,
        worktree=worktree,
        artifacts=run_artifacts,
        execution=spec.execution,
        network=spec.network.mode if spec.network else NetworkMode.NONE,
        forwarded_names=list(env_pairs),
        command=container_command,
    )
    eval_outcome = runner.run(
        argv,
        cwd=worktree,
        env={**os.environ, **env_pairs},
        timeout_seconds=timeout_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if eval_outcome.timed_out:
        docker_exec.force_remove_container(container_name, runner, log_dir=run_artifacts)
        return EvaluationOutcome(
            status=EvaluationStatus.FAILED_TIMEOUT,
            failure_reason=f"Evaluation timed out after {timeout_seconds:.0f}s; container removed.",
            commands=commands,
            fingerprint=enriched,
        )
    if eval_outcome.exit_code != 0:
        return EvaluationOutcome(
            status=EvaluationStatus.FAILED_EXECUTION,
            failure_reason=f"Container exited {eval_outcome.exit_code}; see stderr.log.",
            commands=commands,
            fingerprint=enriched,
        )

    final_status, final_reason, metrics, metric_warnings, results_path = _finalize_result(
        spec, worktree, run_artifacts
    )
    return EvaluationOutcome(
        status=final_status,
        failure_reason=final_reason,
        metrics=metrics,
        warnings=metric_warnings,
        results_path=results_path,
        commands=commands,
        fingerprint=enriched,
    )
