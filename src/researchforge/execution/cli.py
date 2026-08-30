"""`researchforge baseline` sub-app."""

from __future__ import annotations

import re
from contextlib import closing
from pathlib import Path

import typer

from researchforge.domain.baseline import BaselineRun, BaselineStatus
from researchforge.domain.environment import EnvironmentResolution, ExecutionEngine
from researchforge.execution.baseline import (
    BaselineBlockedError,
    BaselinePreparation,
    prepare_baseline,
    run_baseline,
)
from researchforge.execution.venv_exec import VENV_WARNING
from researchforge.storage.baseline_repository import delete_baselines, get_latest_baseline
from researchforge.storage.db import open_project_db
from researchforge.storage.experiment_repository import list_executions
from researchforge.utils.output import JsonOption, echo_json, echo_model

baseline_app = typer.Typer(name="baseline", no_args_is_help=True, help="Baseline runs.")


def _print_resolution(resolution: EnvironmentResolution) -> None:
    typer.echo(f"Status:         {resolution.status.value}")
    typer.echo(f"Execution mode: {resolution.execution_mode.value}")
    for reason in resolution.reasons:
        typer.echo(f"  - {reason}")
    if resolution.required_user_actions:
        typer.echo("Required actions:")
        for action in resolution.required_user_actions:
            typer.echo(f"  * {action}")


def _print_run(run: BaselineRun) -> None:
    typer.echo(f"Baseline:  {run.baseline_id}")
    typer.echo(f"Status:    {run.status.value}")
    typer.echo(f"Mode:      {run.execution_mode.value}")
    typer.echo(f"Commit:    {run.commit_sha[:12]}")
    typer.echo(f"Contract:  v{run.contract_version}")
    typer.echo(f"Duration:  {run.duration_seconds:.1f}s")
    if run.metrics is not None:
        mean_note = " (mean)" if run.repeats is not None else ""
        typer.echo(
            f"Metric:    {run.metrics.primary_metric.name} = "
            f"{run.metrics.primary_metric.value}{mean_note}"
        )
        for name, value in run.metrics.secondary_metrics.items():
            typer.echo(f"           {name} = {value}")
    if run.repeats is not None:
        repeats = run.repeats
        values = ", ".join(f"{value:.6g}" for value in repeats.values)
        typer.echo(f"Repeats:   {len(repeats.values)} of {repeats.requested} measured [{values}]")
        if repeats.stdev is not None:
            spread = f"σ = {repeats.stdev:.6g}"
            if repeats.coefficient_of_variation is not None:
                spread += f"  ({repeats.coefficient_of_variation * 100:.2f}% of the mean)"
            typer.echo(f"Spread:    {spread}")
    if run.failure_reason:
        typer.echo(f"Failure:   {run.failure_reason}")
    for warning in run.warnings:
        typer.echo(f"warning: {warning}")
    typer.echo(f"Artifacts: {run.stdout_path.rsplit('/', 1)[0]}")


@baseline_app.command()
def run(
    check: bool = typer.Option(  # noqa: B008
        False, "--check", help="Resolve the environment and stop (no execution)."
    ),
    auto_recover: bool = typer.Option(  # noqa: B008
        True,
        "--auto-recover/--no-auto-recover",
        help=(
            "Automatically attempt setup-command fixes when the baseline fails "
            "due to a setup error (default: on)."
        ),
    ),
    n_runs: int = typer.Option(  # noqa: B008
        1,
        "--n-runs",
        min=1,
        help=(
            "Measure the baseline this many times and freeze the mean. Use it "
            "when the benchmark is noisy, so improvements are compared against "
            "an average rather than one lucky run."
        ),
    ),
    json_output: JsonOption = False,
) -> None:
    """Run the baseline in an isolated worktree and store the result.

    When setup fails ResearchForge diagnoses the error and tries progressively
    better setup commands automatically (--auto-recover, enabled by default).
    """
    from researchforge.utils.progress import LiveProgress

    # ── Gate checks (fast, no progress needed) ───────────────────────────────
    with closing(open_project_db()) as conn:
        try:
            prep = prepare_baseline(conn)
        except BaselineBlockedError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None

        if check:
            if json_output:
                echo_model(prep.resolution)
            else:
                _print_resolution(prep.resolution)
            raise typer.Exit(code=0 if prep.resolution.execution_mode.value != "none" else 1)

    if prep.resolution.execution_mode is ExecutionEngine.VENV and not json_output:
        typer.echo(f"warning: {VENV_WARNING}")

    # ── Run with live progress ────────────────────────────────────────────────
    with (
        LiveProgress("Running baseline", enabled=not json_output) as progress,
        closing(open_project_db()) as conn,
    ):
        try:
            result = run_baseline(conn, on_phase=progress.phase, n_runs=n_runs)
        except BaselineBlockedError as exc:
            if exc.resolution is not None:
                if json_output:
                    echo_model(exc.resolution)
                else:
                    _print_resolution(exc.resolution)
            else:
                typer.echo(str(exc))
            raise typer.Exit(code=1) from None

    # ── Auto-recovery for setup failures ─────────────────────────────────────
    if result.status is BaselineStatus.FAILED_SETUP and auto_recover and not json_output:
        result = _attempt_setup_recovery(result, prep)

    from researchforge.analytics.service import record_event

    record_event(
        "baseline_completed",
        ok=result.status is BaselineStatus.SUCCEEDED,
        category=None if result.status is BaselineStatus.SUCCEEDED else result.status.value,
    )
    if json_output:
        echo_model(result)
    else:
        _print_run(result)
        if result.status is BaselineStatus.SUCCEEDED:
            typer.echo(
                "Baseline established. "
                "Next: researchforge experiment plan hyp-001 --synthesize"
            )
        else:
            typer.echo("Baseline failed — experiments are blocked until it succeeds.")
    if result.status is not BaselineStatus.SUCCEEDED:
        raise typer.Exit(code=1)


def _attempt_setup_recovery(result: BaselineRun, prep: BaselinePreparation) -> BaselineRun:
    """Read setup_stderr.log, diagnose, and try fixes automatically."""
    from researchforge.config.paths import contract_path
    from researchforge.contract.service import approve_contract
    from researchforge.execution.setup_recovery import diagnose
    from researchforge.storage.db import open_project_db

    # Read the stderr log
    artifacts_dir = Path(result.stdout_path).parent
    stderr_path = artifacts_dir / "setup_stderr.log"
    if not stderr_path.is_file():
        return result

    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    current_cmd = prep.contract.spec.execution.setup_command or ""
    diagnosis = diagnose(stderr, current_cmd, prep.repo_root)

    if not diagnosis.fixes:
        typer.echo(f"\nAuto-recovery: {diagnosis.raw_cause}")
        if diagnosis.docker_hint:
            typer.echo(f"\n{diagnosis.docker_hint}")
        return result

    typer.echo(f"\nAuto-recovery triggered: {diagnosis.raw_cause}")

    # Try each fix in order
    yaml_path = contract_path(prep.repo_root)
    for fix in diagnosis.fixes:
        typer.echo(f"  Trying: {fix.description}")
        typer.echo(f"  Command: {fix.new_command}")

        # Patch researchforge.yaml in-place
        if not yaml_path.is_file():
            typer.echo("  researchforge.yaml not found — cannot auto-fix.")
            break

        old_text = yaml_path.read_text(encoding="utf-8")
        new_text = _swap_setup_command(old_text, _quote_cmd(fix.new_command))
        if new_text == old_text:
            # setup_command line not found — skip this fix
            continue
        yaml_path.write_text(new_text, encoding="utf-8")

        # Re-approve the patched contract
        try:
            with closing(open_project_db()) as conn:
                approve_contract(conn, path=yaml_path, repo_root=prep.repo_root)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  Re-approval failed: {exc}")
            continue

        # Retry baseline
        try:
            with closing(open_project_db()) as conn:
                retry_result = run_baseline(conn)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  Baseline retry failed: {exc}")
            continue

        if retry_result.status is BaselineStatus.SUCCEEDED:
            typer.echo(f"  ✓ Auto-recovery succeeded with: {fix.new_command}")
            return retry_result

        typer.echo(f"  ✗ Still failing ({retry_result.status.value})")
        if fix.is_last_resort:
            break

    typer.echo("\nAuto-recovery exhausted all known fixes.")
    if diagnosis.docker_hint:
        typer.echo(f"\n{diagnosis.docker_hint}")

    # ── Final escalation: auto-generate Dockerfile + switch to Docker mode ──
    _try_docker_escalation(result, prep, yaml_path)
    return result


def _try_docker_escalation(
    result: BaselineRun,
    prep: BaselinePreparation,
    yaml_path: Path,
) -> None:
    """When all venv fixes fail: generate a Dockerfile, switch the contract to
    Docker mode, re-approve, and retry the baseline automatically.

    Only runs if Docker is available on this machine.
    """
    import shutil
    import subprocess as _subprocess

    if not shutil.which("docker"):
        typer.echo(
            "\nDocker is not installed. Install it to enable automatic Docker mode:\n"
            "  macOS:  brew install --cask docker  (then open Docker Desktop)\n"
            "  Linux:  curl -fsSL https://get.docker.com | sh\n"
            "After installing Docker, re-run: researchforge baseline run"
        )
        return

    # Check Docker daemon is running
    try:
        _subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True,
        )
    except (_subprocess.CalledProcessError, _subprocess.TimeoutExpired, OSError):
        typer.echo(
            "\nDocker is installed but the daemon is not running.\n"
            "Start Docker Desktop (macOS/Windows) or: sudo systemctl start docker (Linux)\n"
            "Then re-run: researchforge baseline run"
        )
        return

    typer.echo("\n── Auto-escalating to Docker mode ──")

    # Step 1: generate Dockerfile if none exists
    dockerfile_path = prep.repo_root / "Dockerfile"
    if not dockerfile_path.is_file():
        typer.echo("Generating Dockerfile…")
        try:
            from researchforge.ai.dockerfile_gen import build_minimal_dockerfile
            from researchforge.storage.db import open_project_db
            from researchforge.storage.scan_repository import get_latest_scan

            with closing(open_project_db()) as conn:
                scan = get_latest_scan(conn)

            if scan is None:
                typer.echo("  Cannot generate Dockerfile: no repository scan found.")
                return
            dockerfile_path.write_text(build_minimal_dockerfile(scan), encoding="utf-8")
            typer.echo(f"  ✓ Dockerfile written to {dockerfile_path}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  Failed to generate Dockerfile: {exc}")
            return
    else:
        typer.echo(f"  Using existing Dockerfile at {dockerfile_path}")

    # Step 2: switch researchforge.yaml to Docker mode
    if not yaml_path.is_file():
        return
    old_yaml = yaml_path.read_text(encoding="utf-8")
    new_yaml = re.sub(
        r"(mode:\s*)(auto|venv)(\s*#.*)?",
        r"\1docker\3",
        old_yaml,
        count=1,
    )
    if new_yaml == old_yaml:
        # mode line not found — add it under [execution]
        new_yaml = re.sub(
            r"(execution:\s*\n)",
            r"\1  mode: docker\n",
            old_yaml,
        )
    yaml_path.write_text(new_yaml, encoding="utf-8")
    typer.echo("  ✓ researchforge.yaml → execution.mode: docker")

    # Step 3: re-approve
    try:
        from researchforge.contract.service import approve_contract

        with closing(open_project_db()) as conn:
            approve_contract(conn, path=yaml_path, repo_root=prep.repo_root)
        typer.echo("  ✓ Contract re-approved with Docker mode")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"  Re-approval failed: {exc}")
        return

    # Step 4: retry baseline
    typer.echo("  Retrying baseline run in Docker…")
    try:
        from researchforge.storage.db import open_project_db

        with closing(open_project_db()) as conn:
            docker_result = run_baseline(conn)
        if docker_result.status is BaselineStatus.SUCCEEDED:
            typer.echo("  ✓ Baseline succeeded in Docker mode!")
        else:
            typer.echo(
                f"  ✗ Docker baseline also failed ({docker_result.status.value}).\n"
                "  Check the Dockerfile and setup_stderr.log for details."
            )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"  Docker baseline failed: {exc}")


def _swap_setup_command(yaml_text: str, quoted_command: str) -> str:
    """Replace the contract's setup_command value, keeping the key's formatting.

    The replacement goes through a function rather than a template string so a
    command containing backslashes or `\\1` is inserted literally.
    """
    return re.sub(
        r"(setup_command:\s*).*",
        lambda match: match.group(1) + quoted_command,
        yaml_text,
    )


def _quote_cmd(cmd: str) -> str:
    """Wrap command in double quotes for YAML if it contains special chars."""
    if any(c in cmd for c in ":#{}[]|>&*?"):
        return '"' + cmd.replace('"', '\\"') + '"'
    return f'"{cmd}"'


@baseline_app.command()
def show(json_output: JsonOption = False) -> None:
    """Show the latest baseline run."""
    with closing(open_project_db()) as conn:
        latest = get_latest_baseline(conn)
    if latest is None:
        typer.echo("No baseline has been run. Run `researchforge baseline run`.")
        raise typer.Exit(code=1)
    if json_output:
        echo_model(latest)
    else:
        _print_run(latest)


# Alias: `researchforge baseline status` → same as `baseline show`
@baseline_app.command("status")
def status(json_output: JsonOption = False) -> None:
    """Show the latest baseline run (alias for `baseline show`)."""
    show(json_output=json_output)


@baseline_app.command("reset")
def reset(
    confirm: bool = typer.Option(  # noqa: B008
        False, "--confirm", help="Required: confirms the frozen baseline should be dropped."
    ),
    force: bool = typer.Option(  # noqa: B008
        False,
        "--force",
        help="Also drop it when experiments have already been measured against it.",
    ),
    json_output: JsonOption = False,
) -> None:
    """Drop the frozen baseline so the next `baseline run` measures a new one.

    The baseline is the reference every recorded improvement is stated against,
    so dropping it while experiments depend on it would leave those numbers
    describing a comparison that no longer exists. That case needs --force, and
    the experiment records are left untouched either way: they keep the baseline
    value they were measured against.
    """
    if not confirm:
        typer.echo(
            "Refusing to reset without --confirm. This drops the frozen baseline; "
            "the next `researchforge baseline run` measures a new one."
        )
        raise typer.Exit(code=1)

    with closing(open_project_db()) as conn:
        existing = get_latest_baseline(conn)
        if existing is None:
            typer.echo("No baseline to reset.")
            raise typer.Exit(code=1)

        dependents = [
            execution.experiment_id
            for execution in list_executions(conn)
            if execution.metrics is not None
        ]
        if dependents and not force:
            typer.echo(
                f"{len(set(dependents))} experiment(s) have been measured against this "
                "baseline. Resetting it leaves their recorded improvements describing a "
                "comparison that no longer exists.\n"
                "Re-run with --confirm --force if that is what you want."
            )
            raise typer.Exit(code=1)

        removed = delete_baselines(conn)

    if json_output:
        echo_json({"removed": removed, "dependent_experiments": sorted(set(dependents))})
        return
    typer.echo(f"Removed {removed} baseline record(s).")
    if dependents:
        typer.echo(
            f"{len(set(dependents))} experiment(s) still record the old baseline value "
            "they were measured against."
        )
    typer.echo("Next: researchforge baseline run")
