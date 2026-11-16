"""Compute router — dispatches experiments to the right execution target.

Reads compute_policy.json and the user's priority list to decide WHERE each
experiment runs. Enforces data locality. Never touches compute targets that
would violate the policy.

Supported targets:
  local_cpu    — subprocess on this machine, data on local disk
  local_gpu    — same, with CUDA/MPS auto-detected
  kaggle_gpu   — push kernel, poll, fetch output
  colab_free   — generate .ipynb, user clicks Run
  colab_pro    — same, A100
  ssh_server   — rsync code, run there, rsync results
  cloud_gpu    — provision instance, run, fetch, terminate
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchforge.kaggle.orchestrator import OvernightSetup


# ---------------------------------------------------------------------------
# ExperimentSpec — what goes in the execution queue
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSpec:
    """All routing information for one experiment."""

    hypothesis_id: str
    parent_exp_ids: list[str] = field(default_factory=list)  # --parent or --merge
    code: str = ""  # complete runnable Python
    title: str = ""

    # Compute requirements
    compute_target: str = ""  # override: force specific target
    needs_gpu: bool = False  # True = prefer GPU targets
    needs_internet: bool = False  # True = forbid internet-off targets
    dataset_sources: list[str] = field(default_factory=list)  # external Kaggle datasets
    # Other notebooks' outputs, Kaggle Models, Docker image, GPU type (api.kernel_inputs)
    inputs: dict[str, Any] = field(default_factory=dict)
    estimated_runtime_mins: int = 30  # for scheduling

    # Evidence quality
    metric_confidence: str = "medium"  # high / medium / low / blind
    priority: int = 5  # 1 = highest, 10 = lowest

    def requires_kaggle_gpu_validation(self) -> bool:
        """True if this experiment can't be evaluated locally."""
        return self.metric_confidence == "blind"

    def can_run_on_target(self, target: str) -> bool:
        """Check if this experiment is compatible with the given target."""
        if self.needs_internet and target in ("kaggle_gpu_offline",):
            return False
        return not (self.needs_gpu and target in ("local_cpu",))


# ---------------------------------------------------------------------------
# ComputeRouter
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    target: str
    reason: str
    data_path: str
    policy_enforced: bool = True


class ComputeRouter:
    """Routes ExperimentSpec to the correct execution target.

    Decision logic:
    1. If spec has an explicit compute_target → use that (check policy first)
    2. If compute_policy says a specific target → use that
    3. Otherwise: try targets in user's priority order
    4. Validate data locality policy before every decision
    """

    def __init__(
        self,
        setup: OvernightSetup,
        priority: list[str] | None = None,
    ) -> None:
        from researchforge.kaggle.compute_policy import ComputePolicy

        self.setup = setup
        self.policy = ComputePolicy.load(setup.artifacts_dir)

        # Default priority: Kaggle GPU first for Kaggle competitions
        selected_gpu = getattr(setup, "selected_gpu", None)
        gpu_val = selected_gpu.value if selected_gpu else "cpu"

        self.priority = priority or _default_priority(gpu_val)

    def route(self, spec: ExperimentSpec) -> RoutingDecision:
        """Decide which target to use for this experiment."""
        # Explicit override in spec
        if spec.compute_target:
            target = spec.compute_target
            self._assert_policy(target, spec)
            return RoutingDecision(
                target=target,
                reason="Explicitly requested by experiment spec",
                data_path=self._data_path_for(target),
            )

        # Policy-mandated target
        if self.policy:
            policy_target = _policy_to_target(self.policy)
            if policy_target:
                self._assert_policy(policy_target, spec)
                return RoutingDecision(
                    target=policy_target,
                    reason=f"Mandated by compute_policy.json ({self.policy.target})",
                    data_path=self.policy.experiment_data_path(),
                )

        # Try targets in priority order
        for target in self.priority:
            if not spec.can_run_on_target(target):
                continue
            if self._target_available(target):
                return RoutingDecision(
                    target=target,
                    reason=f"First available in priority order ({self.priority})",
                    data_path=self._data_path_for(target),
                )

        # Fallback
        return RoutingDecision(
            target="local_cpu",
            reason="No preferred target available, falling back to local CPU",
            data_path=str(self.setup.data_dir),
        )

    def _assert_policy(self, target: str, spec: ExperimentSpec) -> None:
        """Raise if routing this spec to target would violate the policy."""
        if not self.policy:
            return
        # Non-local target chosen but policy says data must stay elsewhere
        if target in ("local_cpu", "local_gpu") and not self.policy.data.can_download_locally:
            from researchforge.kaggle.orchestrator import OvernightError

            raise OvernightError(
                f"Cannot route to {target}: compute_policy forbids local download.\n"
                f"Data must stay at '{self.policy.data.location}'.\n"
                f"Use target: {_policy_to_target(self.policy) or 'kaggle_gpu'}"
            )

    def _target_available(self, target: str) -> bool:
        """Quick availability check for a target (fast, no API calls)."""
        if target in ("local_cpu", "local_gpu"):
            return True
        if target in ("kaggle_gpu", "kaggle_gpu_offline"):
            creds = Path.home() / ".kaggle" / "kaggle.json"
            return creds.exists()
        if target in ("colab_free", "colab_pro"):
            return True  # always viable (user must click Run)
        if target in ("ssh_server",):
            host = getattr(self.setup, "ssh_host", "")
            return bool(host)
        if target == "cloud_gpu":
            for key in ("RUNPOD_API_KEY", "LAMBDA_API_KEY", "VAST_API_KEY", "AWS_ACCESS_KEY_ID"):
                if os.environ.get(key):
                    return True
            return False
        return False

    def _data_path_for(self, target: str) -> str:
        """Return the data path experiment code should use for this target."""
        if self.policy:
            return self.policy.experiment_data_path()
        slug = self.setup.competition_slug
        if target in ("kaggle_gpu", "kaggle_gpu_offline"):
            return f"/kaggle/input/{slug}/"
        if target in ("colab_free", "colab_pro"):
            return f"/content/data/{slug}/"
        return str(self.setup.data_dir)


# ---------------------------------------------------------------------------
# Execution dispatch
# ---------------------------------------------------------------------------


def dispatch_experiment(
    spec: ExperimentSpec,
    decision: RoutingDecision,
    setup: OvernightSetup,
    timeout_seconds: float = 1800,
) -> dict[str, Any]:
    """Execute an experiment on the chosen target. Returns result dict.

    result = {
      "target": str,
      "passed": bool,
      "metric_value": float | None,
      "error": str | None,
      "duration_seconds": float,
    }
    """
    target = decision.target

    if target in ("local_cpu", "local_gpu"):
        return _run_local(spec, decision, setup, timeout_seconds)

    if target in ("kaggle_gpu", "kaggle_gpu_offline"):
        return _run_kaggle(spec, decision, setup, timeout_seconds)

    if target in ("colab_free", "colab_pro"):
        return _run_colab(spec, decision, setup)

    if target == "ssh_server":
        return _run_ssh(spec, decision, setup, timeout_seconds)

    if target == "cloud_gpu":
        return _run_cloud_gpu(spec, decision, setup, timeout_seconds)

    # Unknown target
    return {
        "target": target,
        "passed": False,
        "metric_value": None,
        "error": f"Unknown target '{target}' — no dispatcher implemented",
        "duration_seconds": 0.0,
    }


def _run_local(spec: Any, decision: Any, setup: Any, timeout: Any) -> dict[str, Any]:
    """Run experiment locally."""
    import os as _os
    import tempfile
    import time

    from researchforge.execution.runner import SubprocessRunner

    start = time.monotonic()
    runner = SubprocessRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = tmp / "experiment.py"
        script.write_text(spec.code)
        stdout_path = tmp / "stdout.txt"
        stderr_path = tmp / "stderr.txt"

        outcome = runner.run(
            [sys.executable, str(script)],
            cwd=setup.data_dir if setup.data_dir.exists() else Path.cwd(),
            env=_os.environ.copy(),
            timeout_seconds=timeout,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

        stdout = stdout_path.read_text() if stdout_path.exists() else ""
        stderr = stderr_path.read_text() if stderr_path.exists() else ""
        metric = _parse_metric(stdout + stderr, setup)

    return {
        "target": decision.target,
        "passed": outcome.ok,
        "metric_value": metric,
        "error": stderr[:300] if not outcome.ok else None,
        "duration_seconds": time.monotonic() - start,
    }


def _run_kaggle(spec: Any, decision: Any, setup: Any, timeout: Any) -> dict[str, Any]:
    """Route to the Kaggle GPU benchmark runner."""
    from researchforge.kaggle.benchmark_runner import BenchmarkRunner
    from researchforge.kaggle.experiment_loop import BenchmarkResult

    runner = BenchmarkRunner(setup)
    try:
        result: BenchmarkResult = runner._run_kaggle_gpu(
            spec.code, timeout, dataset_sources=spec.dataset_sources, inputs=spec.inputs
        )
        return {
            "target": decision.target,
            "passed": result.passed,
            "metric_value": result.metric_value if result.passed else None,
            "error": result.error_message,
            "duration_seconds": result.duration_seconds,
        }
    except Exception as exc:
        return {
            "target": decision.target,
            "passed": False,
            "metric_value": None,
            "error": str(exc),
            "duration_seconds": 0.0,
        }


def _run_colab(spec: Any, decision: Any, setup: Any) -> dict[str, Any]:
    """Run on Colab unattended when the Colab CLI is signed in; else a notebook to run."""
    import tempfile

    from researchforge.kaggle import colab

    if colab.ready()[0]:
        with tempfile.TemporaryDirectory() as tmp:
            out = colab.run_round(
                spec.code,
                setup.competition_slug,
                colab.ColabConfig.load(setup.artifacts_dir),
                Path(tmp),
                getattr(setup, "time_budget_hours", 1) * 3600,
            )
        metric = _parse_metric(out["stdout_text"], setup) if out["ok"] else None
        return {
            "target": decision.target,
            "passed": metric is not None,
            "metric_value": metric,
            "error": None if metric is not None else (out["stderr_text"][-300:] or "no metric"),
            "duration_seconds": 0.0,
        }
    notebook_path = setup.artifacts_dir / f"colab_experiment_{spec.hypothesis_id}.ipynb"
    _generate_colab_notebook(spec, setup, notebook_path)

    return {
        "target": decision.target,
        "passed": None,  # None = pending user action
        "metric_value": None,
        "error": None,
        "duration_seconds": 0.0,
        "action_required": (
            f"Open {notebook_path.name} in Colab, click Run All. "
            "Save results to Drive before session expires."
        ),
        "notebook_path": str(notebook_path),
    }


def _run_ssh(spec: Any, decision: Any, setup: Any, timeout: Any) -> dict[str, Any]:
    """Run experiment on a remote SSH server.

    Steps:
      1. Load SSHConfig from artifacts_dir
      2. Create a temp dir on the remote
      3. Write experiment code locally, rsync to remote
      4. Optionally rsync competition data if ssh_config.rsync_data is True
      5. Execute code via ssh, capture stdout/stderr
      6. Parse metric from output
      7. Clean up remote temp dir
    """
    import tempfile
    import time

    from researchforge.kaggle.ssh_config import SSHConfig

    start = time.monotonic()
    artifacts_dir: Path = setup.artifacts_dir

    cfg = SSHConfig.load(artifacts_dir)
    if cfg is None:
        return {
            "target": decision.target,
            "passed": False,
            "metric_value": None,
            "error": (
                "No SSH config found. Run: "
                f"researchforge kaggle ssh-setup {setup.competition_slug} "
                "--host user@yourserver"
            ),
            "duration_seconds": 0.0,
        }

    ssh_args = cfg.ssh_base_args()
    rsync_ssh = f"ssh {' '.join(ssh_args)}"

    # ── 1. Create remote temp dir ────────────────────────────────────────
    try:
        mk = subprocess.run(
            ["ssh"] + ssh_args + [cfg.host, "mktemp -d /tmp/rf_exp_XXXXXX"],
            capture_output=True,
            text=True,
            timeout=cfg.timeout_connect_s + 5,
            check=False,
        )
        if mk.returncode != 0:
            return {
                "target": decision.target,
                "passed": False,
                "metric_value": None,
                "error": f"SSH mktemp failed: {mk.stderr.strip()[:200]}",
                "duration_seconds": time.monotonic() - start,
            }
        remote_tmpdir = mk.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "target": decision.target,
            "passed": False,
            "metric_value": None,
            "error": f"SSH connect failed: {exc}",
            "duration_seconds": time.monotonic() - start,
        }

    stdout = ""
    stderr = ""
    try:
        # ── 2. Write code to local temp file and rsync to remote ─────────
        with tempfile.TemporaryDirectory() as tmpdir:
            local_script = Path(tmpdir) / "experiment.py"
            local_script.write_text(spec.code)

            rsync_code = subprocess.run(
                [
                    "rsync",
                    "-az",
                    "-e",
                    rsync_ssh,
                    str(local_script),
                    f"{cfg.host}:{remote_tmpdir}/experiment.py",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if rsync_code.returncode != 0:
                return {
                    "target": decision.target,
                    "passed": False,
                    "metric_value": None,
                    "error": f"rsync code failed: {rsync_code.stderr.strip()[:200]}",
                    "duration_seconds": time.monotonic() - start,
                }

            # ── 3. Optionally rsync competition data ─────────────────────
            if cfg.rsync_data and setup.data_dir.exists():
                rsync_data = subprocess.run(
                    [
                        "rsync",
                        "-az",
                        "--delete",
                        "-e",
                        rsync_ssh,
                        f"{setup.data_dir}/",
                        f"{cfg.host}:{remote_tmpdir}/data/",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                if rsync_data.returncode != 0:
                    return {
                        "target": decision.target,
                        "passed": False,
                        "metric_value": None,
                        "error": f"rsync data failed: {rsync_data.stderr.strip()[:200]}",
                        "duration_seconds": time.monotonic() - start,
                    }

        # ── 4. Execute on remote ──────────────────────────────────────────
        # Patch DATA_PATH in the code to point to the remote dir
        remote_data = f"{remote_tmpdir}/data" if cfg.rsync_data else str(setup.data_dir)
        patch_cmd = (
            f"sed -i 's|DATA_PATH = .*|DATA_PATH = \"{remote_data}\"|g' "
            f"{remote_tmpdir}/experiment.py 2>/dev/null || true"
        )
        run_cmd = f"cd {remote_tmpdir} && {cfg.python_path} experiment.py 2>&1"
        full_cmd = f"{patch_cmd}; {run_cmd}"

        run_result = subprocess.run(
            ["ssh"] + ssh_args + [cfg.host, full_cmd],
            capture_output=True,
            text=True,
            timeout=timeout + cfg.timeout_connect_s + 10,
            check=False,
        )
        stdout = run_result.stdout
        stderr = run_result.stderr
        exit_ok = run_result.returncode == 0

    except subprocess.TimeoutExpired:
        return {
            "target": decision.target,
            "passed": False,
            "metric_value": None,
            "error": f"SSH experiment timed out after {timeout}s",
            "duration_seconds": time.monotonic() - start,
        }
    finally:
        # ── 5. Clean up remote temp dir (best-effort) ─────────────────────
        with contextlib.suppress(Exception):
            subprocess.run(
                ["ssh"] + ssh_args + [cfg.host, f"rm -rf {remote_tmpdir}"],
                capture_output=True,
                timeout=15,
                check=False,
            )

    metric = _parse_metric(stdout + stderr, setup)
    return {
        "target": decision.target,
        "passed": exit_ok,
        "metric_value": metric,
        "error": (stdout + stderr)[:300] if not exit_ok else None,
        "duration_seconds": time.monotonic() - start,
    }


# ---------------------------------------------------------------------------
# Cloud GPU  (RunPod / Lambda Labs — provision → SSH → run → terminate)
# ---------------------------------------------------------------------------


def _run_cloud_gpu(spec: Any, decision: Any, setup: Any, timeout: Any) -> dict[str, Any]:
    """Provision a cloud GPU instance, run the experiment via SSH, then terminate.

    Steps:
      1. Load CloudGPUConfig from artifacts_dir
      2. Provision instance (RunPod or Lambda Labs)
      3. Wait for SSH to become available
      4. Build a temporary SSHConfig pointing at the new instance
      5. Delegate to _run_ssh() — reuses all SSH execution logic
      6. Terminate instance in finally block

    The instance is always terminated even if the experiment fails.
    """
    import time as _time

    from researchforge.kaggle.cloud_gpu_config import (
        CloudGPUConfig,
        provision_and_wait,
        terminate_instance,
    )
    from researchforge.kaggle.ssh_config import SSHConfig

    start = _time.monotonic()
    artifacts_dir: Path = setup.artifacts_dir

    cfg = CloudGPUConfig.load(artifacts_dir)
    if cfg is None:
        return {
            "target": decision.target,
            "passed": False,
            "metric_value": None,
            "error": (
                "No cloud GPU config found. Run: "
                f"researchforge kaggle cloud-gpu-setup {setup.competition_slug} "
                "--provider runpod"
            ),
            "duration_seconds": 0.0,
        }

    pod_id: str | None = None
    try:
        # ── Provision ────────────────────────────────────────────────────
        print(f"☁️  Provisioning {cfg.provider} GPU ({cfg.default_gpu_type()})...", flush=True)
        host, port, pod_id = provision_and_wait(cfg)
        print(f"✅ Instance ready: {host}:{port}  pod={pod_id}", flush=True)

        # ── Build a temp SSH config pointing at the new instance ─────────
        ssh_cfg = SSHConfig(
            host=f"{host}" if port == 22 else f"{host}:{port}",
            key_path="",  # RunPod/Lambda injects account SSH keys
            remote_dir="/tmp/rf_cloud",
            python_path=cfg.python_path,
            rsync_data=cfg.rsync_data,
            timeout_connect_s=30,
        )
        # Save temporarily so _run_ssh can load it
        original_cfg_path = artifacts_dir / "ssh_config.json"
        original_backup = None
        if original_cfg_path.exists():
            original_backup = original_cfg_path.read_text()
        ssh_cfg.save(artifacts_dir)

        # For SSH on non-22 ports, we need to pass port differently.
        # RunPod uses host:port format — patch SSHConfig host if needed.
        if port != 22:
            # Override: use -p flag format in ssh_base_args via the host field
            ssh_cfg.host = host
            ssh_base_override = ["-p", str(port)]
        else:
            ssh_base_override = []
        # Patch ssh_base_args to include -p if needed
        if ssh_base_override:
            original_ssh_base_args = ssh_cfg.ssh_base_args

            def patched_ssh_base_args() -> Any:
                return original_ssh_base_args() + ssh_base_override

            ssh_cfg.ssh_base_args = patched_ssh_base_args  # type: ignore[method-assign]

        ssh_cfg.save(artifacts_dir)

        # ── Run via SSH ───────────────────────────────────────────────────
        result = _run_ssh(spec, decision, setup, timeout)
        result["cloud_pod_id"] = pod_id
        result["cloud_provider"] = cfg.provider
        return result

    except Exception as exc:
        return {
            "target": decision.target,
            "passed": False,
            "metric_value": None,
            "error": str(exc)[:400],
            "duration_seconds": _time.monotonic() - start,
            "cloud_pod_id": pod_id,
        }
    finally:
        # ── Restore original SSH config ───────────────────────────────────
        try:
            if original_backup is not None:
                (artifacts_dir / "ssh_config.json").write_text(original_backup)
            elif (artifacts_dir / "ssh_config.json").exists():
                (artifacts_dir / "ssh_config.json").unlink()
        except Exception:
            pass
        # ── Terminate instance ────────────────────────────────────────────
        if pod_id is not None:
            try:
                print(f"☁️  Terminating {cfg.provider} instance {pod_id}...", flush=True)
                terminate_instance(cfg, pod_id)
                print(f"✅ Instance {pod_id} terminated", flush=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Colab notebook generator
# ---------------------------------------------------------------------------


def _generate_colab_notebook(spec: ExperimentSpec, setup: Any, output_path: Path) -> None:
    """Generate a self-contained Colab notebook for this experiment."""
    import json as _json

    slug = setup.competition_slug
    cells: list[dict[str, Any]] = [
        {
            "cell_type": "code",
            "source": [
                "# ResearchForge experiment: " + spec.title + "\n",
                "# Hypothesis: " + spec.hypothesis_id + "\n",
                "\n",
                "# Step 1: Read credentials from Colab Secrets\n",
                "from google.colab import userdata\n",
                "import os\n",
                "for key in ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'ANTHROPIC_API_KEY', "
                "'NTFY_TOPIC']:\n",
                "    try: os.environ[key] = userdata.get(key)\n",
                "    except: pass\n",
            ],
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "source": [
                "# Step 2: Download competition data INSIDE Colab (not to local machine)\n",
                f"!kaggle competitions download -c {slug} -p /content/data/ -q\n",
                "import subprocess; subprocess.run(['bash', '-c', 'cd /content/data && ls *.zip "
                "2>/dev/null && unzip -q *.zip 2>/dev/null || true'])\n",
                f"DATA_DIR = '/content/data/{slug}/'\n",
                "import os; os.makedirs(DATA_DIR, exist_ok=True)\n",
            ],
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "source": [
                "# Step 3: Run experiment\n",
                "# NOTE: Use DATA_DIR for all data loading\n",
                "\n",
            ]
            + [line + "\n" for line in spec.code.splitlines()],
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "source": [
                "# Step 4: Save results to Drive before session expires!\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "import shutil, os\n",
                "results_dir = '/content/drive/MyDrive/researchforge_results/'\n",
                "os.makedirs(results_dir, exist_ok=True)\n",
                "# Save any output files here:\n",
                "# shutil.copy('results.json', results_dir)\n",
                "print('Done! Save any output files to', results_dir)\n",
            ],
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        },
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json.dumps(notebook, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_priority(gpu_target: str) -> list[str]:
    """Default target priority based on the user's GPU choice."""
    base = ["kaggle_gpu", "local_gpu", "local_cpu", "colab_free", "ssh_server"]
    if gpu_target in ("cpu",):
        base = ["local_cpu", "kaggle_gpu", "colab_free", "ssh_server"]
    return base


def _policy_to_target(policy: Any) -> str | None:
    """Map compute policy target name to a router target name."""
    mapping = {
        "kaggle_gpu": "kaggle_gpu",
        "kaggle_gpu_offline": "kaggle_gpu_offline",
        "local_cpu": "local_cpu",
        "local_gpu": "local_gpu",
        "colab_free": "colab_free",
        "colab_pro": "colab_pro",
        "ssh_dedicated": "ssh_server",
        "ssh_workstation": "ssh_server",
        "hpc_slurm": "ssh_server",
        "cloud_vm": "cloud_gpu",
    }
    target = getattr(policy, "target", "")
    return mapping.get(target)


def _parse_metric(text: str, setup: Any) -> float | None:
    """Parse metric value from stdout."""
    import re

    metric_name = ""
    if setup.contract:
        metric_name = setup.contract.objective.primary_metric.name

    patterns = [
        rf"(?i){re.escape(metric_name)}[:\s=]+([0-9]+\.[0-9]+)" if metric_name else r"(?!x)x",
        r"(?i)CV Score[:\s=]+([0-9]+\.[0-9]+)",
        r"(?i)Metric[:\s=]+([0-9]+\.[0-9]+)",
        r"(?i)Score[:\s=]+([0-9]+\.[0-9]+)",
        # A line that is only a number; never any long decimal (a timing, a loss).
        r"(?m)^[ \t]*([0-9]+\.[0-9]+)[ \t]*$",
    ]
    for pattern in patterns:
        if not pattern or pattern == r"(?!x)x":
            continue
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return None
