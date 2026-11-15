"""Compute policy: enforces data locality and operation constraints per target.

Written to .overnight_<slug>/compute_policy.json when the user chooses a compute
target. Read by:
  - BenchmarkRunner before any data operation
  - prepare-round output (tells Claude which path to use in experiment code)
  - orchestrator before any download

Any violation raises OvernightError immediately — never a silent failure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class DataRule:
    location: str  # kaggle_servers | local_disk | remote_server_disk |
    # colab_instance | hpc_scratch | cloud_instance_disk
    experiment_code_path: str  # the path Claude MUST use in experiment code
    can_download_locally: bool
    reason: str = ""
    downloaded_by: str = ""  # e.g. "kaggle_api_inside_notebook" for Colab
    already_present: bool = False  # HPC: data may already be on scratch


@dataclass
class ComputePolicy:
    """Data locality and operation policy for a specific compute target."""

    target: str  # e.g. "kaggle_gpu", "local_cpu", "ssh_dedicated"
    slug: str
    internet_in_kernel: bool = True
    data: DataRule = field(
        default_factory=lambda: DataRule(
            location="local_disk",
            experiment_code_path="./data/",
            can_download_locally=True,
        )
    )
    forbidden_locally: list[str] = field(default_factory=list)
    what_runs_where: dict[str, str] = field(default_factory=dict)
    results_path: str = ""
    # SSH / HPC / Cloud extras
    host: str = ""
    ssh_key: str = ""
    scheduler: str = ""  # slurm | pbs | none
    partition: str = ""
    gpu_type: str = ""
    time_limit: str = ""
    auto_terminate: bool = False
    hourly_cost_usd: float = 0.0
    session_warning: str = ""  # Colab session expiry

    # ------------------------------------------------------------------
    # Core enforcement
    # ------------------------------------------------------------------

    def assert_no_local_download(self, operation: str) -> None:
        """Raise OvernightError if a local download is attempted when forbidden."""
        from researchforge.kaggle.orchestrator import OvernightError

        if not self.data.can_download_locally:
            raise OvernightError(
                f"\n⛔  POLICY VIOLATION: '{operation}' would download data locally.\n"
                f"   Compute target '{self.target}' requires data stays at "
                f"'{self.data.location}'.\n"
                f"   Use this path in experiment code: {self.data.experiment_code_path}\n"
                f"   Reason: {self.data.reason}"
            )

    def experiment_data_path(self) -> str:
        """The ONLY path Claude should use when loading data in experiment code."""
        return self.data.experiment_code_path

    def is_internet_available(self) -> bool:
        return self.internet_in_kernel

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / "compute_policy.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, artifacts_dir: Path) -> ComputePolicy | None:
        path = artifacts_dir / "compute_policy.json"
        if not path.exists():
            return None
        with path.open() as fh:
            data = json.load(fh)
        # Reconstruct nested DataRule
        data_rule = DataRule(**data.pop("data"))
        return cls(**data, data=data_rule)

    def summary_lines(self) -> list[str]:
        """Short human-readable summary for the approval screen."""
        lines = [
            f"  Compute policy: {self.target}",
            f"  Data location : {self.data.location}",
            f"  Experiment path: {self.data.experiment_code_path}",
            "  Local download : "
            + ("✅ allowed" if self.data.can_download_locally else "🚫 forbidden"),
        ]
        if self.data.reason:
            lines.append(f"  Reason         : {self.data.reason}")
        if self.session_warning:
            lines.append(f"  ⚠️  {self.session_warning}")
        return lines


# ---------------------------------------------------------------------------
# Factory: generate correct policy for each compute target
# ---------------------------------------------------------------------------


def generate_policy_for_target(
    target: str,
    slug: str,
    artifacts_dir: Path,
    *,
    data_size_gb: float = 0.0,
    free_disk_gb: float = 0.0,
    # SSH / HPC fields
    host: str = "",
    ssh_key: str = "",
    server_data_path: str = "",
    scheduler: str = "",
    partition: str = "",
    gpu_type: str = "",
    time_limit: str = "",
    data_already_present: bool = False,
    # Cloud
    auto_terminate: bool = False,
    hourly_cost_usd: float = 0.0,
) -> ComputePolicy:
    """Create the correct ComputePolicy for the chosen compute target.

    Call this immediately after the user confirms their compute choice.
    The policy is saved to artifacts_dir/compute_policy.json.
    """
    data_dir = str(artifacts_dir / "data")
    results_dir = str(artifacts_dir / "kernel_outputs")

    if target in ("local_cpu", "local_gpu"):
        policy = ComputePolicy(
            target=target,
            slug=slug,
            internet_in_kernel=True,
            data=DataRule(
                location="local_disk",
                experiment_code_path=data_dir,
                can_download_locally=True,
                reason=f"Data fits locally ({data_size_gb:.1f} GB, {free_disk_gb:.0f} GB free)",
            ),
            what_runs_where={
                "proposals": "local_machine",
                "experiments": "local_machine",
                "results": "local",
            },
            results_path=str(artifacts_dir / "output"),
        )

    elif target == "kaggle_gpu":
        policy = ComputePolicy(
            target=target,
            slug=slug,
            internet_in_kernel=True,
            data=DataRule(
                location="kaggle_servers",
                experiment_code_path=f"/kaggle/input/{slug}/",
                can_download_locally=False,
                reason=(
                    f"Data stays on Kaggle's servers ({data_size_gb:.0f} GB). "
                    "Attached automatically via competition_sources in kernel-metadata.json."
                ),
            ),
            forbidden_locally=[
                "kaggle competitions download",
                "curl or wget > 10MB to local disk",
                "kaggle datasets download to local",
            ],
            what_runs_where={
                "proposals": "local_machine",
                "experiments": "kaggle_kernel",
                "results": "fetched_to_local",
            },
            results_path=results_dir,
        )

    elif target == "kaggle_gpu_offline":
        policy = ComputePolicy(
            target=target,
            slug=slug,
            internet_in_kernel=False,
            data=DataRule(
                location="kaggle_servers",
                experiment_code_path=f"/kaggle/input/{slug}/",
                can_download_locally=False,
                reason="Internet disabled in kernel. Data auto-attached via competition_sources.",
            ),
            forbidden_locally=[
                "kaggle competitions download",
                "any internet call from experiment code",
            ],
            what_runs_where={
                "proposals": "local_machine (pre-planned batch)",
                "experiments": "kaggle_kernel (no internet)",
                "results": "fetched_to_local",
            },
            results_path=results_dir,
        )

    elif target in ("colab_free", "colab_pro"):
        policy = ComputePolicy(
            target=target,
            slug=slug,
            internet_in_kernel=True,
            data=DataRule(
                location="colab_instance",
                experiment_code_path=f"/content/data/{slug}/",
                can_download_locally=False,
                downloaded_by="kaggle_api_inside_notebook_cell",
                reason="Data downloads inside Colab instance, not to local machine.",
            ),
            what_runs_where={
                "proposals": "local_machine",
                "experiments": "colab_instance",
                "results": "google_drive_sync_then_local",
            },
            results_path=str(artifacts_dir / "colab_results"),
            session_warning=(
                "Colab session expires (~12h). Save results to Google Drive "
                "before the session ends."
            ),
        )

    elif target in ("ssh_dedicated", "ssh_workstation"):
        server_path = server_data_path or f"/workspace/data/{slug}/"
        policy = ComputePolicy(
            target=target,
            slug=slug,
            internet_in_kernel=True,
            host=host,
            ssh_key=ssh_key or "~/.ssh/id_rsa",
            data=DataRule(
                location="remote_server_disk",
                experiment_code_path=server_path,
                can_download_locally=False,
                already_present=data_already_present,
                reason=(
                    f"Data on remote server at {server_path}. "
                    "Downloaded there during ssh-setup, never to local machine."
                ),
            ),
            what_runs_where={
                "proposals": "local_machine",
                "experiments": "remote_server (tmux persistent session)",
                "results": "rsync_to_local",
            },
            results_path=str(artifacts_dir / "remote_results"),
        )

    elif target == "hpc_slurm":
        scratch_path = server_data_path or f"/scratch/$USER/{slug}/"
        policy = ComputePolicy(
            target=target,
            slug=slug,
            internet_in_kernel=True,  # may vary — set by caller
            host=host,
            ssh_key=ssh_key or "~/.ssh/id_rsa",
            scheduler="slurm",
            partition=partition or "gpu",
            gpu_type=gpu_type or "any",
            time_limit=time_limit or "12:00:00",
            data=DataRule(
                location="hpc_scratch",
                experiment_code_path=scratch_path,
                can_download_locally=False,
                already_present=data_already_present,
                reason=f"Data on HPC scratch at {scratch_path}.",
            ),
            what_runs_where={
                "proposals": "local_machine (pre-planned batch)",
                "experiments": "hpc_compute_node (sbatch job)",
                "results": "rsync_to_local",
            },
            results_path=str(artifacts_dir / "hpc_results"),
        )

    elif target == "cloud_vm":
        instance_path = server_data_path or f"/workspace/data/{slug}/"
        policy = ComputePolicy(
            target=target,
            slug=slug,
            internet_in_kernel=True,
            host=host,
            ssh_key=ssh_key or "~/.ssh/id_rsa",
            auto_terminate=auto_terminate,
            hourly_cost_usd=hourly_cost_usd,
            data=DataRule(
                location="cloud_instance_disk",
                experiment_code_path=instance_path,
                can_download_locally=False,
                reason="Data downloads to cloud instance, not to local machine.",
            ),
            what_runs_where={
                "proposals": "local_machine",
                "experiments": "cloud_instance (tmux persistent session)",
                "results": "rsync_to_local",
            },
            results_path=str(artifacts_dir / "cloud_results"),
        )

    else:
        # Unknown target — safe default (allow local, flag as unknown)
        policy = ComputePolicy(
            target=f"unknown:{target}",
            slug=slug,
            data=DataRule(
                location="local_disk",
                experiment_code_path=data_dir,
                can_download_locally=True,
                reason="Unknown target — defaulting to local.",
            ),
        )

    policy.save(artifacts_dir)
    return policy
