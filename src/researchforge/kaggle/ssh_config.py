"""SSH compute target configuration.

Stores connection details for running experiments on a remote GPU server.
Saved to .overnight_<slug>/ssh_config.json so execute-round can use it.

Setup:
    researchforge kaggle ssh-setup <slug> --host user@server --key ~/.ssh/id_rsa

The remote server needs:
  - SSH access with the provided key
  - Python 3.9+ with experiment dependencies installed
  - The competition data already present (or rsync will copy it)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SSHConfig:
    """SSH connection configuration for a remote experiment server."""

    host: str  # user@hostname  or  just hostname (then user = current user)
    key_path: str = ""  # path to SSH private key (~/.ssh/id_rsa)
    remote_dir: str = "~/researchforge_experiments"  # base dir on remote
    python_path: str = "python"  # python executable on remote
    rsync_data: bool = False  # True = rsync competition data to remote
    timeout_connect_s: int = 15  # SSH connect timeout

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / "ssh_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, artifacts_dir: Path) -> SSHConfig | None:
        path = artifacts_dir / "ssh_config.json"
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def ssh_base_args(self) -> list[str]:
        """Build the common ssh/rsync arguments."""
        args = [
            "-o",
            f"ConnectTimeout={self.timeout_connect_s}",
            "-o",
            "BatchMode=yes",  # never prompt for password
            "-o",
            "StrictHostKeyChecking=no",
        ]
        if self.key_path:
            args += ["-i", str(Path(self.key_path).expanduser())]
        return args

    def test_connection(self) -> tuple[bool, str]:
        """Return (ok, message) after a quick SSH connectivity check."""
        try:
            result = subprocess.run(
                ["ssh"] + self.ssh_base_args() + [self.host, "echo RF_OK"],
                capture_output=True,
                text=True,
                timeout=self.timeout_connect_s + 5,
                check=False,
            )
            if result.returncode == 0 and "RF_OK" in result.stdout:
                return True, f"SSH connection to {self.host} OK"
            return False, f"SSH failed (rc={result.returncode}): {result.stderr.strip()[:200]}"
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return False, f"SSH error: {exc}"

    def remote_python_version(self) -> str:
        """Return the Python version string on the remote, or '' on failure."""
        try:
            result = subprocess.run(
                ["ssh"] + self.ssh_base_args() + [self.host, f"{self.python_path} --version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return (result.stdout + result.stderr).strip()
        except Exception:
            return ""
