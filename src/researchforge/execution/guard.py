"""Guard command execution.

A guard is a verification command that must stay green regardless of whether
the primary metric improved.  It is run after every successful benchmark
stage in the experiment's worktree.

The guard separates "did the metric improve?" from "did anything else break?"
— a concept from the autoresearch toolkit.  A failed guard is recorded as
a constraint-style failure so it shows up in results and the audit trail.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GuardResult:
    passed: bool
    command: str
    exit_code: int
    stdout: str
    stderr: str

    def failure_reason(self) -> str:
        tail = (self.stderr or self.stdout)[-400:].strip()
        return f"guard command failed (exit {self.exit_code}): {self.command!r}\n{tail}"


def run_guard(
    command: str,
    worktree: Path,
    timeout_seconds: float = 300.0,
    extra_env: dict[str, str] | None = None,
) -> GuardResult:
    """Run the guard command in the experiment worktree.

    Never raises — all outcomes (including timeouts) are returned as a
    ``GuardResult`` so callers can record the failure cleanly.
    """
    import os

    env = {**os.environ, **(extra_env or {})}
    try:
        proc = subprocess.run(
            command,
            shell=True,  # noqa: S602 — guard command is author-configured
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        return GuardResult(
            passed=proc.returncode == 0,
            command=command,
            exit_code=proc.returncode,
            stdout=proc.stdout[-2000:],
            stderr=proc.stderr[-2000:],
        )
    except subprocess.TimeoutExpired:
        return GuardResult(
            passed=False,
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"guard timed out after {timeout_seconds:.0f}s",
        )
    except Exception as exc:
        return GuardResult(
            passed=False,
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"guard raised unexpected error: {exc}",
        )
