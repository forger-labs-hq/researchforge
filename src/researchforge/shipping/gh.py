"""GitHub CLI seam for `ship pr`.

Short foreground calls with captured output (unlike the evaluation
CommandRunner, which streams to log files). Everything goes through the
injectable ProcessRunner so tests never touch the network or a real gh.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class ProcessResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class ProcessRunner(Protocol):
    def run(
        self, argv: list[str], *, cwd: Path, timeout_seconds: float = 60.0
    ) -> ProcessResult: ...


class SubprocessProcessRunner:
    def run(self, argv: list[str], *, cwd: Path, timeout_seconds: float = 60.0) -> ProcessResult:
        try:
            completed = subprocess.run(  # noqa: S603
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProcessResult(exit_code=124, stdout="", stderr=str(exc))
        return ProcessResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class GhError(Exception):
    """A gh/git remote operation failed; message includes the tool's output."""


class GhClient:
    def __init__(self, runner: ProcessRunner | None = None) -> None:
        self._runner = runner or SubprocessProcessRunner()

    def available(self) -> bool:
        return shutil.which("gh") is not None

    def auth_ok(self, cwd: Path) -> bool:
        return self._runner.run(["gh", "auth", "status"], cwd=cwd).ok

    def default_remote_exists(self, cwd: Path) -> bool:
        return self._runner.run(["git", "remote", "get-url", "origin"], cwd=cwd).ok

    def list_remotes(self, cwd: Path) -> list[tuple[str, str]]:
        """Configured `(name, push url)` pairs, in git's order."""
        result = self._runner.run(["git", "remote", "--verbose"], cwd=cwd)
        if not result.ok:
            raise GhError(f"could not list git remotes: {result.stderr.strip()}")
        remotes: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "(push)":
                remotes.append((parts[0], parts[1]))
        return remotes

    def push_commit(self, cwd: Path, commit: str, branch: str, destination: str) -> None:
        """Push one commit to `branch` on a remote name or URL, fast-forward only."""
        result = self._runner.run(
            ["git", "push", destination, f"{commit}:refs/heads/{branch}"],
            cwd=cwd,
            timeout_seconds=300.0,
        )
        if not result.ok:
            raise GhError(f"git push failed: {result.stderr.strip() or result.stdout.strip()}")

    def viewer_can_push(self, cwd: Path, repo: str) -> bool:
        """Whether the authenticated gh user has write access to `repo`."""
        result = self._runner.run(
            [
                "gh",
                "repo",
                "view",
                repo,
                "--json",
                "viewerPermission",
                "-q",
                ".viewerPermission",
            ],
            cwd=cwd,
        )
        return result.ok and result.stdout.strip() in ("WRITE", "MAINTAIN", "ADMIN")

    def default_branch(self, cwd: Path, repo: str) -> str:
        """`repo`'s default branch — the base a pull request targets."""
        result = self._runner.run(
            [
                "gh",
                "repo",
                "view",
                repo,
                "--json",
                "defaultBranchRef",
                "-q",
                ".defaultBranchRef.name",
            ],
            cwd=cwd,
        )
        if not result.ok or not result.stdout.strip():
            raise GhError(
                f"could not resolve the default branch of {repo}: "
                f"{result.stderr.strip() or 'no branch reported'}"
            )
        return result.stdout.strip()

    def viewer_login(self, cwd: Path) -> str:
        """The authenticated gh account (the fork owner for cross-repo heads)."""
        result = self._runner.run(["gh", "api", "user", "-q", ".login"], cwd=cwd)
        if not result.ok or not result.stdout.strip():
            raise GhError(f"could not resolve the gh account: {result.stderr.strip()}")
        return result.stdout.strip()

    def fork_and_add_remote(self, cwd: Path, repo: str, remote: str = "fork") -> None:
        """Create (or reuse) a fork of `repo` and add it as `remote`."""
        if self._runner.run(["git", "remote", "get-url", remote], cwd=cwd).ok:
            return  # remote already wired (e.g. a previous ship pr)
        result = self._runner.run(
            ["gh", "repo", "fork", repo, "--remote", "--remote-name", remote],
            cwd=cwd,
            timeout_seconds=120.0,
        )
        if not result.ok:
            raise GhError(f"gh repo fork failed: {result.stderr.strip() or result.stdout.strip()}")

    def commits_ahead(self, cwd: Path, branch: str, base_ref: str) -> int | None:
        """Commits on `branch` that are not on `base_ref` (None when unknown)."""
        count = self._runner.run(["git", "rev-list", "--count", f"{base_ref}..{branch}"], cwd=cwd)
        if not count.ok:
            return None
        try:
            return int(count.stdout.strip())
        except ValueError:
            return None

    def create_draft_pr(
        self,
        cwd: Path,
        *,
        branch: str,
        title: str,
        body_file: Path,
        base: str | None,
        repo: str | None = None,
        head_owner: str | None = None,
    ) -> str:
        """Open a DRAFT pull request; returns its URL.

        `repo` + `head_owner` produce a cross-repository PR (fork workflow):
        base repo `repo`, head `head_owner:branch`.
        """
        head = f"{head_owner}:{branch}" if head_owner else branch
        argv = [
            "gh",
            "pr",
            "create",
            "--draft",
            "--head",
            head,
            "--title",
            title,
            "--body-file",
            str(body_file),
        ]
        if repo is not None:
            argv += ["--repo", repo]
        if base is not None:
            argv += ["--base", base]
        result = self._runner.run(argv, cwd=cwd, timeout_seconds=120.0)
        if not result.ok:
            raise GhError(f"gh pr create failed: {result.stderr.strip() or result.stdout.strip()}")
        url = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        if not url.startswith("http"):
            raise GhError(f"gh pr create did not return a URL (got {url!r}).")
        return url
