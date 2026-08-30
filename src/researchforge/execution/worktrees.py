"""Git worktree management for baseline and experiment isolation.

Worktrees live under `.researchforge/worktrees/` and are always created
detached at a resolved commit — no branches are created or moved, so the
user's working tree and branches are never touched.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from researchforge.config.paths import worktrees_dir

_VALID_NAME = re.compile(r"^[a-zA-Z0-9._-]+$")
_GIT_TIMEOUT_S = 60
_FETCH_TIMEOUT_S = 300


class WorktreeError(Exception):
    """A git worktree operation failed; message includes git's stderr."""


class WorktreeManager:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.worktrees_root = worktrees_dir(self.repo_root)

    def _git(self, *args: str, timeout: float = _GIT_TIMEOUT_S, strip: bool = True) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_root), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorktreeError(f"git {' '.join(args)} failed to run: {exc}") from exc
        if result.returncode != 0:
            raise WorktreeError(
                f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout.strip() if strip else result.stdout

    def _path_for(self, name: str) -> Path:
        if name in (".", "..") or not _VALID_NAME.match(name):
            raise WorktreeError(f"Invalid worktree name: {name!r}")
        path = (self.worktrees_root / name).resolve()
        if not path.is_relative_to(self.worktrees_root) or path == self.worktrees_root:
            raise WorktreeError(f"Worktree path escapes the worktrees directory: {name!r}")
        return path

    def resolve_ref(self, ref: str) -> str:
        """Resolve any ref to a full commit sha."""
        return self._git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")

    def create(self, name: str, ref: str, *, recreate: bool = False) -> Path:
        """Create a detached worktree at `ref`; returns its path."""
        path = self._path_for(name)
        if path.exists():
            if not recreate:
                raise WorktreeError(f"Worktree {name!r} already exists at {path}.")
            self.remove(name)
        sha = self.resolve_ref(ref)
        self.ensure_ignored()  # keep the user's `git status` clean
        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "--detach", str(path), sha)
        return path

    def remove(self, name: str) -> None:
        """Remove a worktree, falling back to rmtree + prune if git refuses."""
        path = self._path_for(name)
        if not path.exists():
            self._git("worktree", "prune")
            return
        try:
            self._git("worktree", "remove", "--force", str(path))
        except WorktreeError:
            shutil.rmtree(path, ignore_errors=True)
        self._git("worktree", "prune")

    def list_names(self) -> list[str]:
        if not self.worktrees_root.is_dir():
            return []
        return sorted(p.name for p in self.worktrees_root.iterdir() if p.is_dir())

    def apply_patch_check(self, worktree: Path, patch: Path) -> tuple[bool, str]:
        """Dry-run a patch against a worktree; returns (applies, git message)."""
        try:
            self._git("-C", str(worktree), "apply", "--check", "--verbose", str(patch))
        except WorktreeError as exc:
            return False, str(exc)
        return True, ""

    def patch_numstat(self, worktree: Path, patch: Path) -> list[str]:
        """Changed paths a patch would touch, extracted by git (never authored)."""
        output = self._git("-C", str(worktree), "apply", "--numstat", "-z", str(patch))
        paths: list[str] = []
        for entry in output.split("\0"):
            if not entry.strip():
                continue
            parts = entry.split("\t")
            if len(parts) >= 3 and parts[2]:
                paths.append(parts[2])
            elif len(parts) == 1 and paths:
                # Rename continuation records (old\0new): keep both paths.
                paths.append(parts[0])
        return paths

    def apply_patch(self, worktree: Path, patch: Path) -> None:
        """Apply a patch to a worktree (raises WorktreeError with git's message)."""
        self._git("-C", str(worktree), "apply", str(patch))

    def changed_paths(self, worktree: Path) -> list[str]:
        """Paths currently modified/added in a worktree, including untracked."""
        # strip=False: entries like " M path" carry a significant leading space.
        output = self._git("-C", str(worktree), "status", "--porcelain", "-z", "-uall", strip=False)
        paths: list[str] = []
        expect_rename_source = False
        for entry in output.split("\0"):
            if not entry:
                continue
            if expect_rename_source:
                paths.append(entry)
                expect_rename_source = False
                continue
            status, path = entry[:2], entry[3:]
            paths.append(path)
            if "R" in status or "C" in status:
                expect_rename_source = True
        return paths

    def check_branch_name(self, name: str) -> bool:
        """Whether `name` is a valid git branch name."""
        try:
            self._git("check-ref-format", "--branch", name)
        except WorktreeError:
            return False
        return True

    def branch_exists(self, name: str) -> bool:
        try:
            self._git("rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
        except WorktreeError:
            return False
        return True

    def create_branch(self, name: str, sha: str) -> None:
        """Create a branch ref at `sha` — never moves HEAD or any existing ref."""
        if self.branch_exists(name):
            raise WorktreeError(f"Branch {name!r} already exists; refusing to move it.")
        self._git("branch", name, sha)

    def delete_branch(self, name: str) -> None:
        self._git("branch", "-D", name)

    def commit_all_in_worktree(self, worktree: Path, message_file: Path) -> str:
        """Stage everything in a worktree and commit; returns the new sha.

        Requires the user's git identity to be configured — the commit is
        authored by the user, never by a synthetic identity.
        """
        try:
            self._git("-C", str(worktree), "config", "user.email")
            self._git("-C", str(worktree), "config", "user.name")
        except WorktreeError:
            raise WorktreeError(
                "Git identity is not configured — set user.name and user.email "
                "(git config) so the shipped commit is authored by you."
            ) from None
        self._git("-C", str(worktree), "add", "-A")
        self._git("-C", str(worktree), "commit", "-F", str(message_file))
        return self._git("-C", str(worktree), "rev-parse", "HEAD")

    def parent_of(self, sha: str) -> str:
        return self._git("rev-parse", f"{sha}^")

    def commit_message(self, sha: str) -> str:
        return self._git("log", "-1", "--format=%B", "--end-of-options", sha, strip=False)

    def fetch(self, source: str, ref: str) -> str:
        """Fetch one ref from a remote name or URL; returns the fetched commit."""
        self._git("fetch", "--no-tags", "--", source, ref, timeout=_FETCH_TIMEOUT_S)
        return self._git("rev-parse", "FETCH_HEAD")

    def blob_sha(self, commit: str, path: str) -> str | None:
        """The blob a path resolves to at `commit`, or None when absent there."""
        try:
            return self._git("rev-parse", "--end-of-options", f"{commit}:{path}")
        except WorktreeError:
            return None

    def checkout_paths(self, worktree: Path, commit: str, paths: Sequence[str]) -> None:
        """Materialize exactly `paths` as they are at `commit` inside a worktree.

        Paths absent at `commit` are deleted, so a change that removed a file
        replays as a removal rather than being silently skipped.
        """
        for path in paths:
            if self.blob_sha(commit, path) is not None:
                self._git("-C", str(worktree), "checkout", commit, "--", path)
            else:
                self._git("-C", str(worktree), "rm", "-f", "--ignore-unmatch", "--", path)

    def diff_names(self, base: str, head: str) -> list[str]:
        output = self._git("diff", "--name-only", "-z", base, head, strip=False)
        return [entry for entry in output.split("\0") if entry]

    def ensure_ignored(self) -> None:
        """Make git ignore `.researchforge/` via .git/info/exclude (local-only)."""
        git_dir = Path(self._git("rev-parse", "--git-common-dir"))
        if not git_dir.is_absolute():
            git_dir = self.repo_root / git_dir
        exclude = git_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        entry = ".researchforge/"
        existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        if entry not in existing.splitlines():
            with exclude.open("a", encoding="utf-8") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                handle.write(f"{entry}\n")
