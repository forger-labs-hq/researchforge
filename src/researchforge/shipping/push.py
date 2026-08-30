"""Where a shipped change is pushed, and what exactly gets pushed.

Two decisions live here, both of which used to be implicit:

* **Destination.** `gh` picks a base repository from remote names on its own,
  and prefers a remote called `upstream`. A project cloned from someone else's
  repository therefore targets *their* repository even when `origin` is yours.
  Targets are resolved from remote URLs here and pinned explicitly instead.
* **Content.** The shipped commit sits on the frozen baseline, because that is
  the tree the benchmark measured. When the destination's branch does not
  contain that baseline, a pull request from it carries every local commit in
  between. Replaying just the changed files onto the destination's branch keeps
  the diff to the change itself — at the cost of no longer being byte-for-byte
  the measured tree, which callers are expected to disclose.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from researchforge.execution.worktrees import WorktreeError, WorktreeManager

_SSH_URL = re.compile(r"^(?:ssh://)?git@(?P<host>[^:/]+)[:/](?P<nwo>[^/]+/[^/]+?)(?:\.git)?$")
_HTTP_URL = re.compile(r"^https?://(?:[^@/]+@)?(?P<host>[^/]+)/(?P<nwo>[^/]+/[^/]+?)(?:\.git)?/?$")

REPLAY_WORKTREE = "ship-replay"


class PushError(Exception):
    """A destination could not be resolved or a replay could not be built."""


@dataclass(frozen=True)
class PushTarget:
    """A GitHub repository this change can be pushed to."""

    nwo: str
    url: str
    remote: str | None = None

    @property
    def source(self) -> str:
        """What `git push` should be given: a remote name when there is one."""
        return self.remote or self.url

    def describe(self) -> str:
        return f"{self.remote} → {self.nwo}" if self.remote else self.nwo


def nwo_from_url(url: str) -> str | None:
    """`owner/name` for a GitHub remote URL, or None for anything else."""
    for pattern in (_SSH_URL, _HTTP_URL):
        match = pattern.match(url.strip())
        if match and match.group("host").endswith("github.com"):
            return match.group("nwo")
    return None


def github_targets(remotes: Sequence[tuple[str, str]]) -> list[PushTarget]:
    """The GitHub repositories among `(name, push url)` remotes, in git's order."""
    targets: list[PushTarget] = []
    seen: set[str] = set()
    for name, url in remotes:
        nwo = nwo_from_url(url)
        if nwo is None or name in seen:
            continue
        seen.add(name)
        targets.append(PushTarget(nwo=nwo, url=url, remote=name))
    return targets


def target_from_url(url: str) -> PushTarget:
    """A destination typed by hand rather than picked from the remotes."""
    nwo = nwo_from_url(url)
    if nwo is None:
        raise PushError(f"{url!r} is not a GitHub repository URL.")
    return PushTarget(nwo=nwo, url=url.strip())


@dataclass(frozen=True)
class Replay:
    """A commit carrying only the shipped change, built on the target's base."""

    commit_sha: str
    base_sha: str
    changed_files: list[str]
    diverged_files: list[str]

    @property
    def clean(self) -> bool:
        """Whether the base holds the same version of every changed file that
        the benchmark measured — if not, the replay lands on different code."""
        return not self.diverged_files


def replay_onto(
    manager: WorktreeManager,
    *,
    source_sha: str,
    baseline_commit: str,
    changed_files: list[str],
    base_sha: str,
    message_file: Path,
) -> Replay:
    """Commit `changed_files` as they are at `source_sha` on top of `base_sha`.

    The result is a single commit whose diff against the base is exactly the
    shipped change, with no baseline scaffolding or unrelated local history.
    """
    if not changed_files:
        raise PushError("the shipped change touches no files — nothing to push.")

    diverged = [
        path
        for path in changed_files
        if manager.blob_sha(baseline_commit, path) != manager.blob_sha(base_sha, path)
    ]

    worktree = manager.create(REPLAY_WORKTREE, base_sha, recreate=True)
    try:
        manager.checkout_paths(worktree, source_sha, changed_files)
        try:
            commit_sha = manager.commit_all_in_worktree(worktree, message_file)
        except WorktreeError as exc:
            raise PushError(
                "the shipped change is already present on the target branch — "
                f"nothing to open a pull request for ({exc})."
            ) from None
    finally:
        manager.remove(REPLAY_WORKTREE)

    replayed = sorted(manager.diff_names(base_sha, commit_sha))
    unexpected = [path for path in replayed if path not in changed_files]
    if unexpected:
        raise PushError(
            f"replay touched files outside the shipped change ({unexpected}) — nothing pushed."
        )
    return Replay(
        commit_sha=commit_sha,
        base_sha=base_sha,
        changed_files=replayed,
        diverged_files=diverged,
    )
