"""The editable source, as the plan's patch will find it, for an AI that cannot see it.

An IDE reads the repository itself before writing a plan. The standalone
provider path cannot, so without this it is asked to diff files it has never
opened — and invents them. Everything here is derived from the contract's
`editable_paths` and the baseline commit: no assumption about language, layout,
or file names, and nothing is read from the working tree, so what the model is
shown is exactly what its patch will be applied to.

"Exactly" includes the lineage. A compounding experiment runs on the baseline
with its ancestors' patches already applied, so a model shown the baseline
would diff against lines its patch will never meet. `content_after` replays
that chain and the result is overlaid on the snapshot, which is what makes the
model's rewrite land on the state it is actually building on.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

MAX_FILE_BYTES = 40_000
"""A file larger than this is listed but not included."""

MAX_TOTAL_BYTES = 200_000
"""Budget across all included files; the rest are listed."""

_GIT_TIMEOUT_SECONDS = 15


class RepoFile(BaseModel):
    """One file, whole. Never truncated: a partial file cannot be rewritten."""

    path: str
    content: str
    size_bytes: int


class OmittedFile(BaseModel):
    path: str
    size_bytes: int
    reason: str


class RepoSnapshot(BaseModel):
    """What the editable paths contain at `commit`, once `applied` is applied."""

    commit: str
    files: list[RepoFile] = Field(default_factory=list)
    omitted: list[OmittedFile] = Field(default_factory=list)
    applied: list[str] = Field(default_factory=list)
    """Experiments whose changes are already in this content, oldest first."""

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def content_of(self, path: str) -> str | None:
        return next((f.content for f in self.files if f.path == path), None)

    def was_omitted(self, path: str) -> OmittedFile | None:
        return next((o for o in self.omitted if o.path == path), None)


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _tracked_paths(repo: Path, commit: str, editable_paths: list[str]) -> list[str]:
    if not editable_paths:
        return []
    listing = _git(repo, "ls-tree", "-r", "--name-only", commit, "--", *editable_paths)
    if listing is None:
        return []
    return [line for line in listing.splitlines() if line.strip()]


def _blob(repo: Path, commit: str, path: str) -> str | None:
    """File contents at `commit`, or None when unreadable as text."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{path}"],
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout
    if b"\0" in raw:  # binary, however it is named
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


CONTEXT_WORKTREE = "plan-context"
"""Scratch checkout used to replay a lineage; removed before this module returns."""


def content_after(repo_root: Path, commit: str, patches: Sequence[str]) -> dict[str, str]:
    """Every file the `patches` touch, as it stands once they are applied to `commit`.

    A best effort by design: a chain that will not compose returns nothing and
    the caller falls back to the commit itself. That is the same state the run
    would fail on anyway, and import re-checks it with a real error message.
    """
    if not patches:
        return {}

    from researchforge.execution.worktrees import WorktreeError, WorktreeManager

    manager = WorktreeManager(repo_root)
    changed: dict[str, str] = {}
    try:
        scratch = manager.create(CONTEXT_WORKTREE, commit, recreate=True)
        for depth, text in enumerate(patches):
            patch_file = scratch / f".rf-context-{depth}.patch"
            patch_file.write_text(text, encoding="utf-8")
            manager.apply_patch(scratch, patch_file)
            patch_file.unlink(missing_ok=True)
        for path in manager.changed_paths(scratch):
            if path.startswith(".rf-context-"):
                continue
            try:
                changed[path] = (scratch / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    except WorktreeError:
        return {}
    finally:
        with contextlib.suppress(WorktreeError):
            manager.remove(CONTEXT_WORKTREE)
    return changed


def _relevance(path: str, terms: list[str]) -> int:
    lowered = path.lower()
    return sum(1 for term in terms if term in lowered)


def _search_terms(prioritize: str) -> list[str]:
    """Words worth matching against a path, from free text."""
    words = {word.strip(".,:;()[]\"'").lower() for word in prioritize.split()}
    return sorted(word for word in words if len(word) > 3)


def collect_editable_files(
    repo_root: Path,
    editable_paths: list[str],
    commit: str,
    *,
    prioritize: str = "",
    overlay: Mapping[str, str] | None = None,
    applied: Sequence[str] = (),
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> RepoSnapshot:
    """Every text file under `editable_paths` at `commit` that fits the budget.

    `overlay` replaces what the commit holds, so a lineage already replayed by
    `content_after` is what the model reads; a path only in the overlay is a
    file that lineage created.

    Over budget, the files most likely to matter are kept: those whose path
    matches the words in `prioritize` (the hypothesis, normally), then the
    smallest, so a budget is spent on many small files rather than one large
    one. Whatever does not fit is still listed by name, so the model can see
    that it exists and ask for nothing it cannot have.
    """
    overlay = overlay or {}
    snapshot = RepoSnapshot(commit=commit, applied=list(applied))
    terms = _search_terms(prioritize)
    candidates: list[tuple[int, int, str]] = []

    def read(path: str) -> str | None:
        return overlay[path] if path in overlay else _blob(repo_root, commit, path)

    tracked = _tracked_paths(repo_root, commit, editable_paths)
    for path in sorted({*tracked, *overlay}):
        content = read(path)
        if content is None:
            snapshot.omitted.append(
                OmittedFile(path=path, size_bytes=0, reason="not readable as text")
            )
            continue
        size = len(content.encode("utf-8"))
        if size > max_file_bytes:
            snapshot.omitted.append(
                OmittedFile(path=path, size_bytes=size, reason="larger than the per-file budget")
            )
            continue
        candidates.append((-_relevance(path, terms), size, path))

    spent = 0
    for _, size, path in sorted(candidates):
        if spent + size > max_total_bytes:
            snapshot.omitted.append(
                OmittedFile(path=path, size_bytes=size, reason="over the total budget")
            )
            continue
        content = read(path)
        if content is None:  # vanished between the two reads
            continue
        snapshot.files.append(RepoFile(path=path, content=content, size_bytes=size))
        spent += size

    snapshot.files.sort(key=lambda f: f.path)
    snapshot.omitted.sort(key=lambda o: o.path)
    return snapshot
