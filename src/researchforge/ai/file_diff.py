"""Turn "here is the file, changed" into a patch git will accept.

A model asked for a unified diff has to invent `@@ -40,6 +40,10 @@` offsets for
files it is only reading, and gets them wrong often enough that whole rounds of
planning are discarded as `corrupt patch`. So it is asked for the changed file
instead, and the diff is computed here — by git, in a scratch repository, so
new files, trailing newlines and mode lines come out exactly as `git apply`
expects them.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

_GIT_TIMEOUT_SECONDS = 30


class PatchSynthesisError(RuntimeError):
    """The diff could not be produced; the message names the file."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )


def _write(root: Path, path: str, content: str) -> None:
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise PatchSynthesisError(f"{path}: refuses to leave the repository")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def synthesize_patch(before: dict[str, str], after: dict[str, str]) -> str:
    """A git patch turning `before` into `after`, keyed by repository path.

    `before` holds the file as it is at the baseline commit; a path missing
    from it is a new file. Only the paths in `after` are considered, so an
    unchanged file costs nothing.
    """
    changed = {path: text for path, text in after.items() if before.get(path) != text}
    if not changed:
        return ""

    with tempfile.TemporaryDirectory(prefix="rf-diff-") as tmp:
        root = Path(tmp)
        init = _git(root, "init", "-q")
        if init.returncode != 0:
            raise PatchSynthesisError(f"could not create a scratch repository: {init.stderr}")

        for path in changed:
            if path in before:
                _write(root, path, before[path])
        _git(root, "add", "-A")
        commit = _git(
            root,
            "-c",
            "user.email=researchforge@localhost",
            "-c",
            "user.name=ResearchForge",
            "commit",
            "-q",
            "--allow-empty",
            "--no-gpg-sign",
            "-m",
            "baseline",
        )
        if commit.returncode != 0:
            raise PatchSynthesisError(f"could not record the baseline state: {commit.stderr}")

        for path, text in changed.items():
            _write(root, path, text)
        _git(root, "add", "-A")

        diff = _git(root, "diff", "--cached", "--no-color", "HEAD")
        if diff.returncode != 0:
            raise PatchSynthesisError(f"could not diff the change: {diff.stderr}")
        return diff.stdout
