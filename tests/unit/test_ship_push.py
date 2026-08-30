"""Destination resolution and change replay for `ship pr` — real git, no network."""

import subprocess
from pathlib import Path

import pytest

from researchforge.execution.worktrees import WorktreeManager
from researchforge.shipping.push import (
    PushError,
    github_targets,
    nwo_from_url,
    replay_onto,
    target_from_url,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


class TestReadingRemoteUrls:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://github.com/acme/repo.git", "acme/repo"),
            ("https://github.com/acme/repo", "acme/repo"),
            ("https://user@github.com/acme/repo.git", "acme/repo"),
            ("git@github.com:acme/repo.git", "acme/repo"),
            ("ssh://git@github.com/acme/repo.git", "acme/repo"),
            ("https://gitlab.com/acme/repo.git", None),
            ("/tmp/local/bare.git", None),
        ],
    )
    def test_owner_and_name(self, url: str, expected: str | None) -> None:
        assert nwo_from_url(url) == expected

    def test_only_github_remotes_are_offered(self) -> None:
        targets = github_targets(
            [
                ("origin", "https://github.com/me/mine.git"),
                ("upstream", "git@github.com:them/theirs.git"),
                ("backup", "/srv/git/mirror.git"),
            ]
        )

        assert [(t.remote, t.nwo) for t in targets] == [
            ("origin", "me/mine"),
            ("upstream", "them/theirs"),
        ]

    def test_a_pasted_url_needs_to_be_github(self) -> None:
        assert target_from_url("https://github.com/me/mine").nwo == "me/mine"
        with pytest.raises(PushError):
            target_from_url("https://example.com/me/mine")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository whose `main` and baseline have diverged, as a real project's
    would once work continued after the baseline was frozen."""
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "--initial-branch=main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "config.py").write_text("EPOCHS = 10\n", encoding="utf-8")
    (root / "notes.md").write_text("baseline notes\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-m", "baseline")
    return root


def _commit(repo: Path, path: str, contents: str, message: str) -> str:
    (repo / path).write_text(contents, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


class TestReplayingOntoABase:
    def test_only_the_changed_files_land_on_the_base(self, repo: Path) -> None:
        baseline = git(repo, "rev-parse", "HEAD")
        # The measured tree also carries scaffolding that must not be shipped.
        _commit(repo, "benchmark.py", "print('bench')\n", "add benchmark")
        winner = _commit(repo, "config.py", "EPOCHS = 15\n", "winner")
        # Meanwhile the branch being targeted moved on independently.
        git(repo, "checkout", "-b", "target", baseline)
        base = _commit(repo, "notes.md", "moved on\n", "unrelated work on main")

        manager = WorktreeManager(repo)
        message = repo / "message.txt"
        message.write_text("Increase epochs\n", encoding="utf-8")

        replay = replay_onto(
            manager,
            source_sha=winner,
            baseline_commit=baseline,
            changed_files=["config.py"],
            base_sha=base,
            message_file=message,
        )

        assert replay.changed_files == ["config.py"]
        assert manager.parent_of(replay.commit_sha) == base
        assert replay.clean  # config.py is untouched on the base
        # The scaffolding commit is not carried along.
        assert "benchmark.py" not in manager.diff_names(base, replay.commit_sha)

    def test_a_base_that_moved_the_same_file_is_reported(self, repo: Path) -> None:
        baseline = git(repo, "rev-parse", "HEAD")
        winner = _commit(repo, "config.py", "EPOCHS = 15\n", "winner")
        git(repo, "checkout", "-b", "target", baseline)
        base = _commit(repo, "config.py", "EPOCHS = 12\n", "someone else tuned it")

        manager = WorktreeManager(repo)
        message = repo / "message.txt"
        message.write_text("Increase epochs\n", encoding="utf-8")

        replay = replay_onto(
            manager,
            source_sha=winner,
            baseline_commit=baseline,
            changed_files=["config.py"],
            base_sha=base,
            message_file=message,
        )

        assert not replay.clean
        assert replay.diverged_files == ["config.py"]

    def test_a_change_already_on_the_base_is_refused(self, repo: Path) -> None:
        baseline = git(repo, "rev-parse", "HEAD")
        winner = _commit(repo, "config.py", "EPOCHS = 15\n", "winner")

        manager = WorktreeManager(repo)
        message = repo / "message.txt"
        message.write_text("Increase epochs\n", encoding="utf-8")

        with pytest.raises(PushError, match="already present"):
            replay_onto(
                manager,
                source_sha=winner,
                baseline_commit=baseline,
                changed_files=["config.py"],
                base_sha=winner,
                message_file=message,
            )

    def test_a_deleted_file_replays_as_a_deletion(self, repo: Path) -> None:
        baseline = git(repo, "rev-parse", "HEAD")
        git(repo, "rm", "notes.md")
        git(repo, "commit", "-m", "drop notes")
        winner = git(repo, "rev-parse", "HEAD")

        manager = WorktreeManager(repo)
        message = repo / "message.txt"
        message.write_text("Drop notes\n", encoding="utf-8")

        replay = replay_onto(
            manager,
            source_sha=winner,
            baseline_commit=baseline,
            changed_files=["notes.md"],
            base_sha=baseline,
            message_file=message,
        )

        assert replay.changed_files == ["notes.md"]
        assert manager.blob_sha(replay.commit_sha, "notes.md") is None
