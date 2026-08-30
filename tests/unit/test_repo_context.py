"""Reading the editable source at the baseline commit.

Without this the standalone planner writes diffs against files it has never
seen, and invents the ones it wishes existed.
"""

import subprocess
from pathlib import Path

import pytest

from researchforge.experiments.repo_context import collect_editable_files, content_after


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text("IMG_SIZE = 640\n", encoding="utf-8")
    (tmp_path / "src" / "model.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "evaluate.py").write_text("print(1)\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "baseline", "--no-gpg-sign")
    return tmp_path


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class TestWhatIsCollected:
    def test_editable_files_are_returned_whole(self, repo: Path) -> None:
        snapshot = collect_editable_files(repo, ["src/"], _head(repo))

        assert snapshot.paths == ["src/config.py", "src/model.py"]
        assert snapshot.content_of("src/config.py") == "IMG_SIZE = 640\n"

    def test_paths_outside_the_editable_set_are_not_read(self, repo: Path) -> None:
        snapshot = collect_editable_files(repo, ["src/"], _head(repo))

        assert "benchmarks/evaluate.py" not in snapshot.paths

    def test_no_editable_paths_reads_nothing(self, repo: Path) -> None:
        snapshot = collect_editable_files(repo, [], _head(repo))

        assert snapshot.files == []

    def test_content_is_the_commit_not_the_working_tree(self, repo: Path) -> None:
        """A patch applies at the baseline, so that is what must be shown."""
        commit = _head(repo)
        (repo / "src" / "config.py").write_text("IMG_SIZE = 9999\n", encoding="utf-8")

        snapshot = collect_editable_files(repo, ["src/"], commit)

        assert snapshot.content_of("src/config.py") == "IMG_SIZE = 640\n"

    def test_a_missing_commit_degrades_to_empty(self, repo: Path) -> None:
        snapshot = collect_editable_files(repo, ["src/"], "0" * 40)

        assert snapshot.files == []


IMG_SIZE_PATCH = """\
diff --git a/src/config.py b/src/config.py
--- a/src/config.py
+++ b/src/config.py
@@ -1 +1 @@
-IMG_SIZE = 640
+IMG_SIZE = 512
"""

NEW_FILE_PATCH = """\
diff --git a/src/tuning.py b/src/tuning.py
new file mode 100644
--- /dev/null
+++ b/src/tuning.py
@@ -0,0 +1 @@
+WARMUP = 3
"""


class TestReplayingALineage:
    """A compounding plan is applied after its ancestors, so that is what it must see."""

    def test_the_chain_is_what_the_next_patch_will_meet(self, repo: Path) -> None:
        after = content_after(repo, _head(repo), [IMG_SIZE_PATCH])

        assert after == {"src/config.py": "IMG_SIZE = 512\n"}

    def test_a_file_the_chain_created_is_part_of_the_state(self, repo: Path) -> None:
        after = content_after(repo, _head(repo), [IMG_SIZE_PATCH, NEW_FILE_PATCH])

        assert after["src/tuning.py"] == "WARMUP = 3\n"

    def test_no_chain_reads_nothing(self, repo: Path) -> None:
        assert content_after(repo, _head(repo), []) == {}

    def test_a_chain_that_does_not_compose_degrades_to_the_commit(self, repo: Path) -> None:
        """Import reports this properly; here it must not take the planner down."""
        assert content_after(repo, _head(repo), [IMG_SIZE_PATCH, IMG_SIZE_PATCH]) == {}

    def test_the_scratch_checkout_does_not_outlive_the_call(self, repo: Path) -> None:
        content_after(repo, _head(repo), [IMG_SIZE_PATCH])

        assert not (repo / ".researchforge" / "worktrees" / "plan-context").exists()


class TestOverlaidContent:
    def test_the_overlay_replaces_what_the_commit_holds(self, repo: Path) -> None:
        snapshot = collect_editable_files(
            repo, ["src/"], _head(repo), overlay={"src/config.py": "IMG_SIZE = 512\n"}
        )

        assert snapshot.content_of("src/config.py") == "IMG_SIZE = 512\n"
        assert snapshot.content_of("src/model.py") == "def run():\n    return 1\n"

    def test_a_file_the_lineage_created_is_shown_too(self, repo: Path) -> None:
        snapshot = collect_editable_files(
            repo, ["src/"], _head(repo), overlay={"src/tuning.py": "WARMUP = 3\n"}
        )

        assert "src/tuning.py" in snapshot.paths

    def test_the_snapshot_names_the_experiments_it_already_contains(self, repo: Path) -> None:
        snapshot = collect_editable_files(
            repo,
            ["src/"],
            _head(repo),
            overlay={"src/config.py": "IMG_SIZE = 512\n"},
            applied=["exp-001", "exp-008"],
        )

        assert snapshot.applied == ["exp-001", "exp-008"]


class TestBinaryAndBudget:
    def test_binary_files_are_listed_but_not_read(self, repo: Path) -> None:
        (repo / "src" / "weights.bin").write_bytes(b"\x00\x01\x02binary")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "weights", "--no-gpg-sign")

        snapshot = collect_editable_files(repo, ["src/"], _head(repo))

        assert "src/weights.bin" not in snapshot.paths
        omitted = snapshot.was_omitted("src/weights.bin")
        assert omitted is not None and "text" in omitted.reason

    def test_a_file_over_the_per_file_budget_is_listed_not_truncated(self, repo: Path) -> None:
        (repo / "src" / "big.py").write_text("x = 1\n" * 5000, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "big", "--no-gpg-sign")

        snapshot = collect_editable_files(repo, ["src/"], _head(repo), max_file_bytes=100)

        assert "src/big.py" not in snapshot.paths
        assert snapshot.was_omitted("src/big.py") is not None

    def test_the_total_budget_keeps_the_files_the_hypothesis_names(self, repo: Path) -> None:
        snapshot = collect_editable_files(
            repo,
            ["src/"],
            _head(repo),
            prioritize="Raise the input resolution in config",
            max_total_bytes=20,
        )

        assert snapshot.paths == ["src/config.py"]
        assert snapshot.was_omitted("src/model.py") is not None
