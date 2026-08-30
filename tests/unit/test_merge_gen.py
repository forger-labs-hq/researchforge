"""The AI-authored merged patch: prompt contents and response handling."""

import pytest

from researchforge.ai.merge_gen import (
    MAX_BRANCH_PATCH_CHARS,
    MergeBranch,
    MergeNotPossibleError,
    build_prompt,
    generate_merged_patch,
    looks_like_unified_diff,
)

DIFF = """\
diff --git a/src/algo.py b/src/algo.py
--- a/src/algo.py
+++ b/src/algo.py
@@ -1,2 +1,2 @@
-IMPROVEMENT = 1
+IMPROVEMENT = 9
 LATENCY = 90.0
"""

LEFT = MergeBranch("exp-001", "Raise improvement", "Sets IMPROVEMENT=5", DIFF)
RIGHT = MergeBranch("exp-003", "Lower latency", "Sets LATENCY=120", DIFF)


class StubProvider:
    """Returns one canned response and records what it was asked."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.system = ""
        self.user = ""

    @property
    def name(self) -> str:
        return "stub/stub"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        self.system = system
        self.user = user
        return self.response


class TestPrompt:
    def test_names_both_experiments(self) -> None:
        prompt = build_prompt(LEFT, RIGHT, ["src/"], ["benchmarks/"])
        assert "exp-001" in prompt
        assert "exp-003" in prompt

    def test_includes_both_change_summaries(self) -> None:
        prompt = build_prompt(LEFT, RIGHT, ["src/"], ["benchmarks/"])
        assert "Sets IMPROVEMENT=5" in prompt
        assert "Sets LATENCY=120" in prompt

    def test_includes_both_diffs(self) -> None:
        prompt = build_prompt(LEFT, RIGHT, ["src/"], ["benchmarks/"])
        assert prompt.count("IMPROVEMENT = 9") == 2

    def test_states_the_permission_boundary(self) -> None:
        prompt = build_prompt(LEFT, RIGHT, ["src/"], ["benchmarks/"])
        assert "src/" in prompt
        assert "benchmarks/" in prompt

    def test_a_huge_patch_is_truncated(self) -> None:
        huge = MergeBranch("exp-009", "Big", "Rewrites everything", "z" * 40_000)
        prompt = build_prompt(huge, RIGHT, ["src/"], ["benchmarks/"])
        assert prompt.count("z") == MAX_BRANCH_PATCH_CHARS

    def test_the_system_prompt_forbids_trusting_the_diffs(self) -> None:
        provider = StubProvider(f"<patch>\n{DIFF}</patch>")
        generate_merged_patch(LEFT, RIGHT, ["src/"], [], provider)
        assert "untrusted" in provider.system


class TestResponseHandling:
    def test_returns_the_tagged_diff(self) -> None:
        provider = StubProvider(f"Here you go.\n<patch>\n{DIFF}</patch>\nDone.")
        assert generate_merged_patch(LEFT, RIGHT, ["src/"], [], provider) == DIFF

    def test_strips_a_markdown_fence(self) -> None:
        provider = StubProvider(f"<patch>\n```diff\n{DIFF}```\n</patch>")
        assert generate_merged_patch(LEFT, RIGHT, ["src/"], [], provider) == DIFF

    def test_a_missing_trailing_newline_is_added(self) -> None:
        provider = StubProvider(f"<patch>{DIFF.rstrip()}</patch>")
        result = generate_merged_patch(LEFT, RIGHT, ["src/"], [], provider)
        assert result.endswith("LATENCY = 90.0\n")

    def test_an_impossible_combination_is_reported(self) -> None:
        provider = StubProvider("<impossible>Both set EPOCHS to different values.</impossible>")
        with pytest.raises(MergeNotPossibleError, match="different values"):
            generate_merged_patch(LEFT, RIGHT, ["src/"], [], provider)

    def test_a_response_without_tags_is_refused(self) -> None:
        provider = StubProvider("I combined them for you, trust me.")
        with pytest.raises(MergeNotPossibleError, match="no <patch>"):
            generate_merged_patch(LEFT, RIGHT, ["src/"], [], provider)

    def test_prose_inside_the_patch_tag_is_refused(self) -> None:
        provider = StubProvider("<patch>Change IMPROVEMENT to 9 in src/algo.py.</patch>")
        with pytest.raises(MergeNotPossibleError, match="not a unified diff"):
            generate_merged_patch(LEFT, RIGHT, ["src/"], [], provider)

    def test_an_empty_patch_tag_is_refused(self) -> None:
        provider = StubProvider("<patch></patch>")
        with pytest.raises(MergeNotPossibleError, match="no <patch>"):
            generate_merged_patch(LEFT, RIGHT, ["src/"], [], provider)


class TestDiffSniffing:
    def test_a_real_diff_passes(self) -> None:
        assert looks_like_unified_diff(DIFF) is True

    def test_prose_fails(self) -> None:
        assert looks_like_unified_diff("Edit src/algo.py and set IMPROVEMENT = 9") is False

    def test_a_header_without_hunks_fails(self) -> None:
        assert looks_like_unified_diff("diff --git a/src/algo.py b/src/algo.py\n") is False
