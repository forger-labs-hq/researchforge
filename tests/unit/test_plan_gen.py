"""AI plan generation: where patches land, and what the importer will accept.

Both regressions here were found by an autorun that produced nothing: every
round's plans were discarded before a single experiment ran.
"""

import json
from pathlib import Path
from typing import cast

import pytest
import yaml

from researchforge.ai import plan_gen
from researchforge.ai.plan_gen import generate_experiment_plan, write_patch_files
from researchforge.experiments.context_export import ExperimentContext
from researchforge.experiments.repo_context import OmittedFile, RepoFile, RepoSnapshot

CONFIG = "MODEL = 'yolov5su.pt'\nIMG_SIZE = 640\nLR0 = 0.01\n"

SNAPSHOT = RepoSnapshot(
    commit="a1c4d7b",
    files=[RepoFile(path="src/config.py", content=CONFIG, size_bytes=len(CONFIG))],
    omitted=[OmittedFile(path="src/weights.bin", size_bytes=99, reason="not readable as text")],
)

PATCH = """\
diff --git a/src/algo.py b/src/algo.py
--- a/src/algo.py
+++ b/src/algo.py
@@ -1 +1 @@
-IMPROVEMENT = 1
+IMPROVEMENT = 9
"""


class StubProvider:
    """Returns one canned response."""

    def __init__(self, response: str) -> None:
        self.response = response

    @property
    def name(self) -> str:
        return "stub/stub"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        return self.response


def _response(plan: str, patches: str = "{}") -> str:
    return f"<plan>\n{plan}\n</plan>\n<patches>\n{patches}\n</patches>"


class FakeContext:
    """Only what the generator reads once the prompt is stubbed out."""

    def __init__(self, repository: RepoSnapshot) -> None:
        self.repository = repository


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    repository: RepoSnapshot | None = None,
) -> tuple[str, dict[str, str]]:
    """Run the generator without building a full ExperimentContext."""
    monkeypatch.setattr(plan_gen, "_build_prompt", lambda ctx: "prompt")
    ctx = FakeContext(repository if repository is not None else SNAPSHOT)
    return generate_experiment_plan(
        cast(ExperimentContext, ctx), cast(plan_gen.AiProvider, StubProvider(response))
    )


def _files_response(plan: str, files: str) -> str:
    return f"<plan>\n{plan}\n</plan>\n<files>\n{files}\n</files>"


PLAN_ONE = (
    "hypothesis_id: hyp-001\n"
    "approach_summary: Sweep the image size.\n"
    "experiments:\n"
    "  - key: imgsz-800\n"
    "    title: Larger input\n"
    "    change_summary: IMG_SIZE 640 to 800.\n"
)


class TestWritePatchFiles:
    def test_directory_prefix_is_not_nested_twice(self, tmp_path: Path) -> None:
        """`patches/x.patch` is relative to the experiments dir, not to patches/."""
        patches_dir = tmp_path / "experiments" / "patches"

        written = write_patch_files(patches_dir, {"patches/improve.patch": PATCH})

        assert written == [patches_dir / "improve.patch"]
        assert (patches_dir / "improve.patch").read_text(encoding="utf-8") == PATCH
        assert not (patches_dir / "patches").exists()

    def test_bare_file_name_lands_in_the_patches_dir(self, tmp_path: Path) -> None:
        patches_dir = tmp_path / "experiments" / "patches"

        write_patch_files(patches_dir, {"improve.patch": PATCH})

        assert (patches_dir / "improve.patch").is_file()

    def test_traversal_in_a_key_cannot_escape(self, tmp_path: Path) -> None:
        patches_dir = tmp_path / "experiments" / "patches"

        write_patch_files(patches_dir, {"../../escaped.patch": PATCH})

        assert (patches_dir / "escaped.patch").is_file()
        assert not (tmp_path / "escaped.patch").exists()

    def test_creates_the_directory_and_returns_every_path(self, tmp_path: Path) -> None:
        patches_dir = tmp_path / "deep" / "patches"

        written = write_patch_files(
            patches_dir, {"patches/a.patch": PATCH, "b.patch": PATCH}
        )

        assert sorted(p.name for p in written) == ["a.patch", "b.patch"]


class TestUnknownTopLevelKeys:
    """An extra key at the top level must not cost a whole round of planning."""

    def test_plan_id_hint_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        plan_yaml, _ = _generate(
            monkeypatch,
            _response(
                "plan_id_hint: plan-001\n"
                "hypothesis_id: hyp-001\n"
                "approach_summary: Try it.\n"
                "experiments:\n"
                "  - key: a\n"
                "    title: A\n"
                "    change_summary: Changes a thing.\n"
                "    env_overrides: {CONF: '0.1'}\n"
            ),
        )

        parsed = yaml.safe_load(plan_yaml)
        assert "plan_id_hint" not in parsed
        assert parsed["hypothesis_id"] == "hyp-001"
        assert parsed["experiments"][0]["env_overrides"] == {"CONF": "0.1"}

    def test_a_clean_plan_is_passed_through_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = (
            "hypothesis_id: hyp-001\n"
            "approach_summary: Try it.\n"
            "experiments:\n"
            "  - key: a\n"
            "    title: A\n"
            "    change_summary: Changes a thing.\n"
            "    patch_file: patches/a.patch\n"
        )

        plan_yaml, _ = _generate(monkeypatch, _response(original))

        assert plan_yaml.strip() == original.strip()

    def test_unknown_keys_inside_an_experiment_are_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`env` instead of `env_overrides` changes what runs — it must still fail."""
        plan_yaml, _ = _generate(
            monkeypatch,
            _response(
                "plan_id_hint: plan-001\n"
                "hypothesis_id: hyp-001\n"
                "approach_summary: Try it.\n"
                "experiments:\n"
                "  - key: a\n"
                "    title: A\n"
                "    change_summary: Changes a thing.\n"
                "    env: {CONF: '0.1'}\n"
            ),
        )

        assert "env" in yaml.safe_load(plan_yaml)["experiments"][0]


class TestPatchesFromRewrittenFiles:
    """The model returns the changed file; the engine computes the diff."""

    def test_the_diff_is_generated_and_linked_to_the_experiment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        changed = CONFIG.replace("IMG_SIZE = 640", "IMG_SIZE = 800")
        plan_yaml, patches = _generate(
            monkeypatch,
            _files_response(PLAN_ONE, json.dumps({"imgsz-800": {"src/config.py": changed}})),
        )

        assert list(patches) == ["imgsz-800.patch"]
        patch = patches["imgsz-800.patch"]
        assert patch.startswith("diff --git a/src/config.py b/src/config.py")
        assert "-IMG_SIZE = 640" in patch
        assert "+IMG_SIZE = 800" in patch
        entry = yaml.safe_load(plan_yaml)["experiments"][0]
        assert entry["patch_file"] == "patches/imgsz-800.patch"

    def test_hunk_headers_come_from_the_real_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The offsets a model used to guess wrong are now git's arithmetic."""
        changed = CONFIG.replace("LR0 = 0.01", "LR0 = 0.001")
        _, patches = _generate(
            monkeypatch,
            _files_response(PLAN_ONE, json.dumps({"imgsz-800": {"src/config.py": changed}})),
        )

        assert "@@ -1,3 +1,3 @@" in patches["imgsz-800.patch"]

    def test_a_new_file_is_diffed_as_a_new_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, patches = _generate(
            monkeypatch,
            _files_response(PLAN_ONE, json.dumps({"imgsz-800": {"src/extra.py": "X = 1\n"}})),
        )

        assert "new file mode" in patches["imgsz-800.patch"]

    def test_a_stale_patch_file_the_model_wrote_is_replaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = PLAN_ONE + "    patch_file: patches/remembered.patch\n"
        changed = CONFIG.replace("IMG_SIZE = 640", "IMG_SIZE = 800")

        plan_yaml, patches = _generate(
            monkeypatch,
            _files_response(plan, json.dumps({"imgsz-800": {"src/config.py": changed}})),
        )

        assert yaml.safe_load(plan_yaml)["experiments"][0]["patch_file"] == (
            "patches/imgsz-800.patch"
        )
        assert list(patches) == ["imgsz-800.patch"]

    def test_a_patch_file_with_no_patch_behind_it_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An env-only experiment must not be left pointing at a missing patch."""
        plan = (
            "hypothesis_id: hyp-001\n"
            "approach_summary: Sweep.\n"
            "experiments:\n"
            "  - key: env-only\n"
            "    title: Env only\n"
            "    change_summary: Uses an environment variable.\n"
            "    patch_file: patches/invented.patch\n"
            "    env_overrides: {IMG_SIZE: '800'}\n"
        )

        plan_yaml, patches = _generate(monkeypatch, _files_response(plan, "{}"))

        entry = yaml.safe_load(plan_yaml)["experiments"][0]
        assert "patch_file" not in entry
        assert entry["env_overrides"] == {"IMG_SIZE": "800"}
        assert patches == {}

    def test_rewriting_a_file_that_was_not_shown_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Diffing against content we never read would produce a bogus patch."""
        with pytest.raises(ValueError, match="cannot be rewritten"):
            _generate(
                monkeypatch,
                _files_response(
                    PLAN_ONE, json.dumps({"imgsz-800": {"src/weights.bin": "nonsense"}})
                ),
            )

    def test_a_no_op_rewrite_produces_no_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        plan_yaml, patches = _generate(
            monkeypatch,
            _files_response(PLAN_ONE, json.dumps({"imgsz-800": {"src/config.py": CONFIG}})),
        )

        assert patches == {}
        assert "patch_file" not in yaml.safe_load(plan_yaml)["experiments"][0]

    def test_unknown_top_level_keys_are_still_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan_yaml, _ = _generate(
            monkeypatch, _files_response("plan_id_hint: plan-001\n" + PLAN_ONE, "{}")
        )

        assert "plan_id_hint" not in yaml.safe_load(plan_yaml)


class TestProgressReporting:
    """A minutes-long provider call has to look like work, not a hang."""

    def test_the_wait_says_what_was_sent(self) -> None:
        ctx = cast(ExperimentContext, FakeContext(SNAPSHOT))

        label = plan_gen.planning_phase_label(ctx, "anthropic/claude-opus-4-5")

        assert "anthropic/claude-opus-4-5" in label
        assert "1 file(s)" in label

    def test_the_result_says_how_many_variants_and_patches(self) -> None:
        summary = plan_gen.describe_plan(PLAN_ONE, {"imgsz-800.patch": "diff"})

        assert "1 variant(s)" in summary
        assert "1 with a code patch" in summary

    def test_a_config_only_plan_says_so(self) -> None:
        assert "config-only" in plan_gen.describe_plan(PLAN_ONE, {})


class TestResponseParsing:
    def test_patches_are_returned_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, patches = _generate(
            monkeypatch,
            _response(
                "hypothesis_id: hyp-001\napproach_summary: s\nexperiments: []\n",
                '{"patches/a.patch": "diff --git a/src/algo.py b/src/algo.py\\n"}',
            ),
        )

        assert list(patches) == ["patches/a.patch"]

    def test_missing_plan_tag_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(plan_gen, "_build_prompt", lambda ctx: "prompt")

        with pytest.raises(ValueError, match="missing <plan>"):
            generate_experiment_plan(
                cast(ExperimentContext, object()),
                cast(plan_gen.AiProvider, StubProvider("<patches>{}</patches>")),
            )
