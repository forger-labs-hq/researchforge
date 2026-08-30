"""Tests for the experiment-plan handshake: context export, import, approval."""

import json
from contextlib import closing
from pathlib import Path

from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.experiments.context_export import ExperimentPlanArtifact
from researchforge.storage.db import open_project_db

IMPROVING_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
new file mode 100644
--- /dev/null
+++ b/src/algo.py
@@ -0,0 +1 @@
+IMPROVEMENT = 1
"""

PROTECTED_PATCH = """\
diff --git a/benchmarks/helper.py b/benchmarks/helper.py
new file mode 100644
--- /dev/null
+++ b/benchmarks/helper.py
@@ -0,0 +1 @@
+CHEAT = True
"""

OUTSIDE_EDITABLE_PATCH = """\
diff --git a/docs/notes.md b/docs/notes.md
new file mode 100644
--- /dev/null
+++ b/docs/notes.md
@@ -0,0 +1 @@
+notes
"""

BROKEN_PATCH = """\
diff --git a/src/missing.py b/src/missing.py
--- a/src/missing.py
+++ b/src/missing.py
@@ -1 +1 @@
-does not exist
+replacement
"""


def _stage_plan(
    base: Path,
    entries: list[tuple[str, str]],
    hypothesis_id: str = "hyp-001",
) -> Path:
    """Write plan.yaml + patches into the staging dir; returns the plan path."""
    staging = base / ".researchforge" / "experiments"
    patches = staging / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    lines = [
        f"hypothesis_id: {hypothesis_id}",
        "approach_summary: Try caching variants.",
        "experiments:",
    ]
    for key, patch_text in entries:
        (patches / f"{key}.patch").write_text(patch_text, encoding="utf-8")
        lines += [
            f"  - key: {key}",
            f"    title: Variant {key}",
            f"    change_summary: Change for {key}.",
            f"    patch_file: patches/{key}.patch",
        ]
    plan = staging / "plan.yaml"
    plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


class TestExperimentPlanContext:
    def test_context_written_with_schema_and_instructions(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(app, ["experiment", "plan", "hyp-001"])

        assert result.exit_code == 0, result.output
        context_path = isolated_project_dir / ".researchforge" / "experiments" / "context.json"
        assert context_path.is_file()
        context = json.loads(context_path.read_text(encoding="utf-8"))
        assert context["hypothesis"]["hypothesis_id"] == "hyp-001"
        assert context["baseline"]["primary_metric"]["name"] == "f1"
        assert ".researchforge/" in context["contract"]["protected_paths"]
        assert context["expected_artifacts"]["plan_schema"] == (
            ExperimentPlanArtifact.model_json_schema()
        )
        joined = " ".join(context["instructions"]).lower()
        assert "unified diff" in joined
        assert "untrusted" in joined

    def test_blocked_without_baseline(
        self, cli_runner: CliRunner, contracted_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["experiment", "plan", "hyp-001"])
        assert result.exit_code == 1

    def test_unknown_hypothesis(self, cli_runner: CliRunner, baselined_project: Path) -> None:
        result = cli_runner.invoke(app, ["experiment", "plan", "hyp-999"])
        assert result.exit_code == 1


class TestPlanImport:
    def test_bare_patch_file_name_resolves_to_the_patches_dir(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        """`patch_file: improve.patch` means the only place a patch may live."""
        staging = isolated_project_dir / ".researchforge" / "experiments"
        (staging / "patches").mkdir(parents=True, exist_ok=True)
        (staging / "patches" / "improve.patch").write_text(IMPROVING_PATCH, encoding="utf-8")
        plan = staging / "plan.yaml"
        plan.write_text(
            "hypothesis_id: hyp-001\n"
            "approach_summary: Try caching variants.\n"
            "experiments:\n"
            "  - key: improve\n"
            "    title: Variant improve\n"
            "    change_summary: Change for improve.\n"
            "    patch_file: improve.patch\n",
            encoding="utf-8",
        )

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 0, result.output
        assert "1 runnable" in result.output

    def test_patch_file_outside_the_patches_dir_is_refused(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        staging = isolated_project_dir / ".researchforge" / "experiments"
        (staging / "patches").mkdir(parents=True, exist_ok=True)
        (staging / "sneaky.patch").write_text(IMPROVING_PATCH, encoding="utf-8")
        plan = staging / "plan.yaml"
        plan.write_text(
            "hypothesis_id: hyp-001\n"
            "approach_summary: Try caching variants.\n"
            "experiments:\n"
            "  - key: sneaky\n"
            "    title: Variant sneaky\n"
            "    change_summary: Change for sneaky.\n"
            "    patch_file: ../sneaky.patch\n",
            encoding="utf-8",
        )

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 1
        assert "must live inside" in result.output

    def test_valid_plan_imports(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(isolated_project_dir, [("improve", IMPROVING_PATCH)])

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 0, result.output
        assert "plan-001 imported: 1 experiment(s) (1 runnable, 0 rejected)" in result.output

        shown = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-001", "--json"]).output
        )
        assert shown["changed_files"] == ["src/algo.py"]
        assert shown["status"] == "planned"
        # Patch copied into artifacts.
        artifact = (
            baselined_project
            / ".researchforge"
            / "artifacts"
            / "experiments"
            / "plan-001"
            / "exp-001"
            / "change.patch"
        )
        assert artifact.is_file()

    def test_protected_patch_recorded_as_rejected(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(
            isolated_project_dir,
            [("improve", IMPROVING_PATCH), ("cheat", PROTECTED_PATCH)],
        )

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 0, result.output
        assert "rejected" in result.output
        assert "will not run" in result.output

        listed = json.loads(cli_runner.invoke(app, ["experiment", "list", "--json"]).output)
        by_id = {e["experiment_id"]: e for e in listed["experiments"]}
        assert by_id["exp-001"]["status"] == "planned"
        assert by_id["exp-002"]["status"] == "rejected"
        assert by_id["exp-002"]["decision"]["outcome"] == "reject"
        assert "benchmarks/helper.py" in by_id["exp-002"]["decision"]["reason"]

    def test_outside_editable_rejected_too(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(
            isolated_project_dir,
            [("improve", IMPROVING_PATCH), ("docs", OUTSIDE_EDITABLE_PATCH)],
        )
        cli_runner.invoke(app, ["experiment", "import", str(plan)])

        shown = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-002", "--json"]).output
        )
        assert shown["status"] == "rejected"
        assert shown["path_violations"][0]["rule"] == "not_editable"

    def test_all_rejected_is_an_error(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(isolated_project_dir, [("cheat", PROTECTED_PATCH)])

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 1
        assert "nothing to run" in result.output
        with closing(open_project_db()) as conn:
            count = conn.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()["n"]
        assert count == 0  # transactional: nothing persisted

    def test_non_applying_patch_is_error_with_git_message(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(isolated_project_dir, [("broken", BROKEN_PATCH)])

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 1
        assert "does not apply" in result.output

    def test_patch_traversal_refused(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        staging = isolated_project_dir / ".researchforge" / "experiments"
        staging.mkdir(parents=True, exist_ok=True)
        (isolated_project_dir / "outside.patch").write_text(IMPROVING_PATCH, encoding="utf-8")
        plan = staging / "plan.yaml"
        plan.write_text(
            "hypothesis_id: hyp-001\n"
            "approach_summary: s\n"
            "experiments:\n"
            "  - key: sneaky\n"
            "    title: t\n"
            "    change_summary: c\n"
            "    patch_file: ../../outside.patch\n",
            encoding="utf-8",
        )

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 1
        assert "must live inside" in result.output

    def test_too_many_experiments_rejected(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        entries = [(f"var-{i}", IMPROVING_PATCH) for i in range(5)]  # max_experiments: 4
        plan = _stage_plan(isolated_project_dir, entries)

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 1
        assert "max_experiments" in result.output

    def test_unknown_hypothesis_rejected(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(
            isolated_project_dir, [("improve", IMPROVING_PATCH)], hypothesis_id="hyp-777"
        )
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1

    def test_reimport_creates_new_plan(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(isolated_project_dir, [("improve", IMPROVING_PATCH)])
        cli_runner.invoke(app, ["experiment", "import", str(plan)])
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert "plan-002" in result.output

    def test_json_error_payload(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(isolated_project_dir, [("broken", BROKEN_PATCH)])
        result = cli_runner.invoke(app, ["experiment", "import", str(plan), "--json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["status"] == "invalid"


class TestApproval:
    def _imported_plan(self, cli_runner: CliRunner, base: Path) -> str:
        plan = _stage_plan(base, [("improve", IMPROVING_PATCH)])
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 0, result.output
        return "plan-001"

    def test_typed_approval(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan_id = self._imported_plan(cli_runner, isolated_project_dir)

        result = cli_runner.invoke(app, ["experiment", "approve", plan_id], input="approve\n")

        assert result.exit_code == 0, result.output
        assert "worst case" in result.output

        shown = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-001", "--json"]).output
        )
        assert shown["status"] == "approved"

    def test_wrong_word_aborts(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan_id = self._imported_plan(cli_runner, isolated_project_dir)
        result = cli_runner.invoke(app, ["experiment", "approve", plan_id], input="nope\n")
        assert result.exit_code == 1

    def test_double_approval_refused(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan_id = self._imported_plan(cli_runner, isolated_project_dir)
        cli_runner.invoke(app, ["experiment", "approve", plan_id, "--yes"])
        result = cli_runner.invoke(app, ["experiment", "approve", plan_id, "--yes"])
        assert result.exit_code == 1

    def test_cancel_plan(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan_id = self._imported_plan(cli_runner, isolated_project_dir)
        result = cli_runner.invoke(app, ["experiment", "cancel", plan_id, "--yes"])

        assert result.exit_code == 0
        shown = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-001", "--json"]).output
        )
        assert shown["status"] == "cancelled"

    def test_status_next_action_progression(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        status = json.loads(cli_runner.invoke(app, ["status", "--json"]).output)
        assert "experiment plan" in status["next_action"]

        plan_id = self._imported_plan(cli_runner, isolated_project_dir)
        status = json.loads(cli_runner.invoke(app, ["status", "--json"]).output)
        assert f"experiment approve {plan_id}" in status["next_action"]

        cli_runner.invoke(app, ["experiment", "approve", plan_id, "--yes"])
        status = json.loads(cli_runner.invoke(app, ["status", "--json"]).output)
        assert f"experiment run {plan_id}" in status["next_action"]
