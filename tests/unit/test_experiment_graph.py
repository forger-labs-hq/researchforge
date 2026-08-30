"""The experiment DAG: parent chains and merges through import, execution, shipping."""

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from researchforge.cli import app

PARENT_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
new file mode 100644
--- /dev/null
+++ b/src/algo.py
@@ -0,0 +1,2 @@
+IMPROVEMENT = 5
+LATENCY = 150.0
"""

# Written against the PARENT's state (src/algo.py exists with IMPROVEMENT=5):
# only applies when the chain was applied first.
CHILD_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
--- a/src/algo.py
+++ b/src/algo.py
@@ -1,2 +1,2 @@
-IMPROVEMENT = 5
+IMPROVEMENT = 7
 LATENCY = 150.0
"""


# A merge needs two branches whose diffs do not overlap, so the shared root
# writes a padded file: one branch edits the top, the other the bottom, and
# neither appears in the other's context lines.
MERGE_ROOT_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
new file mode 100644
--- /dev/null
+++ b/src/algo.py
@@ -0,0 +1,9 @@
+IMPROVEMENT = 1
+PAD_A = 0
+PAD_B = 0
+PAD_C = 0
+PAD_D = 0
+PAD_E = 0
+PAD_F = 0
+PAD_G = 0
+LATENCY = 150.0
"""

# Lifts f1 to 0.85; leaves latency alone.
ACCURACY_BRANCH_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
--- a/src/algo.py
+++ b/src/algo.py
@@ -1,4 +1,4 @@
-IMPROVEMENT = 1
+IMPROVEMENT = 5
 PAD_A = 0
 PAD_B = 0
 PAD_C = 0
"""

# Drops latency to 120; leaves f1 alone.
LATENCY_BRANCH_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
--- a/src/algo.py
+++ b/src/algo.py
@@ -6,4 +6,4 @@
 PAD_E = 0
 PAD_F = 0
 PAD_G = 0
-LATENCY = 150.0
+LATENCY = 120.0
"""


def _stage_merge_plan(base: Path, merge_parents: str = "[accuracy, latency]") -> Path:
    """A diamond: root → {accuracy, latency} → merge (no patch of its own)."""
    staging = base / ".researchforge" / "experiments"
    patches = staging / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    (patches / "root.patch").write_text(MERGE_ROOT_PATCH, encoding="utf-8")
    (patches / "accuracy.patch").write_text(ACCURACY_BRANCH_PATCH, encoding="utf-8")
    (patches / "latency.patch").write_text(LATENCY_BRANCH_PATCH, encoding="utf-8")
    plan = staging / "plan.yaml"
    plan.write_text(
        "hypothesis_id: hyp-001\n"
        "approach_summary: Two independent wins, then combine them.\n"
        "experiments:\n"
        "  - {key: root, title: Root knobs, change_summary: r, "
        "patch_file: patches/root.patch}\n"
        "  - {key: accuracy, title: Raise improvement, change_summary: a, "
        "patch_file: patches/accuracy.patch, parent: root}\n"
        "  - {key: latency, title: Lower latency, change_summary: l, "
        "patch_file: patches/latency.patch, parent: root}\n"
        "  - {key: combined, title: Combine both wins, change_summary: c, "
        f"parents: {merge_parents}}}\n",
        encoding="utf-8",
    )
    return plan


# A re-authored merge: written against the BASELINE, where src/algo.py does
# not exist yet, and already carrying both branches' values. Applying either
# ancestor patch on top of it would fail, so a passing run proves the executor
# left the chain alone.
AUTHORED_MERGE_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
new file mode 100644
--- /dev/null
+++ b/src/algo.py
@@ -0,0 +1,2 @@
+IMPROVEMENT = 9
+LATENCY = 90.0
"""

# Written against the authored merge's state, so it only applies when the walk
# stopped at the merge instead of climbing to its parents.
AUTHORED_MERGE_CHILD_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
--- a/src/algo.py
+++ b/src/algo.py
@@ -1,2 +1,2 @@
-IMPROVEMENT = 9
+IMPROVEMENT = 11
 LATENCY = 90.0
"""


def _run_diamond(cli_runner: CliRunner, base: Path) -> None:
    """Measure root → {accuracy, latency} → merge so exp-001..004 exist."""
    plan = _stage_merge_plan(base)
    assert cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"]).exit_code == 0


def _stage_authored_merge(base: Path, entries: str) -> Path:
    """A second plan whose merge carries its parents' changes in one patch."""
    staging = base / ".researchforge" / "experiments"
    patches = staging / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    (patches / "combined.patch").write_text(AUTHORED_MERGE_PATCH, encoding="utf-8")
    (patches / "refine.patch").write_text(AUTHORED_MERGE_CHILD_PATCH, encoding="utf-8")
    plan = staging / "plan.yaml"
    plan.write_text(
        "hypothesis_id: hyp-001\n"
        "approach_summary: Re-author the overlapping combination.\n"
        f"experiments:\n{entries}",
        encoding="utf-8",
    )
    return plan


COMBINED_ENTRY = (
    "  - {key: combined, title: Combine both wins, change_summary: c, "
    "patch_file: patches/combined.patch, parents: [exp-002, exp-003], "
    "patch_includes_parents: true}\n"
)


def _stage_tree_plan(base: Path, child_parent: str = "root") -> Path:
    staging = base / ".researchforge" / "experiments"
    patches = staging / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    (patches / "root.patch").write_text(PARENT_PATCH, encoding="utf-8")
    (patches / "child.patch").write_text(CHILD_PATCH, encoding="utf-8")
    plan = staging / "plan.yaml"
    plan.write_text(
        "hypothesis_id: hyp-001\n"
        "approach_summary: Branching knobs.\n"
        "experiments:\n"
        "  - key: root\n"
        "    title: Root improvement\n"
        "    change_summary: Set knobs.\n"
        "    patch_file: patches/root.patch\n"
        "  - key: child\n"
        "    title: Child refinement\n"
        "    change_summary: Bump improvement on top of root.\n"
        "    patch_file: patches/child.patch\n"
        f"    parent: {child_parent}\n",
        encoding="utf-8",
    )
    return plan


class TestImportValidation:
    def test_unknown_parent(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_tree_plan(isolated_project_dir, child_parent="exp-999")
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "unknown parent experiment" in result.output

    def test_same_plan_cycle(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        staging = isolated_project_dir / ".researchforge" / "experiments"
        patches = staging / "patches"
        patches.mkdir(parents=True, exist_ok=True)
        (patches / "a.patch").write_text(PARENT_PATCH, encoding="utf-8")
        (patches / "b.patch").write_text(CHILD_PATCH, encoding="utf-8")
        plan = staging / "plan.yaml"
        plan.write_text(
            "hypothesis_id: hyp-001\napproach_summary: Cycle.\nexperiments:\n"
            "  - {key: a, title: A, change_summary: a, patch_file: patches/a.patch, parent: b}\n"
            "  - {key: b, title: B, change_summary: b, patch_file: patches/b.patch, parent: a}\n",
            encoding="utf-8",
        )
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "cycle" in result.output

    def test_child_patch_must_apply_on_chain(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        """A child written against the wrong parent state is an import error."""
        staging = isolated_project_dir / ".researchforge" / "experiments"
        patches = staging / "patches"
        patches.mkdir(parents=True, exist_ok=True)
        (patches / "root.patch").write_text(PARENT_PATCH, encoding="utf-8")
        # Child expects IMPROVEMENT = 6, but the parent sets 5 -> won't apply.
        (patches / "child.patch").write_text(
            CHILD_PATCH.replace("-IMPROVEMENT = 5", "-IMPROVEMENT = 6"), encoding="utf-8"
        )
        plan = staging / "plan.yaml"
        plan.write_text(
            "hypothesis_id: hyp-001\napproach_summary: Bad chain.\nexperiments:\n"
            "  - {key: root, title: R, change_summary: r, patch_file: patches/root.patch}\n"
            "  - {key: child, title: C, change_summary: c, patch_file: patches/child.patch, "
            "parent: root}\n",
            encoding="utf-8",
        )
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "does not apply on top of its ancestor chain" in result.output

    def test_unmeasured_db_parent_refused(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        # First import creates exp-001/exp-002 in `planned` state (never run).
        plan = _stage_tree_plan(isolated_project_dir)
        assert cli_runner.invoke(app, ["experiment", "import", str(plan)]).exit_code == 0
        # Second plan branching on the unmeasured exp-001 must be refused.
        second = _stage_tree_plan(isolated_project_dir, child_parent="exp-001")
        result = cli_runner.invoke(app, ["experiment", "import", str(second)])
        assert result.exit_code == 1
        assert "only measured experiments" in result.output


class TestMergeImport:
    def test_merge_records_both_parents(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_merge_plan(isolated_project_dir)

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 0, result.output
        listed = json.loads(cli_runner.invoke(app, ["experiment", "list", "--json"]).output)
        rows = {row["experiment_id"]: row for row in listed["experiments"]}
        assert rows["exp-004"]["parent_experiment_ids"] == ["exp-002", "exp-003"]
        assert rows["exp-004"]["is_merge"] is True
        assert rows["exp-002"]["is_merge"] is False

    def test_merge_inherits_its_ancestors_changed_files(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_merge_plan(isolated_project_dir)
        assert cli_runner.invoke(app, ["experiment", "import", str(plan)]).exit_code == 0

        shown = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-004", "--json"]).output
        )
        assert shown["patch_text"] == ""
        assert shown["changed_files"] == ["src/algo.py"]
        assert shown["status"] == "planned"

    def test_single_parent_still_reads_as_one_id(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_tree_plan(isolated_project_dir)
        assert cli_runner.invoke(app, ["experiment", "import", str(plan)]).exit_code == 0

        shown = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-002", "--json"]).output
        )
        assert shown["parent_experiment_ids"] == ["exp-001"]
        assert shown["parent_experiment_id"] == "exp-001"
        assert shown["is_merge"] is False

    def test_conflicting_branches_refused_at_import(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        """Two branches editing the same line cannot be composed — say so now."""
        staging = isolated_project_dir / ".researchforge" / "experiments"
        patches = staging / "patches"
        patches.mkdir(parents=True, exist_ok=True)
        (patches / "root.patch").write_text(MERGE_ROOT_PATCH, encoding="utf-8")
        (patches / "accuracy.patch").write_text(ACCURACY_BRANCH_PATCH, encoding="utf-8")
        (patches / "rival.patch").write_text(
            ACCURACY_BRANCH_PATCH.replace("+IMPROVEMENT = 5", "+IMPROVEMENT = 9"),
            encoding="utf-8",
        )
        plan = staging / "plan.yaml"
        plan.write_text(
            "hypothesis_id: hyp-001\napproach_summary: Rival edits.\nexperiments:\n"
            "  - {key: root, title: R, change_summary: r, patch_file: patches/root.patch}\n"
            "  - {key: accuracy, title: A, change_summary: a, "
            "patch_file: patches/accuracy.patch, parent: root}\n"
            "  - {key: rival, title: V, change_summary: v, "
            "patch_file: patches/rival.patch, parent: root}\n"
            "  - {key: combined, title: C, change_summary: c, parents: [accuracy, rival]}\n",
            encoding="utf-8",
        )

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 1
        assert "merge conflict" in result.output

    def test_merge_on_unknown_parent(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_merge_plan(isolated_project_dir, merge_parents="[accuracy, exp-999]")
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "unknown parent experiment" in result.output

    def test_merge_cycle_refused(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_merge_plan(isolated_project_dir, merge_parents="[accuracy, combined]")
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "cycle" in result.output

    def test_both_parent_fields_refused(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        staging = isolated_project_dir / ".researchforge" / "experiments"
        patches = staging / "patches"
        patches.mkdir(parents=True, exist_ok=True)
        (patches / "root.patch").write_text(MERGE_ROOT_PATCH, encoding="utf-8")
        (patches / "child.patch").write_text(ACCURACY_BRANCH_PATCH, encoding="utf-8")
        plan = staging / "plan.yaml"
        plan.write_text(
            "hypothesis_id: hyp-001\napproach_summary: Ambiguous ancestry.\nexperiments:\n"
            "  - {key: root, title: R, change_summary: r, patch_file: patches/root.patch}\n"
            "  - {key: child, title: C, change_summary: c, "
            "patch_file: patches/child.patch, parent: root, parents: [root]}\n",
            encoding="utf-8",
        )

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])

        assert result.exit_code == 1
        assert "either `parent` or `parents`" in result.output

    def test_patchless_entry_needs_two_parents(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        """Only a merge may omit both patch_file and env_overrides."""
        plan = _stage_merge_plan(isolated_project_dir, merge_parents="[accuracy]")
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "must provide either patch_file or env_overrides" in result.output


class TestMergeExecution:
    def test_merge_measures_both_branches_combined(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_merge_plan(isolated_project_dir)

        result = cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"])
        assert result.exit_code == 0, result.output

        report = json.loads(
            cli_runner.invoke(app, ["results", "show", "run-001", "--json"]).output
        )
        rows = {row["experiment_id"]: row for row in report["candidates"]}
        # accuracy alone: f1 0.85 at 150ms. latency alone: f1 0.81 at 120ms.
        # The merge shows accuracy's f1 AND latency's ms — only reachable if
        # both ancestor patches landed in the same worktree.
        assert rows["exp-002"]["primary_value"] == 0.85
        assert rows["exp-003"]["primary_value"] == 0.81
        assert rows["exp-004"]["primary_value"] == 0.85
        assert rows["exp-004"]["secondary_values"]["p95_latency_ms"] == 120.0
        assert rows["exp-002"]["secondary_values"]["p95_latency_ms"] == 150.0

    def test_shared_root_applied_once(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        """The diamond's root is reachable twice but must not be applied twice."""
        plan = _stage_merge_plan(isolated_project_dir)
        assert cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"]).exit_code == 0

        executions = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-004", "--json"]).output
        )
        assert executions["status"] != "failed_setup"

    def test_ship_merge_composes_every_ancestor(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_merge_plan(isolated_project_dir)
        assert cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"]).exit_code == 0
        assert (
            cli_runner.invoke(app, ["validate", "run-001", "-e", "exp-004", "--yes"]).exit_code
            == 0
        )

        ship = cli_runner.invoke(app, ["ship", "branch", "exp-004", "--yes", "--json"])
        assert ship.exit_code == 0, ship.output
        branch = json.loads(ship.output)["branch"]

        content = subprocess.run(
            ["git", "-C", str(funnel_project), "show", f"{branch}:src/algo.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "IMPROVEMENT = 5" in content.stdout
        assert "LATENCY = 120.0" in content.stdout


class TestAuthoredMerge:
    """A merge whose patch already contains its parents, for overlapping diffs."""

    def test_parents_are_recorded_as_lineage(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        _run_diamond(cli_runner, isolated_project_dir)
        plan = _stage_authored_merge(isolated_project_dir, COMBINED_ENTRY)

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 0, result.output

        shown = json.loads(
            cli_runner.invoke(app, ["experiment", "show", "exp-005", "--json"]).output
        )
        assert shown["parent_experiment_ids"] == ["exp-002", "exp-003"]
        assert shown["is_merge"] is True
        assert shown["patch_includes_parents"] is True

    def test_execution_applies_its_own_patch_alone(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        _run_diamond(cli_runner, isolated_project_dir)
        plan = _stage_authored_merge(isolated_project_dir, COMBINED_ENTRY)

        result = cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"])
        assert result.exit_code == 0, result.output

        report = json.loads(
            cli_runner.invoke(app, ["results", "show", "run-002", "--json"]).output
        )
        row = next(r for r in report["candidates"] if r["experiment_id"] == "exp-005")
        # IMPROVEMENT=9 and LATENCY=90 come from the authored patch only; the
        # ancestors would have set 5 and 120, and would not have applied at all.
        assert row["primary_value"] == 0.89
        assert row["secondary_values"]["p95_latency_ms"] == 90.0

    def test_a_child_composes_from_the_authored_patch(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        """The chain walk stops at the merge for its descendants too."""
        _run_diamond(cli_runner, isolated_project_dir)
        plan = _stage_authored_merge(
            isolated_project_dir,
            COMBINED_ENTRY
            + "  - {key: refine, title: Refine the combination, change_summary: r, "
            "patch_file: patches/refine.patch, parent: combined}\n",
        )

        result = cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"])
        assert result.exit_code == 0, result.output

        report = json.loads(
            cli_runner.invoke(app, ["results", "show", "run-002", "--json"]).output
        )
        values = {r["experiment_id"]: r["primary_value"] for r in report["candidates"]}
        assert values["exp-005"] == 0.89
        assert values["exp-006"] == 0.91

    def test_ship_lands_the_authored_patch_alone(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        _run_diamond(cli_runner, isolated_project_dir)
        plan = _stage_authored_merge(isolated_project_dir, COMBINED_ENTRY)
        assert cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"]).exit_code == 0
        assert (
            cli_runner.invoke(app, ["validate", "run-002", "-e", "exp-005", "--yes"]).exit_code
            == 0
        )

        ship = cli_runner.invoke(app, ["ship", "branch", "exp-005", "--yes", "--json"])
        assert ship.exit_code == 0, ship.output
        branch = json.loads(ship.output)["branch"]

        content = subprocess.run(
            ["git", "-C", str(funnel_project), "show", f"{branch}:src/algo.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "IMPROVEMENT = 9" in content.stdout
        assert "LATENCY = 90.0" in content.stdout

    def test_without_a_patch_file_it_is_refused(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_authored_merge(
            isolated_project_dir,
            "  - {key: combined, title: C, change_summary: c, "
            "parents: [exp-001, exp-002], patch_includes_parents: true}\n",
        )
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "patch_includes_parents requires patch_file" in result.output

    def test_with_a_single_parent_it_is_refused(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_authored_merge(
            isolated_project_dir,
            "  - {key: combined, title: C, change_summary: c, "
            "patch_file: patches/combined.patch, parent: exp-001, "
            "patch_includes_parents: true}\n",
        )
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "declare at least two parents" in result.output


class TestTreeExecution:
    def test_chain_applied_and_measured(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_tree_plan(isolated_project_dir)
        result = cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"])
        assert result.exit_code == 0, result.output

        listed = cli_runner.invoke(app, ["experiment", "list", "--json"])
        rows = {row["experiment_id"]: row for row in json.loads(listed.output)["experiments"]}
        child_id = next(eid for eid, row in rows.items() if row["parent_experiment_id"] is not None)
        parent_id = rows[child_id]["parent_experiment_id"]
        assert rows[parent_id]["parent_experiment_id"] is None

        results = cli_runner.invoke(app, ["results", "show", "run-001", "--json"])
        report = json.loads(results.output)
        values = {row["experiment_id"]: row["primary_value"] for row in report["candidates"]}
        # Parent measured alone (0.85); child measured with BOTH patches (0.87)
        # — 0.87 is only reachable if the ancestor chain was applied.
        assert values[parent_id] == 0.85
        assert values[child_id] == 0.87

        show = cli_runner.invoke(app, ["experiment", "show", child_id])
        assert f"Parent:     {parent_id}" in show.output

        text_results = cli_runner.invoke(app, ["results", "show", "run-001"])
        assert f"parent: {parent_id}" in text_results.output

    def test_ship_branched_winner_composes_chain(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_tree_plan(isolated_project_dir)
        assert cli_runner.invoke(app, ["experiment", "start", str(plan), "--yes"]).exit_code == 0
        listed = cli_runner.invoke(app, ["experiment", "list", "--json"])
        child_id = next(
            row["experiment_id"]
            for row in json.loads(listed.output)["experiments"]
            if row["parent_experiment_id"] is not None
        )
        assert (
            cli_runner.invoke(app, ["validate", "run-001", "-e", child_id, "--yes"]).exit_code == 0
        )

        ship = cli_runner.invoke(app, ["ship", "branch", child_id, "--yes", "--json"])
        assert ship.exit_code == 0, ship.output
        payload = json.loads(ship.output)

        # Single commit on the baseline whose content composes the chain.
        log = subprocess.run(
            ["git", "-C", str(funnel_project), "log", "--format=%H", payload["branch"]],
            capture_output=True,
            text=True,
            check=True,
        )
        assert len(log.stdout.split()) == 2  # winning commit + fixture baseline commit
        content = subprocess.run(
            ["git", "-C", str(funnel_project), "show", f"{payload['branch']}:src/algo.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "IMPROVEMENT = 7" in content.stdout  # child's edit on parent's file


class TestContextExport:
    def test_prior_experiments_in_context(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(app, ["experiment", "plan", "hyp-001", "--json"])
        assert result.exit_code == 0, result.output
        context = json.loads(result.output)
        priors = {p["experiment_id"]: p for p in context["prior_experiments"]}
        assert "exp-001" in priors and "exp-002" in priors
        assert priors["exp-001"]["primary_value"] is not None
        assert any("parent" in i for i in context["instructions"])

    def test_branching_on_rejected_parent_allowed(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        # exp-002 in the fixture is rejected (latency violator) — branching on
        # it is allowed; its patch creates src/algo.py, so the child edits it.
        staging = isolated_project_dir / ".researchforge" / "experiments"
        patches = staging / "patches"
        patches.mkdir(parents=True, exist_ok=True)
        (patches / "retry.patch").write_text(
            CHILD_PATCH.replace("-IMPROVEMENT = 5", "-IMPROVEMENT = 6").replace(
                "LATENCY = 150.0", "LATENCY = 250.0"
            ),
            encoding="utf-8",
        )
        plan = staging / "plan.yaml"
        plan.write_text(
            "hypothesis_id: hyp-001\napproach_summary: Explore around the rejection.\n"
            "experiments:\n"
            "  - {key: retry, title: Retry around rejection, change_summary: r, "
            "patch_file: patches/retry.patch, parent: exp-002}\n",
            encoding="utf-8",
        )
        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 0, result.output
