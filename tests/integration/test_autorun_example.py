"""Three autorun rounds against examples/simple-python, end to end.

Only the AI is replaced. Each round's "planning" writes a real plan.yaml and a
real patch, which then go through the actual importer, the actual git worktree
executor, the actual constraint checks and the actual ranking — against the demo
repository shipped with the project rather than a purpose-built fixture. What
this pins down is that the loop compounds on its own winner and stops when the
measurements stop improving, using numbers the example genuinely produces:

    baseline                     f1 = 0.75   p95 = 72ms
    round 1, normalize tokens    f1 = 0.90   p95 = 72ms
    round 2, + two keywords      f1 = 1.00   p95 = 72ms
    round 3, + an unused keyword f1 = 1.00   (no improvement, so the loop ends)
"""

import shutil
import sqlite3
import subprocess
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.autorun import engine
from researchforge.autorun.engine import AutorunConfig, run_autorun
from researchforge.config.paths import research_log_path
from researchforge.domain.hypothesis import Hypothesis, Level, NoveltyConfidence
from researchforge.experiments.importers import import_experiment_plan
from researchforge.storage.db import open_project_db
from researchforge.storage.experiment_repository import list_experiments
from researchforge.storage.hypothesis_repository import list_hypotheses, replace_hypotheses
from researchforge.storage.project_repository import get_project

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "simple-python"

NORMALIZE_PATCH = '''\
diff --git a/src/config.py b/src/config.py
--- a/src/config.py
+++ b/src/config.py
@@ -1,4 +1,4 @@
 """Tunable classifier settings — experiments patch this file."""
 
-NORMALIZE = False
+NORMALIZE = True
 NGRAM_EXPANSION = False
'''

KEYWORDS_PATCH = """\
diff --git a/src/classifier.py b/src/classifier.py
--- a/src/classifier.py
+++ b/src/classifier.py
@@ -22,6 +22,8 @@ KEYWORDS = {
     "terrible": -2,
     "hate": -2,
     "awful": -2,
+    "solid": 3,
+    "disappointing": -2,
 }
 
 BIGRAMS = {
"""

UNUSED_KEYWORD_PATCH = """\
diff --git a/src/classifier.py b/src/classifier.py
--- a/src/classifier.py
+++ b/src/classifier.py
@@ -24,6 +24,7 @@ KEYWORDS = {
     "awful": -2,
     "solid": 3,
     "disappointing": -2,
+    "mediocre": -1,
 }
 
 BIGRAMS = {
"""

ROUNDS = [
    ("normalize", "Normalize tokens before matching keywords.", NORMALIZE_PATCH),
    ("keywords", "Add two sentiment keywords the lexicon was missing.", KEYWORDS_PATCH),
    ("unused", "Add a keyword no sample contains.", UNUSED_KEYWORD_PATCH),
]


def _hypothesis(hypothesis_id: str, title: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        title=title,
        claim=f"{title} raises sentiment F1.",
        rationale="The classifier misses keywords it should match.",
        feasibility=Level.HIGH,
        estimated_effort=Level.LOW,
        novelty_confidence=NoveltyConfidence.UNKNOWN,
        proposed_experiment="Patch the classifier and benchmark it.",
    )


class ScriptedAI:
    """Plans one pre-written variant per round and invents the next hypothesis."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.round = 0
        self.parents_seen: list[str | None] = []
        self.logs_seen: list[str] = []

    def plan_all(
        self,
        conn: sqlite3.Connection,
        hypotheses: list[Hypothesis],
        provider_hint: str | None,
        model_hint: str | None,
        parent_experiment_id: str | None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        key, summary, patch = ROUNDS[min(self.round, len(ROUNDS) - 1)]
        self.round += 1
        self.parents_seen.append(parent_experiment_id)

        staging = self.project_dir / ".researchforge" / "experiments"
        patches = staging / "patches"
        patches.mkdir(parents=True, exist_ok=True)
        (patches / f"{key}.patch").write_text(patch, encoding="utf-8")

        plan_ids = []
        for hypothesis in hypotheses:
            plan_file = staging / "plan.yaml"
            plan_file.write_text(
                f"hypothesis_id: {hypothesis.hypothesis_id}\n"
                f"approach_summary: {summary}\n"
                "experiments:\n"
                f"  - key: {key}\n"
                f"    title: Variant {key}\n"
                f"    change_summary: {summary}\n"
                f"    patch_file: patches/{key}.patch\n"
                + (
                    f"    parent: {parent_experiment_id}\n"
                    if parent_experiment_id is not None
                    else ""
                ),
                encoding="utf-8",
            )
            result, plan = import_experiment_plan(conn, plan_file)
            assert result.ok, result.errors
            assert plan is not None
            plan_ids.append(plan.plan_id)
        return plan_ids

    def resynthesize(
        self,
        conn: sqlite3.Connection,
        provider_hint: str | None,
        model_hint: str | None,
        log_content: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        self.logs_seen.append(log_content)
        project = get_project(conn)
        assert project is not None
        existing = list_hypotheses(conn)
        new_id = f"hyp-{len(existing) + 1:03d}"
        replace_hypotheses(
            conn, project.id, [*existing, _hypothesis(new_id, f"Follow-up {new_id}")]
        )
        return [new_id]


@pytest.fixture
def example_loop(
    cli_runner: CliRunner, isolated_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ScriptedAI]:
    """examples/simple-python, contracted and baselined, with the AI scripted."""
    from researchforge.cli import app

    runner = cli_runner
    repo = isolated_project_dir / "demo"
    shutil.copytree(EXAMPLE, repo, ignore=shutil.ignore_patterns("__pycache__", "artifacts"))
    subprocess.run(["git", "init", "-qb", "main"], cwd=repo, check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Demo"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "baseline"], check=True)

    create = runner.invoke(
        app,
        [
            "project",
            "create",
            "--mode",
            "improve_repository",
            "--objective",
            "Improve sentiment classification F1 without exceeding the latency budget",
        ],
    )
    assert create.exit_code == 0, create.output
    assert runner.invoke(app, ["repo", "scan", str(repo)]).exit_code == 0

    shutil.copy(repo / "researchforge.example.yaml", repo / "researchforge.yaml")
    assert runner.invoke(app, ["contract", "approve", "--yes"]).exit_code == 0

    with closing(open_project_db()) as conn:
        project = get_project(conn)
        assert project is not None
        replace_hypotheses(
            conn, project.id, [_hypothesis("hyp-001", "Token normalization improves F1")]
        )

    assert runner.invoke(app, ["baseline", "run"]).exit_code == 0

    ai = ScriptedAI(isolated_project_dir)
    monkeypatch.setattr(engine, "plan_all_hypotheses", ai.plan_all)
    monkeypatch.setattr(engine, "resynthesize_hypotheses", ai.resynthesize)
    yield ai


def _run() -> engine.AutorunResult:
    with closing(open_project_db()) as conn:
        return run_autorun(conn, AutorunConfig(global_stall=1, compound=True, yes=True))


class TestThreeRoundsOnTheExample:
    def test_the_loop_runs_three_rounds_and_stops_when_it_stops_improving(
        self, example_loop: ScriptedAI
    ) -> None:
        result = _run()

        assert len(result.rounds) == 3
        assert [r.improved_over_previous for r in result.rounds] == [True, True, False]
        assert "global stall" in result.stop_reason

    def test_each_round_measures_what_the_example_actually_produces(
        self, example_loop: ScriptedAI
    ) -> None:
        result = _run()

        assert result.baseline_value == pytest.approx(0.75)
        assert result.rounds[0].best_metric_value == pytest.approx(0.90)
        assert result.rounds[1].best_metric_value == pytest.approx(1.00)
        assert result.rounds[2].best_metric_value == pytest.approx(1.00)

    def test_the_best_result_is_the_compounded_one(self, example_loop: ScriptedAI) -> None:
        result = _run()

        assert result.best_metric_value == pytest.approx(1.00)
        assert result.metric_name == "f1"
        assert result.total_experiments == 3

    def test_each_round_builds_on_the_previous_winner(self, example_loop: ScriptedAI) -> None:
        _run()

        # Round 1 has nothing to build on; rounds 2 and 3 name a parent, which is
        # what makes the keyword patch apply on top of the normalize patch.
        assert example_loop.parents_seen[0] is None
        assert example_loop.parents_seen[1] == "exp-001"
        assert example_loop.parents_seen[2] == "exp-002"

    def test_the_compounded_experiment_records_its_lineage(
        self, example_loop: ScriptedAI
    ) -> None:
        _run()

        with closing(open_project_db()) as conn:
            experiments = {e.experiment_id: e for e in list_experiments(conn)}
        assert experiments["exp-001"].parent_experiment_ids == []
        assert experiments["exp-002"].parent_experiment_ids == ["exp-001"]
        assert experiments["exp-003"].parent_experiment_ids == ["exp-002"]

    def test_every_experiment_stayed_within_the_latency_constraint(
        self, example_loop: ScriptedAI
    ) -> None:
        from researchforge.storage.experiment_repository import list_executions

        _run()

        with closing(open_project_db()) as conn:
            executions = list_executions(conn)
        checked = [e for e in executions if e.constraints]
        assert checked, "the contract sets a latency constraint, so it should be checked"
        for execution in checked:
            assert all(c.passed for c in execution.constraints), execution.experiment_id

    def test_the_research_log_records_each_round_for_the_next_one(
        self, example_loop: ScriptedAI
    ) -> None:
        _run()

        log = research_log_path(Path.cwd()).read_text(encoding="utf-8")
        assert "Round 1" in log
        assert "Round 2" in log
        assert "0.9" in log

    def test_the_ai_is_shown_the_log_before_re_synthesizing(
        self, example_loop: ScriptedAI
    ) -> None:
        _run()

        assert example_loop.logs_seen, "re-synthesis should be given the running log"
        assert "Round 1" in example_loop.logs_seen[-1]

    def test_the_loop_is_resumable_from_its_persisted_state(
        self, example_loop: ScriptedAI
    ) -> None:
        from researchforge.autorun.state import load_state

        _run()

        state = load_state(Path.cwd())
        assert state is not None
        assert len(state.rounds) == 3
        assert state.best_experiment_id == "exp-002"
