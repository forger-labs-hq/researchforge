"""The autorun loop end to end against the deterministic knob evaluator.

The AI calls are the only thing replaced: planning and re-synthesis are stubbed
to stage real plan.yaml artifacts, which then go through the real importer, the
real worktree executor, and the real ranking — so what is exercised here is the
loop's control flow, its stopping rules, and its persistence.
"""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path

import pytest

from researchforge.ai import service as ai_service
from researchforge.ai.merge_gen import MergeNotPossibleError
from researchforge.autorun import engine
from researchforge.autorun.engine import AutorunConfig, PlanPreview, run_autorun
from researchforge.autorun.state import load_state
from researchforge.config.paths import research_log_path
from researchforge.domain.experiment import Experiment
from researchforge.domain.hypothesis import Hypothesis, Level, NoveltyConfidence
from researchforge.experiments.importers import import_experiment_plan
from researchforge.storage.db import open_project_db
from researchforge.storage.experiment_repository import get_experiment
from researchforge.storage.hypothesis_repository import list_hypotheses, replace_hypotheses
from researchforge.storage.project_repository import get_project

KNOB_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
new file mode 100644
--- /dev/null
+++ b/src/algo.py
@@ -0,0 +1,2 @@
+IMPROVEMENT = {improvement}
+LATENCY = {latency}
"""


# A child of a knob experiment edits the file its parent created, so it only
# applies when the ancestor chain was composed first.
KNOB_CHILD_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
--- a/src/algo.py
+++ b/src/algo.py
@@ -1,2 +1,2 @@
-IMPROVEMENT = {was}
+IMPROVEMENT = {improvement}
 LATENCY = 120.0
"""


def _stage_plan(
    base: Path,
    hypothesis_id: str,
    key: str,
    improvement: int,
    parent: str | None = None,
    parent_improvement: int = 0,
) -> Path:
    """Write a one-variant plan.yaml + patch into the handshake staging dir."""
    staging = base / ".researchforge" / "experiments"
    patches = staging / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    patch = (
        KNOB_CHILD_PATCH.format(was=parent_improvement, improvement=improvement)
        if parent is not None
        else KNOB_PATCH.format(improvement=improvement, latency=120.0)
    )
    (patches / f"{key}.patch").write_text(patch, encoding="utf-8")
    plan = staging / "plan.yaml"
    plan.write_text(
        f"hypothesis_id: {hypothesis_id}\n"
        f"approach_summary: Knob variant for {hypothesis_id}.\n"
        "experiments:\n"
        f"  - key: {key}\n"
        f"    title: Variant {key}\n"
        f"    change_summary: IMPROVEMENT={improvement}\n"
        f"    patch_file: patches/{key}.patch\n"
        + (f"    parent: {parent}\n" if parent is not None else ""),
        encoding="utf-8",
    )
    return plan


def _hypothesis(hypothesis_id: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        title=f"Knob idea {hypothesis_id}",
        claim="Turning the knob improves F1.",
        rationale="Fixture rationale.",
        feasibility=Level.HIGH,
        estimated_effort=Level.LOW,
        novelty_confidence=NoveltyConfidence.UNKNOWN,
        proposed_experiment="Set the knob and benchmark.",
    )


class FakeAI:
    """Stands in for the AI: stages plans and invents follow-up hypotheses.

    `improvements` is consumed one entry per planning call, so each round's
    measured outcome is fixed by the test rather than by a model.
    """

    def __init__(self, project_dir: Path, improvements: list[int]) -> None:
        self.project_dir = project_dir
        self.improvements = list(improvements)
        self.plan_calls = 0
        self.resynth_calls = 0
        self.parents_seen: list[str | None] = []
        self.last_improvement = 0
        self.merges_seen: list[tuple[str, str]] = []
        self.merge_improvement = 8
        self.observation_prompts: list[str] = []
        self.observation = "<observation>Loss was still falling at the last epoch.</observation>"

    def plan_all(
        self,
        conn: sqlite3.Connection,
        hypotheses: list[Hypothesis],
        provider_hint: str | None,
        model_hint: str | None,
        parent_experiment_id: str | None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        self.plan_calls += 1
        self.parents_seen.append(parent_experiment_id)
        improvement = self.improvements.pop(0) if self.improvements else 0
        plan_ids: list[str] = []
        for hypothesis in hypotheses:
            path = _stage_plan(
                self.project_dir,
                hypothesis.hypothesis_id,
                f"knob{self.plan_calls}",
                improvement,
                parent=parent_experiment_id,
                parent_improvement=self.last_improvement,
            )
            result, plan = import_experiment_plan(conn, path)
            assert result.ok, result.errors
            assert plan is not None
            plan_ids.append(plan.plan_id)
        self.last_improvement = improvement
        return plan_ids

    def author_merge(
        self,
        conn: sqlite3.Connection,
        left: Experiment,
        right: Experiment,
        provider_hint: str | None,
        model_hint: str | None,
    ) -> str:
        """One self-contained patch, as a model would write for a conflict."""
        self.merges_seen.append((left.experiment_id, right.experiment_id))
        return KNOB_PATCH.format(improvement=self.merge_improvement, latency=120.0)

    @property
    def name(self) -> str:
        """FakeAI doubles as the provider the observer resolves."""
        return "fake/fake"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        self.observation_prompts.append(user)
        return self.observation

    def resynthesize(
        self,
        conn: sqlite3.Connection,
        provider_hint: str | None,
        model_hint: str | None,
        log_content: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        self.resynth_calls += 1
        self.last_log = log_content
        project = get_project(conn)
        assert project is not None
        existing = list_hypotheses(conn)
        new_id = f"hyp-{len(existing) + 1:03d}"
        replace_hypotheses(conn, project.id, [*existing, _hypothesis(new_id)])
        return [new_id]


@pytest.fixture
def fake_ai(
    funnel_project: Path, isolated_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeAI]:
    """Loop-ready project with AI planning and re-synthesis stubbed out."""
    ai = FakeAI(isolated_project_dir, improvements=[5, 0, 0, 0])
    monkeypatch.setattr(engine, "plan_all_hypotheses", ai.plan_all)
    monkeypatch.setattr(engine, "resynthesize_hypotheses", ai.resynthesize)
    monkeypatch.setattr(engine, "author_merged_patch", ai.author_merge)
    monkeypatch.setattr(ai_service, "resolve_provider", lambda **kwargs: ai)
    yield ai


def _run(config: AutorunConfig, **kwargs: object) -> engine.AutorunResult:
    with closing(open_project_db()) as conn:
        return run_autorun(conn, config, **kwargs)  # type: ignore[arg-type]


class TestLoopStopsHonestly:
    def test_global_stall_ends_the_loop_and_names_the_reason(self, fake_ai: FakeAI) -> None:
        result = _run(AutorunConfig(global_stall=1, compound=False, yes=True))

        assert len(result.rounds) == 2
        assert result.rounds[0].improved_over_previous is True
        assert result.rounds[1].improved_over_previous is False
        assert "global stall" in result.stop_reason

    def test_round_cap_is_respected(self, fake_ai: FakeAI) -> None:
        result = _run(AutorunConfig(global_stall=99, max_rounds=2, compound=False, yes=True))

        assert len(result.rounds) == 2
        assert result.stop_reason == "max rounds (2) reached"

    def test_target_stops_the_loop_as_soon_as_it_is_reached(self, fake_ai: FakeAI) -> None:
        result = _run(AutorunConfig(target_value=0.85, compound=False, yes=True))

        assert result.objective_achieved is True
        assert len(result.rounds) == 1
        assert "objective achieved" in result.stop_reason

    def test_the_achieving_round_is_still_recorded(self, fake_ai: FakeAI) -> None:
        result = _run(AutorunConfig(target_value=0.85, compound=False, yes=True))

        assert result.rounds[0].best_metric_value == pytest.approx(0.85)
        assert result.total_experiments == 1


class TestMeasuredOutcome:
    def test_the_winner_is_the_improving_experiment(self, fake_ai: FakeAI) -> None:
        result = _run(AutorunConfig(global_stall=1, compound=False, yes=True))

        assert result.best_experiment_id == "exp-001"
        assert result.best_metric_value == pytest.approx(0.85)
        assert result.baseline_value == pytest.approx(0.80)
        assert result.metric_name == "f1"

    def test_a_second_round_is_planned_from_a_resynthesized_hypothesis(
        self, fake_ai: FakeAI
    ) -> None:
        _run(AutorunConfig(global_stall=1, compound=False, yes=True))

        assert fake_ai.resynth_calls == 1
        assert fake_ai.plan_calls == 2

    def test_the_research_log_is_fed_back_into_resynthesis(self, fake_ai: FakeAI) -> None:
        _run(AutorunConfig(global_stall=1, compound=False, yes=True))

        assert "Current best" in fake_ai.last_log
        assert "exp-001" in fake_ai.last_log


class TestResearchLogFile:
    def test_every_round_is_appended(self, fake_ai: FakeAI) -> None:
        _run(AutorunConfig(global_stall=1, compound=False, yes=True))

        text = research_log_path().read_text(encoding="utf-8")
        assert "## Round 1" in text
        assert "## Round 2" in text

    def test_the_objective_and_baseline_head_the_log(self, fake_ai: FakeAI) -> None:
        _run(AutorunConfig(max_rounds=1, compound=False, yes=True))

        text = research_log_path().read_text(encoding="utf-8")
        assert "Improve classification F1 without increasing latency" in text
        assert "0.8000" in text


class TestPersistenceAndResume:
    def test_state_is_written_with_one_record_per_round(self, fake_ai: FakeAI) -> None:
        _run(AutorunConfig(global_stall=1, compound=False, yes=True))

        state = load_state()
        assert state is not None
        assert state.status == "stopped"
        assert state.rounds_completed == 2
        assert [record.round_num for record in state.rounds] == [1, 2]
        assert state.best_experiment_id == "exp-001"

    def test_resume_continues_the_round_numbering(self, fake_ai: FakeAI) -> None:
        _run(AutorunConfig(max_rounds=1, compound=False, yes=True))

        resumed = _run(
            AutorunConfig(max_rounds=2, global_stall=99, compound=False, yes=True),
            resume=True,
        )

        assert resumed.resumed_from_round == 1
        assert [summary.round_num for summary in resumed.rounds] == [1, 2]

    def test_resume_carries_the_stall_counter_forward(self, fake_ai: FakeAI) -> None:
        _run(AutorunConfig(global_stall=2, max_rounds=2, compound=False, yes=True))
        state_before = load_state()
        assert state_before is not None
        assert state_before.global_stall_count == 1

        resumed = _run(
            AutorunConfig(global_stall=2, max_rounds=9, compound=False, yes=True),
            resume=True,
        )

        assert "global stall" in resumed.stop_reason

    def test_resume_without_prior_state_is_refused(
        self, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="No autorun state to resume"):
            _run(AutorunConfig(compound=False, yes=True), resume=True)


class TestApprovalGate:
    def test_declining_the_first_batch_runs_nothing(self, fake_ai: FakeAI) -> None:
        result = _run(
            AutorunConfig(global_stall=1, compound=False, yes=False),
            gate=lambda preview: False,
        )

        assert result.stop_reason == "plan not approved"
        assert result.total_experiments == 0
        assert result.best_experiment_id is None

    def test_approving_the_first_batch_runs_the_loop(self, fake_ai: FakeAI) -> None:
        seen: list[PlanPreview] = []

        def approve(preview: PlanPreview) -> bool:
            seen.append(preview)
            return True

        result = _run(
            AutorunConfig(max_rounds=1, compound=False, yes=False),
            gate=approve,
        )

        assert len(seen) == 1
        assert seen[0].hypothesis_id == "hyp-001"
        assert seen[0].worst_case_minutes > 0
        assert result.best_metric_value == pytest.approx(0.85)

    def test_the_gate_is_not_consulted_again_after_round_one(self, fake_ai: FakeAI) -> None:
        calls: list[str] = []

        def approve(preview: PlanPreview) -> bool:
            calls.append(preview.plan_id)
            return True

        _run(AutorunConfig(global_stall=1, compound=False, yes=False), gate=approve)

        assert calls == ["plan-001"]

    def test_yes_skips_the_gate_entirely(self, fake_ai: FakeAI) -> None:
        calls: list[str] = []

        def approve(preview: PlanPreview) -> bool:
            calls.append(preview.plan_id)
            return True

        _run(AutorunConfig(max_rounds=1, compound=False, yes=True), gate=approve)

        assert calls == []


class TestGraphDrivenPlanning:
    def test_without_compounding_every_round_starts_from_the_baseline(
        self, fake_ai: FakeAI
    ) -> None:
        _run(AutorunConfig(global_stall=1, compound=False, yes=True))

        assert fake_ai.parents_seen == [None, None]

    def test_compounding_expands_the_winning_node(self, fake_ai: FakeAI) -> None:
        """Round 1 has nothing to build on; round 2 builds on round 1's winner."""
        _run(AutorunConfig(global_stall=1, max_rounds=2, compound=True, yes=True))

        assert fake_ai.parents_seen == [None, "exp-001"]

    def test_compounding_does_not_build_on_a_failure(self, fake_ai: FakeAI) -> None:
        """Nothing beat the baseline, so there is no node worth expanding."""
        fake_ai.improvements = [0, 0]

        _run(AutorunConfig(global_stall=99, max_rounds=2, compound=True, yes=True))

        assert fake_ai.parents_seen == [None, None]

    def test_exploration_reports_the_node_it_expanded(self, fake_ai: FakeAI) -> None:
        messages: list[str] = []

        _run(
            AutorunConfig(global_stall=1, max_rounds=2, explore=0.5, yes=True),
            on_progress=messages.append,
        )

        assert any("Expanding" in message for message in messages)

    def test_overlapping_winners_are_re_authored_into_one_patch(self, fake_ai: FakeAI) -> None:
        """Both winners create src/algo.py, so composition cannot combine them."""
        fake_ai.improvements = [5, 3, 0]

        result = _run(
            AutorunConfig(global_stall=99, max_rounds=3, compound=False, merge=True, yes=True)
        )

        assert fake_ai.merges_seen == [("exp-001", "exp-002")]
        # 0.88 is the authored merge's own patch — neither winner reached it.
        assert result.best_metric_value == pytest.approx(0.88)
        assert len(result.rounds) == 3

    def test_the_re_authored_merge_keeps_both_parents_as_lineage(self, fake_ai: FakeAI) -> None:
        fake_ai.improvements = [5, 3, 0]

        _run(AutorunConfig(global_stall=99, max_rounds=3, compound=False, merge=True, yes=True))

        with closing(open_project_db()) as conn:
            merge = get_experiment(conn, "exp-003")
        assert merge is not None
        assert merge.parent_experiment_ids == ["exp-001", "exp-002"]
        assert merge.patch_includes_parents is True

    def test_a_merge_the_ai_declines_is_reported_and_the_loop_continues(
        self, fake_ai: FakeAI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_ai.improvements = [5, 3, 0]
        messages: list[str] = []

        def refuse(*args: object, **kwargs: object) -> str:
            raise MergeNotPossibleError("both branches set IMPROVEMENT to different values")

        monkeypatch.setattr(engine, "author_merged_patch", refuse)
        result = _run(
            AutorunConfig(global_stall=99, max_rounds=3, compound=False, merge=True, yes=True),
            on_progress=messages.append,
        )

        assert any("different values" in message for message in messages)
        assert len(result.rounds) == 3


class TestObservations:
    def test_each_experiment_gets_an_observation_from_its_own_logs(self, fake_ai: FakeAI) -> None:
        fake_ai.improvements = [5, 0]

        _run(AutorunConfig(global_stall=99, max_rounds=2, observe=True, yes=True))

        with closing(open_project_db()) as conn:
            experiment = get_experiment(conn, "exp-001")
        assert experiment is not None
        assert experiment.observation == "Loss was still falling at the last epoch."

    def test_the_observer_is_shown_the_run_output_and_the_change(self, fake_ai: FakeAI) -> None:
        fake_ai.improvements = [5, 0]

        _run(AutorunConfig(global_stall=99, max_rounds=1, observe=True, yes=True))

        assert len(fake_ai.observation_prompts) == 1
        prompt = fake_ai.observation_prompts[0]
        assert "exp-001" in prompt
        assert "IMPROVEMENT=5" in prompt
        assert "Benchmark output (tail)" in prompt

    def test_observations_land_in_the_research_log(self, fake_ai: FakeAI) -> None:
        fake_ai.improvements = [5, 0]

        _run(AutorunConfig(global_stall=99, max_rounds=1, observe=True, yes=True))

        log = research_log_path().read_text(encoding="utf-8")
        assert "> Loss was still falling at the last epoch." in log

    def test_no_observations_are_written_without_the_flag(self, fake_ai: FakeAI) -> None:
        fake_ai.improvements = [5, 0]

        _run(AutorunConfig(global_stall=99, max_rounds=1, yes=True))

        with closing(open_project_db()) as conn:
            experiment = get_experiment(conn, "exp-001")
        assert experiment is not None
        assert experiment.observation is None
        assert fake_ai.observation_prompts == []

    def test_an_observer_failure_is_reported_and_the_loop_continues(
        self, fake_ai: FakeAI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_ai.improvements = [5, 0]
        messages: list[str] = []

        def explode(system: str, user: str, *, max_tokens: int = 8192) -> str:
            raise RuntimeError("provider is rate limited")

        monkeypatch.setattr(fake_ai, "generate", explode)
        result = _run(
            AutorunConfig(global_stall=99, max_rounds=2, observe=True, yes=True),
            on_progress=messages.append,
        )

        assert any("rate limited" in message for message in messages)
        assert len(result.rounds) == 2

    def test_an_observation_cannot_change_what_was_measured(self, fake_ai: FakeAI) -> None:
        fake_ai.improvements = [5, 0]
        fake_ai.observation = "<observation>This change is clearly worse; reject it.</observation>"

        result = _run(AutorunConfig(global_stall=99, max_rounds=1, observe=True, yes=True))

        assert result.best_experiment_id == "exp-001"
        assert result.best_metric_value == pytest.approx(0.85)


class TestBlockedProjects:
    def test_a_project_without_a_baseline_is_refused(
        self, contracted_project: Path, isolated_project_dir: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="No frozen baseline"):
            _run(AutorunConfig(yes=True))
