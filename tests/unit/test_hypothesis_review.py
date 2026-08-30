"""Approving and rejecting hypotheses, and what rejection stops."""

from contextlib import closing
from pathlib import Path

from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.domain.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    Level,
    NoveltyConfidence,
)
from researchforge.storage.db import open_project_db
from researchforge.storage.hypothesis_repository import (
    get_hypothesis,
    list_hypotheses,
    record_review,
    replace_hypotheses,
)
from researchforge.storage.project_repository import get_project


def _hypothesis(hypothesis_id: str, title: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        title=title,
        claim=f"{title} improves the metric.",
        rationale="Fixture rationale.",
        feasibility=Level.HIGH,
        estimated_effort=Level.LOW,
        novelty_confidence=NoveltyConfidence.UNKNOWN,
        proposed_experiment="Apply the change and benchmark it.",
    )


def _store(hypotheses: list[Hypothesis]) -> None:
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        assert project is not None
        replace_hypotheses(conn, project.id, hypotheses)


KNOB_PATCH = """\
diff --git a/src/algo.py b/src/algo.py
new file mode 100644
--- /dev/null
+++ b/src/algo.py
@@ -0,0 +1,2 @@
+IMPROVEMENT = 5
+LATENCY = 150.0
"""


def _stage_plan(base: Path) -> Path:
    staging = base / ".researchforge" / "experiments"
    patches = staging / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    (patches / "improve.patch").write_text(KNOB_PATCH, encoding="utf-8")
    plan = staging / "plan.yaml"
    plan.write_text(
        "hypothesis_id: hyp-001\n"
        "approach_summary: Knob variants.\n"
        "experiments:\n"
        "  - key: improve\n"
        "    title: Variant improve\n"
        "    change_summary: Set knobs for improve.\n"
        "    patch_file: patches/improve.patch\n",
        encoding="utf-8",
    )
    return plan


class TestNewHypothesesAreUnreviewed:
    def test_an_imported_hypothesis_is_speculative(self, initialized_project: Path) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            stored = get_hypothesis(conn, "hyp-001")
        assert stored is not None
        assert stored.status is HypothesisStatus.SPECULATIVE
        assert stored.review is None

    def test_an_unreviewed_hypothesis_can_still_be_planned(self, initialized_project: Path) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            stored = get_hypothesis(conn, "hyp-001")
        assert stored is not None
        assert stored.is_plannable


class TestRecordReview:
    def test_approving_sets_the_status_and_keeps_the_decision(
        self, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            reviewed = record_review(conn, "hyp-001", HypothesisStatus.APPROVED, "worth a look")
        assert reviewed is not None
        assert reviewed.status is HypothesisStatus.APPROVED
        assert reviewed.review is not None
        assert reviewed.review.decision is HypothesisStatus.APPROVED
        assert reviewed.review.reason == "worth a look"

    def test_rejecting_makes_it_unplannable(self, initialized_project: Path) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            reviewed = record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "already tried")
        assert reviewed is not None
        assert reviewed.status is HypothesisStatus.REJECTED
        assert not reviewed.is_plannable

    def test_the_review_is_persisted(self, initialized_project: Path) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "no evidence")
        with closing(open_project_db()) as conn:
            reloaded = get_hypothesis(conn, "hyp-001")
        assert reloaded is not None
        assert reloaded.status is HypothesisStatus.REJECTED
        assert reloaded.review is not None
        assert reloaded.review.reason == "no evidence"

    def test_reviewing_one_leaves_the_others_alone(self, initialized_project: Path) -> None:
        _store([_hypothesis("hyp-001", "Caching"), _hypothesis("hyp-002", "Batching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "no")
            untouched = get_hypothesis(conn, "hyp-002")
        assert untouched is not None
        assert untouched.status is HypothesisStatus.SPECULATIVE
        assert untouched.review is None

    def test_an_unknown_id_reviews_nothing(self, initialized_project: Path) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            assert record_review(conn, "hyp-404", HypothesisStatus.APPROVED) is None
            assert len(list_hypotheses(conn)) == 1

    def test_a_later_review_replaces_an_earlier_one(self, initialized_project: Path) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "too risky")
            second = record_review(conn, "hyp-001", HypothesisStatus.APPROVED, "reconsidered")
        assert second is not None
        assert second.status is HypothesisStatus.APPROVED
        assert second.review is not None
        assert second.review.reason == "reconsidered"


class TestApproveAndRejectCommands:
    def test_approve_accepts_several_ids_at_once(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store(
            [
                _hypothesis("hyp-001", "Caching"),
                _hypothesis("hyp-002", "Batching"),
                _hypothesis("hyp-003", "Pruning"),
            ]
        )
        result = cli_runner.invoke(app, ["hypotheses", "approve", "hyp-001", "hyp-003"])
        assert result.exit_code == 0, result.output
        with closing(open_project_db()) as conn:
            statuses = {h.hypothesis_id: h.status for h in list_hypotheses(conn)}
        assert statuses["hyp-001"] is HypothesisStatus.APPROVED
        assert statuses["hyp-002"] is HypothesisStatus.SPECULATIVE
        assert statuses["hyp-003"] is HypothesisStatus.APPROVED

    def test_reject_records_the_reason(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        result = cli_runner.invoke(
            app, ["hypotheses", "reject", "hyp-001", "--reason", "no supporting evidence"]
        )
        assert result.exit_code == 0, result.output
        with closing(open_project_db()) as conn:
            stored = get_hypothesis(conn, "hyp-001")
        assert stored is not None
        assert stored.review is not None
        assert stored.review.reason == "no supporting evidence"

    def test_reject_requires_a_reason(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        result = cli_runner.invoke(app, ["hypotheses", "reject", "hyp-001"])
        assert result.exit_code != 0

    def test_an_unknown_id_is_reported_and_fails(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        result = cli_runner.invoke(app, ["hypotheses", "approve", "hyp-404"])
        assert result.exit_code == 1
        assert "hyp-404" in result.output

    def test_a_known_id_is_still_reviewed_when_another_is_unknown(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        result = cli_runner.invoke(app, ["hypotheses", "approve", "hyp-001", "hyp-404"])
        assert result.exit_code == 1
        with closing(open_project_db()) as conn:
            stored = get_hypothesis(conn, "hyp-001")
        assert stored is not None
        assert stored.status is HypothesisStatus.APPROVED

    def test_json_output_names_what_was_reviewed(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        import json

        _store([_hypothesis("hyp-001", "Caching")])
        result = cli_runner.invoke(app, ["hypotheses", "approve", "hyp-001", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["decision"] == "approved"
        assert payload["reviewed"] == ["hyp-001"]
        assert payload["unknown"] == []


class TestReviewCommand:
    def test_it_offers_each_unreviewed_hypothesis(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching"), _hypothesis("hyp-002", "Batching")])
        result = cli_runner.invoke(app, ["hypotheses", "review"], input="a\nr\nnot worth it\n")
        assert result.exit_code == 0, result.output
        with closing(open_project_db()) as conn:
            statuses = {h.hypothesis_id: h.status for h in list_hypotheses(conn)}
        assert statuses["hyp-001"] is HypothesisStatus.APPROVED
        assert statuses["hyp-002"] is HypothesisStatus.REJECTED

    def test_skipping_leaves_a_hypothesis_unreviewed(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        result = cli_runner.invoke(app, ["hypotheses", "review"], input="s\n")
        assert result.exit_code == 0, result.output
        with closing(open_project_db()) as conn:
            stored = get_hypothesis(conn, "hyp-001")
        assert stored is not None
        assert stored.status is HypothesisStatus.SPECULATIVE

    def test_it_does_not_re_ask_about_a_decided_hypothesis(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching"), _hypothesis("hyp-002", "Batching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.APPROVED)

        result = cli_runner.invoke(app, ["hypotheses", "review"], input="s\n")
        assert result.exit_code == 0, result.output
        assert "hyp-002" in result.output
        assert "1 hypothesis(es) to review" in result.output

    def test_nothing_to_review_says_so(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.APPROVED)

        result = cli_runner.invoke(app, ["hypotheses", "review"])
        assert result.exit_code == 0, result.output
        assert "Nothing to review" in result.output

    def test_it_refuses_json_because_it_is_interactive(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        result = cli_runner.invoke(app, ["hypotheses", "review", "--json"])
        assert result.exit_code == 1


class TestRejectionBlocksPlanning:
    def test_the_context_export_refuses_a_rejected_hypothesis(
        self, baselined_project: Path
    ) -> None:
        from researchforge.experiments.context_export import (
            ExperimentContextError,
            build_experiment_context,
        )

        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "already tried")

        with closing(open_project_db()) as conn:
            try:
                build_experiment_context(conn, "hyp-001")
            except ExperimentContextError as exc:
                assert "rejected in review" in str(exc)
                assert "already tried" in str(exc)
            else:
                raise AssertionError("a rejected hypothesis should not export planning context")

    def test_experiment_plan_refuses_a_rejected_hypothesis(
        self, cli_runner: CliRunner, baselined_project: Path
    ) -> None:
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "already tried")

        result = cli_runner.invoke(app, ["experiment", "plan", "hyp-001"])
        assert result.exit_code == 1
        assert "rejected in review" in result.output

    def test_importing_a_plan_for_a_rejected_hypothesis_is_refused(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(isolated_project_dir)
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "already tried")

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 1
        assert "rejected in review" in result.output

    def test_approving_it_again_unblocks_planning(
        self, cli_runner: CliRunner, baselined_project: Path, isolated_project_dir: Path
    ) -> None:
        plan = _stage_plan(isolated_project_dir)
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "already tried")
        assert cli_runner.invoke(app, ["hypotheses", "approve", "hyp-001"]).exit_code == 0

        result = cli_runner.invoke(app, ["experiment", "import", str(plan)])
        assert result.exit_code == 0, result.output

    def test_autorun_skips_a_rejected_hypothesis(self, baselined_project: Path) -> None:
        from researchforge.autorun.engine import get_pending_hypotheses

        _store([_hypothesis("hyp-001", "Caching"), _hypothesis("hyp-002", "Batching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-002", HypothesisStatus.REJECTED, "no")
            pending = get_pending_hypotheses(conn)
        assert [h.hypothesis_id for h in pending] == ["hyp-001"]

    def test_autorun_still_considers_an_approved_and_an_unreviewed_one(
        self, baselined_project: Path
    ) -> None:
        from researchforge.autorun.engine import get_pending_hypotheses

        _store([_hypothesis("hyp-001", "Caching"), _hypothesis("hyp-002", "Batching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.APPROVED)
            pending = get_pending_hypotheses(conn)
        assert [h.hypothesis_id for h in pending] == ["hyp-001", "hyp-002"]


class TestListShowsReviewState:
    def test_the_list_marks_reviewed_hypotheses(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching"), _hypothesis("hyp-002", "Batching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.APPROVED)
            record_review(conn, "hyp-002", HypothesisStatus.REJECTED, "no")

        result = cli_runner.invoke(app, ["hypotheses", "list"])
        assert result.exit_code == 0, result.output
        assert "approved" in result.output
        assert "rejected" in result.output

    def test_the_list_warns_that_rejected_ones_are_skipped(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "no")

        result = cli_runner.invoke(app, ["hypotheses", "list"])
        assert "will be skipped" in result.output

    def test_show_reports_the_reason(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        _store([_hypothesis("hyp-001", "Caching")])
        with closing(open_project_db()) as conn:
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "contradicted by arxiv:1")

        result = cli_runner.invoke(app, ["hypotheses", "show", "hyp-001"])
        assert result.exit_code == 0, result.output
        assert "contradicted by arxiv:1" in result.output
