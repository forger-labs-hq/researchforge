"""The derived audit trail: the pure event builders, and the CLI over them."""

import json
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from researchforge.audit.service import build_trail
from researchforge.audit.trail import (
    AuditEvent,
    AuditEventKind,
    baseline_events,
    deliverable_events,
    execution_events,
    experiment_events,
    hypothesis_events,
    landscape_events,
    order_events,
    plan_events,
    run_events,
    search_events,
    unapproved_plans,
)
from researchforge.cli import app
from researchforge.domain.baseline import BaselineRun, BaselineStatus, EnvironmentFingerprint
from researchforge.domain.deliverable import Deliverable, DeliverableKind
from researchforge.domain.environment import ExecutionEngine
from researchforge.domain.experiment import (
    BenchmarkStage,
    Decision,
    DecisionOutcome,
    ExecutionArtifacts,
    ExecutionRecordStatus,
    Experiment,
    ExperimentExecution,
    ExperimentPlan,
    ExperimentRunGroup,
    ExperimentStatus,
    PlanApproval,
    PlanStatus,
    RunStatus,
)
from researchforge.domain.hypothesis import (
    Hypothesis,
    HypothesisReview,
    HypothesisStatus,
    Level,
    NoveltyConfidence,
)
from researchforge.execution.metrics import MetricResult
from researchforge.storage.db import open_project_db
from researchforge.storage.hypothesis_repository import record_review, replace_hypotheses
from researchforge.storage.project_repository import get_project

NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


def _fingerprint() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        platform="test",
        execution_mode=ExecutionEngine.VENV,
        contract_id="c1",
        contract_version=1,
        commit_sha="a" * 40,
    )


def _metrics(value: float) -> MetricResult:
    return MetricResult.model_validate(
        {"schema_version": 1, "primary_metric": {"name": "f1", "value": value}}
    )


def _baseline(
    status: BaselineStatus = BaselineStatus.SUCCEEDED, value: float = 0.80
) -> BaselineRun:
    return BaselineRun(
        baseline_id="b1",
        contract_id="c1",
        contract_version=1,
        commit_sha="a" * 40,
        execution_mode=ExecutionEngine.VENV,
        command="eval",
        status=status,
        metrics=_metrics(value) if status is BaselineStatus.SUCCEEDED else None,
        failure_reason=None if status is BaselineStatus.SUCCEEDED else "boom",
        fingerprint=_fingerprint(),
        stdout_path="s",
        stderr_path="e",
        started_at=NOW,
        completed_at=NOW + timedelta(minutes=2),
        duration_seconds=120.0,
    )


def _plan(approval: PlanApproval | None = None, status: PlanStatus = PlanStatus.PLANNED) -> (
    ExperimentPlan
):
    return ExperimentPlan(
        plan_id="plan-001",
        hypothesis_id="hyp-001",
        contract_id="c1",
        contract_version=1,
        baseline_id="b1",
        baseline_commit="a" * 40,
        approach_summary="Knob variants.",
        status=status,
        approval=approval,
        source_file="plan.yaml",
        created_at=NOW,
        updated_at=NOW,
    )


def _experiment(decision: Decision | None = None) -> Experiment:
    return Experiment(
        experiment_id="exp-001",
        plan_id="plan-001",
        hypothesis_id="hyp-001",
        title="Cache the hot path",
        change_summary="c",
        patch_text="p",
        patch_sha256="0" * 64,
        status=ExperimentStatus.PROMISING,
        decision=decision,
        created_at=NOW,
        updated_at=NOW + timedelta(minutes=30),
    )


def _execution(
    stage: BenchmarkStage = BenchmarkStage.FULL,
    status: ExecutionRecordStatus = ExecutionRecordStatus.SUCCEEDED,
    value: float | None = 0.85,
) -> ExperimentExecution:
    return ExperimentExecution(
        execution_id="e1",
        experiment_id="exp-001",
        run_id="run-001",
        hypothesis_id="hyp-001",
        baseline_commit="a" * 40,
        execution_mode=ExecutionEngine.VENV,
        benchmark_stage=stage,
        attempt=1,
        change_summary="c",
        started_at=NOW,
        completed_at=NOW + timedelta(minutes=5),
        duration_seconds=300.0,
        status=status,
        metrics=_metrics(value) if value is not None else None,
        artifacts=ExecutionArtifacts(diff_path="d", stdout_path="o", stderr_path="e"),
        fingerprint=_fingerprint(),
    )


def _hypothesis(review: HypothesisReview | None = None) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="hyp-001",
        title="Caching improves F1",
        claim="Memoizing hot paths improves F1.",
        rationale="Fixture rationale.",
        feasibility=Level.HIGH,
        estimated_effort=Level.LOW,
        novelty_confidence=NoveltyConfidence.UNKNOWN,
        proposed_experiment="Apply caching variants and benchmark.",
        status=review.decision if review is not None else HypothesisStatus.SPECULATIVE,
        review=review,
    )


class TestSearchEvents:
    def test_a_search_becomes_one_event(self) -> None:
        events = search_events(
            [
                {
                    "run_id": "sr-1",
                    "queries": ["vision transformers"],
                    "fetched_count": 40,
                    "selected_count": 8,
                    "created_at": NOW.isoformat(),
                }
            ]
        )
        assert len(events) == 1
        assert events[0].kind is AuditEventKind.PAPERS_SEARCHED
        assert events[0].subject == "sr-1"
        assert "40 fetched" in events[0].summary
        assert "8 stored" in events[0].summary

    def test_a_row_without_a_timestamp_is_skipped_not_guessed(self) -> None:
        assert search_events([{"run_id": "sr-1", "fetched_count": 3}]) == []


class TestHypothesisEvents:
    def test_an_unreviewed_hypothesis_produces_no_event(self) -> None:
        assert hypothesis_events([_hypothesis()]) == []

    def test_a_rejection_records_its_reason(self) -> None:
        review = HypothesisReview(
            decision=HypothesisStatus.REJECTED, reason="contradicted", decided_at=NOW
        )
        events = hypothesis_events([_hypothesis(review)])
        assert len(events) == 1
        assert events[0].kind is AuditEventKind.HYPOTHESIS_REVIEWED
        assert "rejected" in events[0].summary
        assert events[0].detail["reason"] == "contradicted"

    def test_an_approval_without_a_reason_carries_no_empty_detail(self) -> None:
        review = HypothesisReview(decision=HypothesisStatus.APPROVED, decided_at=NOW)
        events = hypothesis_events([_hypothesis(review)])
        assert events[0].detail == {}


class TestBaselineEvents:
    def test_a_successful_baseline_reports_the_value_it_froze(self) -> None:
        events = baseline_events([_baseline()])
        assert len(events) == 1
        assert events[0].kind is AuditEventKind.BASELINE_MEASURED
        assert "f1 = 0.8" in events[0].summary

    def test_a_failed_baseline_is_still_an_event(self) -> None:
        events = baseline_events([_baseline(status=BaselineStatus.FAILED_SETUP)])
        assert len(events) == 1
        assert "failed_setup" in events[0].summary
        assert events[0].detail["status"] == "failed_setup"


class TestPlanEvents:
    def test_an_unapproved_plan_only_records_its_creation(self) -> None:
        events = plan_events([_plan()], [_experiment()])
        assert [event.kind for event in events] == [AuditEventKind.PLAN_CREATED]
        assert "1 experiment(s)" in events[0].summary

    def test_a_typed_approval_is_reported_as_a_person_typing_it(self) -> None:
        approval = PlanApproval(
            approved_at=NOW + timedelta(minutes=5),
            method="typed",
            experiment_ids=["exp-001"],
            estimated_max_minutes=10,
        )
        events = plan_events([_plan(approval, PlanStatus.APPROVED)], [_experiment()])
        approved = [e for e in events if e.kind is AuditEventKind.PLAN_APPROVED]
        assert len(approved) == 1
        assert "typed confirmation" in approved[0].summary
        assert approved[0].detail["method"] == "typed"

    def test_a_flag_approval_is_reported_as_unattended(self) -> None:
        approval = PlanApproval(
            approved_at=NOW + timedelta(minutes=5),
            method="flag",
            experiment_ids=["exp-001"],
            estimated_max_minutes=10,
        )
        events = plan_events([_plan(approval, PlanStatus.APPROVED)], [_experiment()])
        approved = [e for e in events if e.kind is AuditEventKind.PLAN_APPROVED]
        assert "--yes flag" in approved[0].summary


class TestExecutionAndExperimentEvents:
    def test_a_benchmark_attempt_reports_stage_status_and_value(self) -> None:
        events = execution_events([_execution()])
        assert len(events) == 1
        assert events[0].kind is AuditEventKind.BENCHMARK_RAN
        assert "full benchmark attempt 1 succeeded" in events[0].summary
        assert "f1 = 0.85" in events[0].summary

    def test_a_failed_attempt_reports_no_value(self) -> None:
        events = execution_events(
            [_execution(status=ExecutionRecordStatus.FAILED_TIMEOUT, value=None)]
        )
        assert "failed_timeout" in events[0].summary
        assert "f1" not in events[0].summary

    def test_an_undecided_experiment_produces_no_event(self) -> None:
        assert experiment_events([_experiment()]) == []

    def test_a_decision_records_its_outcome_and_reason(self) -> None:
        decision = Decision(outcome=DecisionOutcome.KEEP, reason="beat the baseline")
        events = experiment_events([_experiment(decision)])
        assert len(events) == 1
        assert events[0].kind is AuditEventKind.EXPERIMENT_DECIDED
        assert "keep" in events[0].summary
        assert "beat the baseline" in events[0].summary


class TestRunAndDeliverableEvents:
    def test_an_open_run_has_a_start_but_no_completion(self) -> None:
        run = ExperimentRunGroup(
            run_id="run-001",
            plan_id="plan-001",
            execution_mode=ExecutionEngine.VENV,
            status=RunStatus.IN_PROGRESS,
            started_at=NOW,
            completed_at=None,
        )
        assert [event.kind for event in run_events([run])] == [AuditEventKind.RUN_STARTED]

    def test_a_finished_run_has_both(self) -> None:
        run = ExperimentRunGroup(
            run_id="run-001",
            plan_id="plan-001",
            execution_mode=ExecutionEngine.VENV,
            status=RunStatus.COMPLETED,
            started_at=NOW,
            completed_at=NOW + timedelta(hours=1),
        )
        kinds = [event.kind for event in run_events([run])]
        assert kinds == [AuditEventKind.RUN_STARTED, AuditEventKind.RUN_COMPLETED]

    def test_a_deliverable_names_what_was_produced_and_where(self) -> None:
        deliverable = Deliverable(
            deliverable_id="d1",
            kind=DeliverableKind.BRANCH,
            experiment_id="exp-001",
            location="researchforge/exp-001",
            commit_sha="b" * 40,
            created_at=NOW,
        )
        events = deliverable_events([deliverable])
        assert len(events) == 1
        assert events[0].kind is AuditEventKind.DELIVERABLE_CREATED
        assert "researchforge/exp-001" in events[0].summary
        assert events[0].detail["commit"] == "b" * 40


class TestLandscapeEvents:
    def test_no_landscape_means_no_event(self) -> None:
        assert landscape_events(None, 0) == []

    def test_an_imported_landscape_reports_its_direction_count(self) -> None:
        events = landscape_events(NOW, 3)
        assert len(events) == 1
        assert "3 direction(s)" in events[0].summary


class TestOrdering:
    def test_events_come_out_oldest_first(self) -> None:
        later = AuditEvent(
            at=NOW + timedelta(hours=1),
            kind=AuditEventKind.RUN_STARTED,
            subject="run-001",
            summary="later",
        )
        earlier = AuditEvent(
            at=NOW,
            kind=AuditEventKind.PROJECT_CREATED,
            subject="p1",
            summary="earlier",
        )
        assert [e.summary for e in order_events([later, earlier])] == ["earlier", "later"]

    def test_events_sharing_a_timestamp_keep_a_stable_order(self) -> None:
        first = AuditEvent(
            at=NOW, kind=AuditEventKind.PLAN_CREATED, subject="plan-002", summary="b"
        )
        second = AuditEvent(
            at=NOW, kind=AuditEventKind.PLAN_CREATED, subject="plan-001", summary="a"
        )
        once = [e.summary for e in order_events([first, second])]
        again = [e.summary for e in order_events([second, first])]
        assert once == again == ["a", "b"]


class TestGateFindings:
    def test_a_planned_plan_without_approval_is_not_a_finding(self) -> None:
        assert unapproved_plans([_plan()]) == []

    def test_a_running_plan_without_approval_is_a_finding(self) -> None:
        assert unapproved_plans([_plan(status=PlanStatus.RUNNING)]) == ["plan-001"]

    def test_an_approved_plan_with_its_approval_is_not_a_finding(self) -> None:
        approval = PlanApproval(
            approved_at=NOW,
            method="typed",
            experiment_ids=["exp-001"],
            estimated_max_minutes=10,
        )
        assert unapproved_plans([_plan(approval, PlanStatus.APPROVED)]) == []

    def test_a_cancelled_plan_is_not_a_finding(self) -> None:
        assert unapproved_plans([_plan(status=PlanStatus.CANCELLED)]) == []


class TestBuildTrailFromARealProject:
    def test_a_fresh_project_records_its_creation(self, initialized_project: Path) -> None:
        with closing(open_project_db()) as conn:
            trail = build_trail(conn)
        kinds = [event.kind for event in trail.events]
        assert AuditEventKind.PROJECT_CREATED in kinds
        assert trail.gate_findings == []

    def test_the_objective_is_carried_on_the_creation_event(
        self, initialized_project: Path
    ) -> None:
        with closing(open_project_db()) as conn:
            trail = build_trail(conn)
        created = next(e for e in trail.events if e.kind is AuditEventKind.PROJECT_CREATED)
        assert "uncertainty-aware routing" in created.detail["objective"]

    def test_a_hypothesis_review_lands_in_the_trail(self, initialized_project: Path) -> None:
        with closing(open_project_db()) as conn:
            project = get_project(conn)
            assert project is not None
            replace_hypotheses(conn, project.id, [_hypothesis()])
            record_review(conn, "hyp-001", HypothesisStatus.REJECTED, "already tried")

        with closing(open_project_db()) as conn:
            trail = build_trail(conn)
        reviewed = [e for e in trail.events if e.kind is AuditEventKind.HYPOTHESIS_REVIEWED]
        assert len(reviewed) == 1
        assert reviewed[0].detail["reason"] == "already tried"

    def test_a_baselined_project_records_contract_and_baseline(
        self, baselined_project: Path
    ) -> None:
        with closing(open_project_db()) as conn:
            trail = build_trail(conn)
        kinds = [event.kind for event in trail.events]
        assert AuditEventKind.CONTRACT_APPROVED in kinds
        assert AuditEventKind.BASELINE_MEASURED in kinds

    def test_the_contract_event_names_the_metric_it_committed_to(
        self, baselined_project: Path
    ) -> None:
        with closing(open_project_db()) as conn:
            trail = build_trail(conn)
        approved = next(e for e in trail.events if e.kind is AuditEventKind.CONTRACT_APPROVED)
        assert approved.detail["metric"] == "f1"
        assert approved.detail["direction"] == "maximize"

    def test_reading_the_same_database_twice_gives_the_same_trail(
        self, baselined_project: Path
    ) -> None:
        with closing(open_project_db()) as conn:
            first = build_trail(conn)
        with closing(open_project_db()) as conn:
            second = build_trail(conn)
        assert first.model_dump_json() == second.model_dump_json()


class TestAuditLogCommand:
    def test_it_prints_the_history(self, cli_runner: CliRunner, initialized_project: Path) -> None:
        result = cli_runner.invoke(app, ["audit", "log"])
        assert result.exit_code == 0, result.output
        assert "project_created" in result.output

    def test_last_limits_to_the_most_recent_entries(
        self, cli_runner: CliRunner, baselined_project: Path
    ) -> None:
        full = cli_runner.invoke(app, ["audit", "log", "--json"])
        assert full.exit_code == 0, full.output
        total = len(json.loads(full.output)["events"])
        assert total > 1

        limited = cli_runner.invoke(app, ["audit", "log", "--last", "1", "--json"])
        assert limited.exit_code == 0, limited.output
        assert len(json.loads(limited.output)["events"]) == 1

    def test_last_keeps_the_newest_not_the_oldest(
        self, cli_runner: CliRunner, baselined_project: Path
    ) -> None:
        full = cli_runner.invoke(app, ["audit", "log", "--json"])
        events = json.loads(full.output)["events"]
        limited = cli_runner.invoke(app, ["audit", "log", "--last", "1", "--json"])
        assert json.loads(limited.output)["events"][0] == events[-1]

    def test_kind_filters_to_one_sort_of_entry(
        self, cli_runner: CliRunner, baselined_project: Path
    ) -> None:
        result = cli_runner.invoke(
            app, ["audit", "log", "--kind", "contract_approved", "--json"]
        )
        assert result.exit_code == 0, result.output
        events = json.loads(result.output)["events"]
        assert events
        assert {event["kind"] for event in events} == {"contract_approved"}

    def test_an_unknown_kind_is_rejected(
        self, cli_runner: CliRunner, initialized_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["audit", "log", "--kind", "nonsense"])
        assert result.exit_code != 0

    def test_it_says_how_many_of_the_total_it_showed(
        self, cli_runner: CliRunner, baselined_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["audit", "log", "--last", "1"])
        assert result.exit_code == 0, result.output
        assert "Showing 1 of" in result.output


class TestAuditExportCommand:
    def test_it_writes_a_json_file(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        destination = tmp_path / "audit.json"
        result = cli_runner.invoke(app, ["audit", "export", str(destination)])
        assert result.exit_code == 0, result.output
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["events"]
        assert payload["gate_findings"] == []

    def test_the_export_holds_everything_not_just_a_window(
        self, cli_runner: CliRunner, baselined_project: Path, tmp_path: Path
    ) -> None:
        destination = tmp_path / "audit.json"
        assert cli_runner.invoke(app, ["audit", "export", str(destination)]).exit_code == 0
        exported = json.loads(destination.read_text(encoding="utf-8"))["events"]

        logged = cli_runner.invoke(app, ["audit", "log", "--json"])
        assert len(exported) == len(json.loads(logged.output)["events"])

    def test_it_creates_missing_parent_directories(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        destination = tmp_path / "reports" / "nested" / "audit.json"
        result = cli_runner.invoke(app, ["audit", "export", str(destination)])
        assert result.exit_code == 0, result.output
        assert destination.is_file()

    def test_json_output_reports_the_path_and_count(
        self, cli_runner: CliRunner, initialized_project: Path, tmp_path: Path
    ) -> None:
        destination = tmp_path / "audit.json"
        result = cli_runner.invoke(app, ["audit", "export", str(destination), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["path"] == str(destination)
        assert payload["event_count"] >= 1
