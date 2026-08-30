"""The project's history, reconstructed from the records it already keeps.

Every consequential step already leaves a durable, timestamped record: a
contract carries when it was approved, a plan carries who typed the approval and
what it estimated, an execution carries its stage and outcome, a deliverable
carries what was shipped. The audit trail is those facts put in order.

Deriving it rather than writing a parallel log is a deliberate choice. A
separate log can disagree with the records — a write forgotten in one code path
is a silent hole, and a project that predates the log has no history at all.
This cannot drift, because it is not a copy of the truth; it is a reading of it.

The cost is honest and worth naming: the trail can only report what the records
preserve. A gate that was declined leaves nothing behind, and neither does a
status that changed twice between two saves. Nothing here infers such an event.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from researchforge.domain.baseline import BaselineRun, BaselineStatus
from researchforge.domain.contract import ExperimentContract
from researchforge.domain.deliverable import Deliverable
from researchforge.domain.experiment import (
    Experiment,
    ExperimentExecution,
    ExperimentPlan,
    ExperimentRunGroup,
    PlanStatus,
)
from researchforge.domain.hypothesis import Hypothesis
from researchforge.domain.project import Project


class AuditEventKind(StrEnum):
    """What happened. Ordered roughly by where it falls in the workflow."""

    PROJECT_CREATED = "project_created"
    PAPERS_SEARCHED = "papers_searched"
    LANDSCAPE_IMPORTED = "landscape_imported"
    HYPOTHESIS_REVIEWED = "hypothesis_reviewed"
    CONTRACT_APPROVED = "contract_approved"
    BASELINE_MEASURED = "baseline_measured"
    PLAN_CREATED = "plan_created"
    PLAN_APPROVED = "plan_approved"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    BENCHMARK_RAN = "benchmark_ran"
    EXPERIMENT_DECIDED = "experiment_decided"
    DELIVERABLE_CREATED = "deliverable_created"


class AuditEvent(BaseModel):
    """One thing that happened, at a time, to a named subject."""

    at: datetime
    kind: AuditEventKind
    subject: str
    """The id this event is about — a project, plan, experiment, run, paper set."""

    summary: str
    detail: dict[str, str] = Field(default_factory=dict)


def project_events(project: Project | None) -> list[AuditEvent]:
    if project is None:
        return []
    mode = f"{project.mode.value} mode" if project.mode is not None else "no mode set"
    return [
        AuditEvent(
            at=project.created_at,
            kind=AuditEventKind.PROJECT_CREATED,
            subject=project.id,
            summary=f"Project created in {mode}: {project.name}",
            detail={"objective": project.objective} if project.objective else {},
        )
    ]


def search_events(search_runs: list[dict[str, object]]) -> list[AuditEvent]:
    """One event per arXiv search, from the search-run rows.

    Search runs are stored as plain rows rather than a model, so the fields are
    read defensively: a row missing its timestamp is skipped instead of guessed.
    """
    events = []
    for row in search_runs:
        created = row.get("created_at")
        if not isinstance(created, str):
            continue
        queries = row.get("queries")
        events.append(
            AuditEvent(
                at=datetime.fromisoformat(created),
                kind=AuditEventKind.PAPERS_SEARCHED,
                subject=str(row.get("run_id", "")),
                summary=(
                    f"Searched arXiv: {row.get('fetched_count', 0)} fetched, "
                    f"{row.get('selected_count', 0)} stored"
                ),
                detail={"queries": str(queries)} if queries else {},
            )
        )
    return events


def landscape_events(imported_at: datetime | None, direction_count: int) -> list[AuditEvent]:
    if imported_at is None:
        return []
    return [
        AuditEvent(
            at=imported_at,
            kind=AuditEventKind.LANDSCAPE_IMPORTED,
            subject="landscape",
            summary=f"Research landscape imported with {direction_count} direction(s)",
        )
    ]


def hypothesis_events(hypotheses: list[Hypothesis]) -> list[AuditEvent]:
    """One event per reviewed hypothesis.

    Unreviewed hypotheses contribute nothing: "nobody has decided yet" is a
    state, not an event, and the import itself is covered by the landscape.
    """
    return [
        AuditEvent(
            at=hypothesis.review.decided_at,
            kind=AuditEventKind.HYPOTHESIS_REVIEWED,
            subject=hypothesis.hypothesis_id,
            summary=(
                f"Hypothesis {hypothesis.review.decision.value}: {hypothesis.title}"
            ),
            detail={"reason": hypothesis.review.reason} if hypothesis.review.reason else {},
        )
        for hypothesis in hypotheses
        if hypothesis.review is not None
    ]


def contract_events(contracts: list[ExperimentContract]) -> list[AuditEvent]:
    return [
        AuditEvent(
            at=contract.approved_at,
            kind=AuditEventKind.CONTRACT_APPROVED,
            subject=f"{contract.contract_id} v{contract.contract_version}",
            summary=(
                f"Contract v{contract.contract_version} approved, freezing the "
                f"baseline at {contract.baseline_commit[:8]}"
            ),
            detail={
                "metric": contract.spec.objective.primary_metric.name,
                "direction": contract.spec.objective.primary_metric.direction.value,
                "baseline_commit": contract.baseline_commit,
            },
        )
        for contract in contracts
    ]


def baseline_events(runs: list[BaselineRun]) -> list[AuditEvent]:
    events = []
    for run in runs:
        succeeded = run.status is BaselineStatus.SUCCEEDED
        if succeeded and run.metrics is not None:
            metric = run.metrics.primary_metric
            summary = f"Baseline frozen: {metric.name} = {metric.value:.4g}"
        else:
            summary = f"Baseline attempt {run.status.value}"
        events.append(
            AuditEvent(
                at=run.completed_at,
                kind=AuditEventKind.BASELINE_MEASURED,
                subject=run.baseline_id,
                summary=summary,
                detail={"commit": run.commit_sha, "status": run.status.value},
            )
        )
    return events


def plan_events(plans: list[ExperimentPlan], experiments: list[Experiment]) -> list[AuditEvent]:
    """Creation for every plan, plus approval for the ones that got it.

    A plan's approval records how it was given — a typed confirmation from a
    person, or a flag on an unattended run — which is exactly the distinction an
    audit is asked about, so it is reported rather than flattened away.
    """
    events = []
    for plan in plans:
        planned = sum(1 for e in experiments if e.plan_id == plan.plan_id)
        events.append(
            AuditEvent(
                at=plan.created_at,
                kind=AuditEventKind.PLAN_CREATED,
                subject=plan.plan_id,
                summary=(
                    f"Plan {plan.plan_id} created for {plan.hypothesis_id} "
                    f"({planned} experiment(s))"
                ),
            )
        )
        if plan.approval is None:
            continue
        approval = plan.approval
        gate = "typed confirmation" if approval.method == "typed" else "--yes flag"
        events.append(
            AuditEvent(
                at=approval.approved_at,
                kind=AuditEventKind.PLAN_APPROVED,
                subject=plan.plan_id,
                summary=(
                    f"Plan {plan.plan_id} approved by {gate} — "
                    f"{len(approval.experiment_ids)} experiment(s), "
                    f"~{approval.estimated_max_minutes} min worst case"
                ),
                detail={
                    "method": approval.method,
                    "experiments": ", ".join(approval.experiment_ids),
                },
            )
        )
    return events


def run_events(runs: list[ExperimentRunGroup]) -> list[AuditEvent]:
    events = []
    for run in runs:
        events.append(
            AuditEvent(
                at=run.started_at,
                kind=AuditEventKind.RUN_STARTED,
                subject=run.run_id,
                summary=f"Run {run.run_id} started for {run.plan_id}",
            )
        )
        if run.completed_at is not None:
            events.append(
                AuditEvent(
                    at=run.completed_at,
                    kind=AuditEventKind.RUN_COMPLETED,
                    subject=run.run_id,
                    summary=f"Run {run.run_id} {run.status.value}",
                )
            )
    return events


def execution_events(executions: list[ExperimentExecution]) -> list[AuditEvent]:
    """One event per benchmark attempt — the evidence everything else rests on."""
    events = []
    for execution in executions:
        measured = ""
        if execution.metrics is not None:
            metric = execution.metrics.primary_metric
            measured = f" → {metric.name} = {metric.value:.4g}"
        events.append(
            AuditEvent(
                at=execution.completed_at or execution.started_at,
                kind=AuditEventKind.BENCHMARK_RAN,
                subject=execution.experiment_id,
                summary=(
                    f"{execution.benchmark_stage.value} benchmark attempt "
                    f"{execution.attempt} {execution.status.value}{measured}"
                ),
                detail={
                    "run_id": execution.run_id,
                    "stage": execution.benchmark_stage.value,
                    "duration_seconds": f"{execution.duration_seconds:.1f}",
                },
            )
        )
    return events


def experiment_events(experiments: list[Experiment]) -> list[AuditEvent]:
    """The decision recorded against each experiment, where one was reached.

    `updated_at` is when the record last changed, which for a decided experiment
    is when its decision landed. It is reported as the decision's time because
    that is the closest thing the records preserve, and no better time exists.
    """
    return [
        AuditEvent(
            at=experiment.updated_at,
            kind=AuditEventKind.EXPERIMENT_DECIDED,
            subject=experiment.experiment_id,
            summary=(
                f"{experiment.experiment_id} {experiment.decision.outcome.value}: "
                f"{experiment.decision.reason}"
            ),
            detail={"status": experiment.status.value, "title": experiment.title},
        )
        for experiment in experiments
        if experiment.decision is not None
    ]


def deliverable_events(deliverables: list[Deliverable]) -> list[AuditEvent]:
    return [
        AuditEvent(
            at=deliverable.created_at,
            kind=AuditEventKind.DELIVERABLE_CREATED,
            subject=deliverable.experiment_id or deliverable.kind.value,
            summary=(
                f"{deliverable.kind.value.replace('_', ' ')} created: {deliverable.location}"
            ),
            detail=(
                {"commit": deliverable.commit_sha} if deliverable.commit_sha else {}
            ),
        )
        for deliverable in deliverables
    ]


def order_events(events: list[AuditEvent]) -> list[AuditEvent]:
    """Oldest first, with a stable order for events sharing a timestamp.

    Records written in the same transaction share a timestamp to the microsecond,
    so kind and subject break the tie — otherwise the trail would reshuffle
    between reads of the same database.
    """
    return sorted(events, key=lambda event: (event.at, event.kind.value, event.subject))


def unapproved_plans(plans: list[ExperimentPlan]) -> list[str]:
    """Plans that ran without an approval on record — nothing should be here.

    Approval is a gate, so an approved-or-later plan with no approval record is
    the one finding an audit exists to surface. Reported separately rather than
    as an event, because the absence of a record has no timestamp.
    """
    return [
        plan.plan_id
        for plan in plans
        if plan.approval is None and plan.status is not PlanStatus.PLANNED
        and plan.status is not PlanStatus.CANCELLED
    ]
