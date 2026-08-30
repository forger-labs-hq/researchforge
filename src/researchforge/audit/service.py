"""Reading the project's records and handing them to the trail builder."""

from __future__ import annotations

import sqlite3

from pydantic import BaseModel, Field

from researchforge.audit.trail import (
    AuditEvent,
    baseline_events,
    contract_events,
    deliverable_events,
    execution_events,
    experiment_events,
    hypothesis_events,
    landscape_events,
    order_events,
    plan_events,
    project_events,
    run_events,
    search_events,
    unapproved_plans,
)
from researchforge.storage.baseline_repository import list_baseline_runs
from researchforge.storage.contract_repository import list_contracts
from researchforge.storage.deliverable_repository import list_deliverables
from researchforge.storage.experiment_repository import (
    list_executions,
    list_experiments,
    list_plans,
    list_runs,
)
from researchforge.storage.hypothesis_repository import list_hypotheses
from researchforge.storage.paper_repository import list_search_runs
from researchforge.storage.project_repository import get_project
from researchforge.storage.synthesis_repository import get_landscape, landscape_imported_at


class AuditTrail(BaseModel):
    """Everything the project's records say happened, in order."""

    events: list[AuditEvent] = Field(default_factory=list)
    gate_findings: list[str] = Field(default_factory=list)
    """Gates that should have left a record and did not — empty is the healthy case."""


def build_trail(conn: sqlite3.Connection) -> AuditTrail:
    """Assemble the trail from every record store that timestamps its writes."""
    plans = list_plans(conn)
    experiments = list_experiments(conn)
    landscape = get_landscape(conn)

    events = [
        *project_events(get_project(conn)),
        *search_events(list_search_runs(conn)),
        *landscape_events(
            landscape_imported_at(conn),
            len(landscape.directions) if landscape is not None else 0,
        ),
        *hypothesis_events(list_hypotheses(conn)),
        *contract_events(list_contracts(conn)),
        *baseline_events(list_baseline_runs(conn)),
        *plan_events(plans, experiments),
        *run_events(list_runs(conn)),
        *execution_events(list_executions(conn)),
        *experiment_events(experiments),
        *deliverable_events(list_deliverables(conn)),
    ]

    findings = [
        f"{plan_id} reached execution with no approval on record"
        for plan_id in unapproved_plans(plans)
    ]
    return AuditTrail(events=order_events(events), gate_findings=findings)
