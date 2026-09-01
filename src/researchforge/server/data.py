"""Read-only project snapshot for the monitoring server.

Connections are opened with sqlite's `mode=ro` URI so the server
structurally cannot mutate project state — a monitoring read can never
interfere with a run in progress.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from researchforge.ai.usage import AiCall
from researchforge.config.paths import contract_path, db_path
from researchforge.domain.baseline import BaselineRun, BaselineStatus
from researchforge.domain.contract import ExperimentContract
from researchforge.domain.deliverable import Deliverable
from researchforge.domain.experiment import (
    Experiment,
    ExperimentExecution,
    ExperimentPlan,
    ExperimentRunGroup,
)
from researchforge.domain.hypothesis import Hypothesis
from researchforge.domain.landscape import ResearchLandscape
from researchforge.domain.paper import Paper
from researchforge.domain.project import Project


def open_readonly(base: Path | None = None) -> sqlite3.Connection:
    """Open the project db read-only; raises FileNotFoundError when absent."""
    path = db_path(base)
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


class ProjectState(BaseModel):
    """Everything the monitoring pages show, as one snapshot."""

    # Presentation-only: URL prefix for every in-page link ("" when the
    # monitor serves one project at the root; "/p/<slug>" under the hub).
    link_prefix: str = Field(default="", exclude=True)

    project: Project
    next_action: str
    papers: list[Paper] = Field(default_factory=list)
    search_runs: list[dict[str, object]] = Field(default_factory=list)
    search_run_papers: dict[str, list[str]] = Field(default_factory=dict)
    landscape: ResearchLandscape | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    contract: ExperimentContract | None = None
    baseline: BaselineRun | None = None
    plans: list[ExperimentPlan] = Field(default_factory=list)
    runs: list[ExperimentRunGroup] = Field(default_factory=list)
    experiments: list[Experiment] = Field(default_factory=list)
    executions: list[ExperimentExecution] = Field(default_factory=list)
    deliverables: list[Deliverable] = Field(default_factory=list)
    ai_calls: list[AiCall] = Field(default_factory=list)

    @property
    def run_in_progress(self) -> bool:
        return any(run.status.value == "in_progress" for run in self.runs)


def read_state(base: Path | None = None) -> ProjectState:
    from contextlib import closing

    from researchforge.cli import _next_action
    from researchforge.contract.service import check_contract_drift
    from researchforge.storage.ai_call_repository import list_ai_calls
    from researchforge.storage.baseline_repository import get_latest_baseline
    from researchforge.storage.contract_repository import get_active_contract
    from researchforge.storage.deliverable_repository import list_deliverables
    from researchforge.storage.experiment_repository import (
        list_executions,
        list_experiments,
        list_plans,
        list_runs,
    )
    from researchforge.storage.hypothesis_repository import list_hypotheses
    from researchforge.storage.paper_repository import (
        list_papers,
        list_search_runs,
        papers_for_search_run,
    )
    from researchforge.storage.project_repository import get_project
    from researchforge.storage.synthesis_repository import get_landscape

    with closing(open_readonly(base)) as conn:
        project = get_project(conn)
        if project is None:
            raise FileNotFoundError("no project record")
        papers = list_papers(conn)
        landscape = get_landscape(conn)
        hypotheses = list_hypotheses(conn)
        contract = get_active_contract(conn)
        latest_baseline = get_latest_baseline(conn)
        repo_root = Path(project.repository.path) if project.repository.path else Path.cwd()
        drifted = check_contract_drift(conn, contract_path(repo_root))
        next_action = _next_action(
            project,
            len(papers),
            len(hypotheses),
            1 if landscape is not None else 0,
            contract_version=contract.contract_version if contract else None,
            contract_drifted=drifted,
            baseline_failed=(
                latest_baseline is not None
                and latest_baseline.status is not BaselineStatus.SUCCEEDED
            ),
            conn=conn,
        )
        search_runs = list_search_runs(conn)
        return ProjectState(
            project=project,
            next_action=next_action,
            papers=papers,
            search_runs=search_runs,
            search_run_papers={
                str(run["run_id"]): papers_for_search_run(conn, str(run["run_id"]))
                for run in search_runs
            },
            landscape=landscape,
            hypotheses=hypotheses,
            contract=contract,
            baseline=latest_baseline,
            plans=list_plans(conn),
            runs=list_runs(conn),
            experiments=list_experiments(conn),
            executions=list_executions(conn),
            deliverables=list_deliverables(conn),
            ai_calls=list_ai_calls(conn),
        )
