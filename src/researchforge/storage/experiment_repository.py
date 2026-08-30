"""Persistence for experiment plans, experiments, run groups, and executions."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from researchforge.domain.experiment import (
    Experiment,
    ExperimentExecution,
    ExperimentPlan,
    ExperimentRunGroup,
    PlanApproval,
    PlanStatus,
    RunStatus,
)
from researchforge.experiments.graph import ParentsFn


def _next_sequential_id(conn: sqlite3.Connection, table: str, column: str, prefix: str) -> int:
    rows = conn.execute(f"SELECT {column} FROM {table}").fetchall()  # noqa: S608 — fixed identifiers
    highest = 0
    for row in rows:
        value = row[column]
        if value.startswith(prefix):
            try:
                highest = max(highest, int(value.removeprefix(prefix)))
            except ValueError:
                continue
    return highest + 1


def next_plan_id(conn: sqlite3.Connection) -> str:
    return f"plan-{_next_sequential_id(conn, 'experiment_plans', 'plan_id', 'plan-'):03d}"


def next_run_id(conn: sqlite3.Connection) -> str:
    return f"run-{_next_sequential_id(conn, 'experiment_runs', 'run_id', 'run-'):03d}"


def next_experiment_ids(conn: sqlite3.Connection, count: int) -> list[str]:
    start = _next_sequential_id(conn, "experiments", "experiment_id", "exp-")
    return [f"exp-{start + offset:03d}" for offset in range(count)]


def insert_plan(
    conn: sqlite3.Connection,
    project_id: str,
    plan: ExperimentPlan,
    experiments: list[Experiment],
) -> None:
    """Persist a plan and its experiments in one transaction."""
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO experiment_plans
                (plan_id, project_id, hypothesis_id, contract_id, contract_version,
                 baseline_id, status, record, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.plan_id,
                project_id,
                plan.hypothesis_id,
                plan.contract_id,
                plan.contract_version,
                plan.baseline_id,
                plan.status.value,
                plan.model_dump_json(),
                now,
                now,
            ),
        )
        for experiment in experiments:
            conn.execute(
                """
                INSERT INTO experiments
                    (experiment_id, project_id, plan_id, hypothesis_id, status,
                     record, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    project_id,
                    experiment.plan_id,
                    experiment.hypothesis_id,
                    experiment.status.value,
                    experiment.model_dump_json(),
                    now,
                    now,
                ),
            )


def get_plan(conn: sqlite3.Connection, plan_id: str) -> ExperimentPlan | None:
    row = conn.execute(
        "SELECT record FROM experiment_plans WHERE plan_id = ?", (plan_id,)
    ).fetchone()
    return ExperimentPlan.model_validate_json(row["record"]) if row is not None else None


def list_plans(conn: sqlite3.Connection) -> list[ExperimentPlan]:
    rows = conn.execute("SELECT record FROM experiment_plans ORDER BY plan_id").fetchall()
    return [ExperimentPlan.model_validate_json(row["record"]) for row in rows]


def update_plan_status(
    conn: sqlite3.Connection,
    plan_id: str,
    status: PlanStatus,
    approval: PlanApproval | None = None,
) -> None:
    plan = get_plan(conn, plan_id)
    if plan is None:
        raise ValueError(f"Unknown plan: {plan_id}")
    updated = plan.model_copy(
        update={
            "status": status,
            "approval": approval if approval is not None else plan.approval,
            "updated_at": datetime.now(UTC),
        }
    )
    with conn:
        conn.execute(
            "UPDATE experiment_plans SET status = ?, record = ?, updated_at = ? WHERE plan_id = ?",
            (status.value, updated.model_dump_json(), updated.updated_at.isoformat(), plan_id),
        )


def get_experiment(conn: sqlite3.Connection, experiment_id: str) -> Experiment | None:
    row = conn.execute(
        "SELECT record FROM experiments WHERE experiment_id = ?", (experiment_id,)
    ).fetchone()
    return Experiment.model_validate_json(row["record"]) if row is not None else None


def stored_parents(conn: sqlite3.Connection) -> ParentsFn:
    """Lineage lookup for graph walks over experiments already persisted."""

    def parents_of(experiment_id: str) -> list[str]:
        stored = get_experiment(conn, experiment_id)
        return list(stored.parent_experiment_ids) if stored else []

    return parents_of


def patch_ancestry(conn: sqlite3.Connection) -> ParentsFn:
    """Lineage lookup for composing patches, rather than for showing provenance.

    An experiment whose patch already contains its parents' changes reads as a
    root here: applying those parents again would conflict with the very diff
    that was written to replace them.  The walk therefore stops at such a node
    for its descendants too, which is exactly what they need — the node's own
    patch carries the whole combined state.
    """

    def parents_of(experiment_id: str) -> list[str]:
        stored = get_experiment(conn, experiment_id)
        if stored is None or stored.patch_includes_parents:
            return []
        return list(stored.parent_experiment_ids)

    return parents_of


def list_experiments(conn: sqlite3.Connection, plan_id: str | None = None) -> list[Experiment]:
    if plan_id is not None:
        rows = conn.execute(
            "SELECT record FROM experiments WHERE plan_id = ? ORDER BY experiment_id",
            (plan_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT record FROM experiments ORDER BY experiment_id").fetchall()
    return [Experiment.model_validate_json(row["record"]) for row in rows]


def update_experiment(conn: sqlite3.Connection, experiment: Experiment) -> None:
    updated = experiment.model_copy(update={"updated_at": datetime.now(UTC)})
    with conn:
        conn.execute(
            "UPDATE experiments SET status = ?, record = ?, updated_at = ? WHERE experiment_id = ?",
            (
                updated.status.value,
                updated.model_dump_json(),
                updated.updated_at.isoformat(),
                updated.experiment_id,
            ),
        )


def insert_run(conn: sqlite3.Connection, project_id: str, run: ExperimentRunGroup) -> None:
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO experiment_runs
                (run_id, project_id, plan_id, status, record, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                project_id,
                run.plan_id,
                run.status.value,
                run.model_dump_json(),
                now,
                now,
            ),
        )


def get_run(conn: sqlite3.Connection, run_id: str) -> ExperimentRunGroup | None:
    row = conn.execute("SELECT record FROM experiment_runs WHERE run_id = ?", (run_id,)).fetchone()
    return ExperimentRunGroup.model_validate_json(row["record"]) if row is not None else None


def list_runs(conn: sqlite3.Connection) -> list[ExperimentRunGroup]:
    rows = conn.execute("SELECT record FROM experiment_runs ORDER BY run_id").fetchall()
    return [ExperimentRunGroup.model_validate_json(row["record"]) for row in rows]


def update_run(conn: sqlite3.Connection, run: ExperimentRunGroup) -> None:
    with conn:
        conn.execute(
            "UPDATE experiment_runs SET status = ?, record = ?, updated_at = ? WHERE run_id = ?",
            (
                run.status.value,
                run.model_dump_json(),
                datetime.now(UTC).isoformat(),
                run.run_id,
            ),
        )


def get_open_run_for_plan(conn: sqlite3.Connection, plan_id: str) -> ExperimentRunGroup | None:
    row = conn.execute(
        "SELECT record FROM experiment_runs WHERE plan_id = ? AND status = ? "
        "ORDER BY run_id DESC LIMIT 1",
        (plan_id, RunStatus.IN_PROGRESS.value),
    ).fetchone()
    return ExperimentRunGroup.model_validate_json(row["record"]) if row is not None else None


def insert_execution(
    conn: sqlite3.Connection, project_id: str, execution: ExperimentExecution
) -> None:
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO experiment_executions
                (execution_id, project_id, run_id, experiment_id, benchmark_stage,
                 attempt, status, record, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution.execution_id,
                project_id,
                execution.run_id,
                execution.experiment_id,
                execution.benchmark_stage.value,
                execution.attempt,
                execution.status.value,
                execution.model_dump_json(),
                now,
                now,
            ),
        )


def update_execution(conn: sqlite3.Connection, execution: ExperimentExecution) -> None:
    with conn:
        conn.execute(
            "UPDATE experiment_executions SET status = ?, record = ?, updated_at = ? "
            "WHERE execution_id = ?",
            (
                execution.status.value,
                execution.model_dump_json(),
                datetime.now(UTC).isoformat(),
                execution.execution_id,
            ),
        )


def list_executions(
    conn: sqlite3.Connection,
    run_id: str | None = None,
    experiment_id: str | None = None,
) -> list[ExperimentExecution]:
    query = "SELECT record FROM experiment_executions"
    clauses = []
    params: list[str] = []
    if run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)
    if experiment_id is not None:
        clauses.append("experiment_id = ?")
        params.append(experiment_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at, execution_id"
    rows = conn.execute(query, params).fetchall()
    return [ExperimentExecution.model_validate_json(row["record"]) for row in rows]
