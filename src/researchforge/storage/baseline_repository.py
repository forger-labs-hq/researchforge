"""Persistence for baseline runs. Every terminal status is stored — failures
are first-class records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from researchforge.domain.baseline import BaselineRun, BaselineStatus


def insert_baseline_run(conn: sqlite3.Connection, project_id: str, run: BaselineRun) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO baseline_runs
                (baseline_id, project_id, contract_id, contract_version, commit_sha,
                 execution_mode, status, record, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.baseline_id,
                project_id,
                run.contract_id,
                run.contract_version,
                run.commit_sha,
                run.execution_mode.value,
                run.status.value,
                run.model_dump_json(),
                datetime.now(UTC).isoformat(),
            ),
        )


def get_latest_baseline(conn: sqlite3.Connection, command_kind: str = "full") -> BaselineRun | None:
    """Latest baseline of the given kind (per-run screening baselines are
    stored with command_kind='screening' and never satisfy the gate)."""
    row = conn.execute(
        "SELECT record FROM baseline_runs "
        "WHERE json_extract(record, '$.command_kind') = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (command_kind,),
    ).fetchone()
    return BaselineRun.model_validate_json(row["record"]) if row is not None else None


def list_baseline_runs(
    conn: sqlite3.Connection, command_kind: str | None = "full"
) -> list[BaselineRun]:
    """Every baseline attempt, oldest first, failures included.

    `command_kind=None` returns the per-run screening baselines as well, which
    only an audit of everything that ran should want.
    """
    if command_kind is None:
        rows = conn.execute("SELECT record FROM baseline_runs ORDER BY created_at").fetchall()
    else:
        rows = conn.execute(
            "SELECT record FROM baseline_runs "
            "WHERE json_extract(record, '$.command_kind') = ? ORDER BY created_at",
            (command_kind,),
        ).fetchall()
    return [BaselineRun.model_validate_json(row["record"]) for row in rows]


def delete_baselines(conn: sqlite3.Connection, command_kind: str = "full") -> int:
    """Remove the frozen baselines of one kind. Returns how many were removed.

    Callers are responsible for deciding whether this is safe: every experiment
    measured against a baseline loses its reference point when it is dropped.
    """
    with conn:
        cursor = conn.execute(
            "DELETE FROM baseline_runs WHERE json_extract(record, '$.command_kind') = ?",
            (command_kind,),
        )
    return cursor.rowcount


def get_latest_successful_baseline(
    conn: sqlite3.Connection, command_kind: str = "full"
) -> BaselineRun | None:
    row = conn.execute(
        "SELECT record FROM baseline_runs "
        "WHERE status = ? AND json_extract(record, '$.command_kind') = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (BaselineStatus.SUCCEEDED.value, command_kind),
    ).fetchone()
    return BaselineRun.model_validate_json(row["record"]) if row is not None else None
