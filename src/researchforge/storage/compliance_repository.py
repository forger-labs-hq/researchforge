"""Persistence for compliance runs, results, approvals, and waivers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from researchforge.compliance.gates import ComplianceRun

# ── compliance runs ──────────────────────────────────────────────────────────


def insert_compliance_run(
    conn: sqlite3.Connection,
    project_id: str,
    run: ComplianceRun,
) -> None:
    """Persist a compliance run and all its individual gate results."""
    now = run.created_at.isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO compliance_runs
                (compliance_run_id, project_id, experiment_id, checkpoint,
                 policy_hash, policy_path, record, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.compliance_run_id,
                project_id,
                run.experiment_id,
                run.checkpoint,
                run.policy_hash,
                run.policy_path,
                run.model_dump_json(),
                now,
            ),
        )
        for result in run.results:
            conn.execute(
                """
                INSERT INTO compliance_results
                    (result_id, compliance_run_id, experiment_id, gate_id,
                     category, status, severity, record, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"res-{uuid4().hex[:12]}",
                    run.compliance_run_id,
                    run.experiment_id,
                    result.gate_id,
                    result.category,
                    result.status,
                    result.severity,
                    result.model_dump_json(),
                    now,
                ),
            )


def get_compliance_runs(conn: sqlite3.Connection, experiment_id: str) -> list[ComplianceRun]:
    """Return all compliance runs for an experiment, oldest first."""
    rows = conn.execute(
        "SELECT record FROM compliance_runs WHERE experiment_id = ? ORDER BY created_at",
        (experiment_id,),
    ).fetchall()
    return [ComplianceRun.model_validate_json(r["record"]) for r in rows]


def get_latest_compliance_run(
    conn: sqlite3.Connection,
    experiment_id: str,
    checkpoint: str | None = None,
) -> ComplianceRun | None:
    """Return the most recent compliance run, optionally filtered by checkpoint."""
    if checkpoint:
        row = conn.execute(
            "SELECT record FROM compliance_runs "
            "WHERE experiment_id = ? AND checkpoint = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (experiment_id, checkpoint),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT record FROM compliance_runs "
            "WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
    return ComplianceRun.model_validate_json(row["record"]) if row else None


# ── approvals ────────────────────────────────────────────────────────────────


def insert_approval(
    conn: sqlite3.Connection,
    project_id: str,
    experiment_id: str,
    actor: str,
    reason: str = "",
    policy_hash: str = "",
) -> str:
    """Record a human approval.  Returns the new approval_id."""
    approval_id = f"appr-{uuid4().hex[:12]}"
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO compliance_approvals
                (approval_id, experiment_id, project_id, actor, reason,
                 policy_hash, revoked, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (approval_id, experiment_id, project_id, actor, reason, policy_hash, now),
        )
    return approval_id


def get_approval(conn: sqlite3.Connection, experiment_id: str) -> dict[str, Any] | None:
    """Return the active (non-revoked) approval for an experiment, or None."""
    row = conn.execute(
        "SELECT * FROM compliance_approvals WHERE experiment_id = ? AND revoked = 0",
        (experiment_id,),
    ).fetchone()
    return dict(row) if row else None


# ── waivers ──────────────────────────────────────────────────────────────────


def insert_waiver(
    conn: sqlite3.Connection,
    project_id: str,
    experiment_id: str,
    gate_id: str,
    reason: str,
    actor: str,
    expires_at: str | None = None,
    policy_hash: str = "",
) -> str:
    """Record a compliance waiver.  Returns the new waiver_id."""
    waiver_id = f"waiv-{uuid4().hex[:12]}"
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO compliance_waivers
                (waiver_id, experiment_id, project_id, gate_id, reason,
                 actor, expires_at, policy_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                waiver_id,
                experiment_id,
                project_id,
                gate_id,
                reason,
                actor,
                expires_at,
                policy_hash,
                now,
            ),
        )
    return waiver_id


def list_waivers(conn: sqlite3.Connection, experiment_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM compliance_waivers WHERE experiment_id = ? ORDER BY created_at",
        (experiment_id,),
    ).fetchall()
    return [dict(r) for r in rows]
