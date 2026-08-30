"""Persistence for hypotheses."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from researchforge.domain.hypothesis import Hypothesis, HypothesisReview, ReviewOutcome


def replace_hypotheses(
    conn: sqlite3.Connection, project_id: str, hypotheses: list[Hypothesis]
) -> None:
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.execute("DELETE FROM hypotheses WHERE project_id = ?", (project_id,))
        for hypothesis in hypotheses:
            conn.execute(
                """
                INSERT INTO hypotheses
                    (hypothesis_id, project_id, title, status, record, imported_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis.hypothesis_id,
                    project_id,
                    hypothesis.title,
                    hypothesis.status.value,
                    hypothesis.model_dump_json(),
                    now,
                    now,
                ),
            )


def list_hypotheses(conn: sqlite3.Connection) -> list[Hypothesis]:
    rows = conn.execute("SELECT record FROM hypotheses ORDER BY hypothesis_id").fetchall()
    return [Hypothesis.model_validate_json(row["record"]) for row in rows]


def next_hypothesis_ids(conn: sqlite3.Connection, count: int) -> list[str]:
    """`count` unused ids, continuing after the highest one stored.

    Numbering never reuses a retired id: plans and experiments refer to
    hypotheses by id, so a recycled id would silently re-point that history.
    """
    rows = conn.execute("SELECT hypothesis_id FROM hypotheses").fetchall()
    used = [int(row["hypothesis_id"].removeprefix("hyp-")) for row in rows]
    start = max(used, default=0) + 1
    return [f"hyp-{number:03d}" for number in range(start, start + count)]


def record_review(
    conn: sqlite3.Connection,
    hypothesis_id: str,
    decision: ReviewOutcome,
    reason: str = "",
) -> Hypothesis | None:
    """Approve or reject one hypothesis, leaving the rest untouched.

    Returns the reviewed hypothesis, or None when the id is unknown. Reviewing
    is the one operation that changes a single hypothesis: everything else
    rewrites the whole set, which would discard a review made in between.
    """
    stored = get_hypothesis(conn, hypothesis_id)
    if stored is None:
        return None

    reviewed = stored.model_copy(
        update={
            "status": decision,
            "review": HypothesisReview(
                decision=decision, reason=reason, decided_at=datetime.now(UTC)
            ),
        }
    )
    with conn:
        conn.execute(
            """
            UPDATE hypotheses SET status = ?, record = ?, updated_at = ?
            WHERE hypothesis_id = ?
            """,
            (
                reviewed.status.value,
                reviewed.model_dump_json(),
                datetime.now(UTC).isoformat(),
                hypothesis_id,
            ),
        )
    return reviewed


def get_hypothesis(conn: sqlite3.Connection, hypothesis_id: str) -> Hypothesis | None:
    row = conn.execute(
        "SELECT record FROM hypotheses WHERE hypothesis_id = ?", (hypothesis_id,)
    ).fetchone()
    return Hypothesis.model_validate_json(row["record"]) if row is not None else None
