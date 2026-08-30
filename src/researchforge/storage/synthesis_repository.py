"""Persistence for the research landscape and evidence claims."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

from researchforge.domain.landscape import ResearchLandscape


def replace_landscape(
    conn: sqlite3.Connection,
    project_id: str,
    landscape: ResearchLandscape,
    source_file: str | None = None,
) -> None:
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.execute("DELETE FROM landscape WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM evidence_claims WHERE project_id = ?", (project_id,))
        conn.execute(
            "INSERT INTO landscape (id, project_id, record, source_file, imported_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid4().hex, project_id, landscape.model_dump_json(), source_file, now),
        )
        for claim in landscape.evidence:
            conn.execute(
                """
                INSERT INTO evidence_claims
                    (evidence_id, project_id, paper_id, evidence_type, record, imported_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.evidence_id,
                    project_id,
                    claim.paper_id,
                    claim.evidence_type.value,
                    claim.model_dump_json(),
                    now,
                ),
            )


def get_landscape(conn: sqlite3.Connection) -> ResearchLandscape | None:
    row = conn.execute("SELECT record FROM landscape ORDER BY imported_at DESC LIMIT 1").fetchone()
    return ResearchLandscape.model_validate_json(row["record"]) if row is not None else None


def landscape_imported_at(conn: sqlite3.Connection) -> datetime | None:
    """When the stored landscape was imported, for reporting a project's history.

    The time lives in the table rather than on the model: it says when this
    project took the landscape in, which is not a property of the landscape.
    """
    row = conn.execute(
        "SELECT imported_at FROM landscape ORDER BY imported_at DESC LIMIT 1"
    ).fetchone()
    return datetime.fromisoformat(row["imported_at"]) if row is not None else None
