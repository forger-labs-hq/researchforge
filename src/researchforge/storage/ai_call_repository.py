"""Persistence for recorded model calls."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime

from researchforge.ai.usage import AiCall, recording


def insert_ai_calls(conn: sqlite3.Connection, project_id: str, calls: Sequence[AiCall]) -> None:
    """Store a batch of model calls. Calls with no usage are kept.

    A provider that reported nothing still made a request, and dropping those
    rows would understate how much the loop talked to the model — the count of
    calls is a fact even when the token count is not.
    """
    if not calls:
        return
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.executemany(
            """
            INSERT INTO ai_calls (
                call_id, project_id, purpose, provider, model,
                input_tokens, output_tokens, duration_seconds, created_at, estimated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    uuid.uuid4().hex,
                    project_id,
                    call.purpose,
                    call.provider,
                    call.model,
                    call.usage.input_tokens,
                    call.usage.output_tokens,
                    call.duration_seconds,
                    now,
                    int(call.estimated),
                )
                for call in calls
            ],
        )


@contextmanager
def metered(conn: sqlite3.Connection, project_id: str) -> Iterator[None]:
    """Record every model call made in the block, then persist it.

    Persists on the way out even when the block raises: a run that crashed
    halfway still spent the tokens it spent, and a crash is exactly when
    somebody wants to know what it cost before it died.
    """
    with recording() as ledger:
        try:
            yield
        finally:
            # Accounting must never be the reason real work fails.
            with suppress(sqlite3.Error):
                insert_ai_calls(conn, project_id, ledger.calls)


def list_ai_calls(conn: sqlite3.Connection) -> list[AiCall]:
    """Every recorded model call, oldest first.

    A project created before token accounting existed has no such table, and the
    monitor opens the database read-only so it cannot migrate one into place.
    That project simply has no calls to report, which is true.
    """
    from researchforge.ai.usage import Usage

    try:
        rows = conn.execute(
            "SELECT purpose, provider, model, input_tokens, output_tokens, "
            "duration_seconds, estimated FROM ai_calls ORDER BY created_at, call_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        AiCall(
            purpose=row["purpose"],
            provider=row["provider"],
            model=row["model"],
            usage=Usage(input_tokens=row["input_tokens"], output_tokens=row["output_tokens"]),
            duration_seconds=row["duration_seconds"],
            estimated=bool(row["estimated"]),
        )
        for row in rows
    ]
