"""SQLite connection and schema management for local project state.

Schema history:
- v1 (Phase 0): ``meta``, ``projects``.
- v2 (Phase 1A): ``repo_scans``, ``papers``, ``search_runs``, ``landscape``,
  ``evidence_claims``, ``hypotheses``.
- v3 (Phase 1B): ``contracts``, ``baseline_runs``.
- v4 (Phase 1C): ``experiment_plans``, ``experiments``, ``experiment_runs``,
  ``experiment_executions``.
- v5 (Phase 1D): ``deliverables``.

All migrations are additive ``CREATE TABLE IF NOT EXISTS`` statements, so
``ensure_schema`` can run on every connection open and silently upgrade
older databases.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from researchforge.config.paths import db_path

SCHEMA_VERSION = 8

_V1_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        mode TEXT,
        objective TEXT,
        repository_metadata TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
]

_V2_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS repo_scans (
        scan_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        compatibility TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        title TEXT NOT NULL,
        published_at TEXT NOT NULL,
        relevance_score REAL NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS search_runs (
        run_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        queries TEXT NOT NULL,
        fetched_count INTEGER NOT NULL,
        deduped_count INTEGER NOT NULL,
        selected_count INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS landscape (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        record TEXT NOT NULL,
        source_file TEXT,
        imported_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_claims (
        evidence_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        paper_id TEXT NOT NULL,
        evidence_type TEXT NOT NULL,
        record TEXT NOT NULL,
        imported_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hypotheses (
        hypothesis_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        title TEXT NOT NULL,
        status TEXT NOT NULL,
        record TEXT NOT NULL,
        imported_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
]

_V3_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS contracts (
        contract_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        source_sha256 TEXT NOT NULL,
        baseline_commit TEXT NOT NULL,
        approved_at TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (project_id, contract_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS baseline_runs (
        baseline_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        contract_id TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        commit_sha TEXT NOT NULL,
        execution_mode TEXT NOT NULL,
        status TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
]

_V4_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS experiment_plans (
        plan_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        hypothesis_id TEXT NOT NULL,
        contract_id TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        baseline_id TEXT NOT NULL,
        status TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        plan_id TEXT NOT NULL,
        hypothesis_id TEXT NOT NULL,
        status TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_runs (
        run_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        plan_id TEXT NOT NULL,
        status TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_executions (
        execution_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        experiment_id TEXT NOT NULL,
        benchmark_stage TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        status TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (experiment_id, benchmark_stage, attempt)
    )
    """,
]

_V5_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS deliverables (
        deliverable_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        experiment_id TEXT,
        location TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
]

_V6_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS search_run_papers (
        run_id TEXT NOT NULL,
        paper_id TEXT NOT NULL,
        PRIMARY KEY (run_id, paper_id)
    )
    """,
]

_V7_TABLES = [
    # One row per model call. Tokens are recorded rather than dollars: prices
    # change and are configuration, so the durable fact is the token count and
    # the model that produced it, priced at read time.
    """
    CREATE TABLE IF NOT EXISTS ai_calls (
        call_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        purpose TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        duration_seconds REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        estimated INTEGER NOT NULL DEFAULT 0
    )
    """,
]

# Sizing an IDE handshake is an estimate, and an estimate must never be
# mistaken for a metered count — hence a column rather than a convention.
_V8_TABLES = [
    "ALTER TABLE ai_calls ADD COLUMN estimated INTEGER NOT NULL DEFAULT 0",
]

_MIGRATIONS: dict[int, list[str]] = {
    1: _V1_TABLES,
    2: _V2_TABLES,
    3: _V3_TABLES,
    4: _V4_TABLES,
    5: _V5_TABLES,
    6: _V6_TABLES,
    7: _V7_TABLES,
    8: _V8_TABLES,
}


def get_connection(path: Path) -> sqlite3.Connection:
    """Open a sqlite connection at `path`, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create any missing tables and record the current schema version."""
    with conn:
        for version in sorted(_MIGRATIONS):
            if version <= SCHEMA_VERSION:
                for ddl in _MIGRATIONS[version]:
                    try:
                        conn.execute(ddl)
                    except sqlite3.OperationalError:
                        # `CREATE TABLE IF NOT EXISTS` is idempotent; `ALTER TABLE
                        # ADD COLUMN` is not, and a database created fresh already
                        # has the column its migration would add.
                        if not ddl.lstrip().upper().startswith("ALTER TABLE"):
                            raise
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Backward-compatible alias for `ensure_schema`."""
    ensure_schema(conn)


def open_project_db(base: Path | None = None) -> sqlite3.Connection:
    """Open (and, if needed, upgrade) the project database under `base`."""
    conn = get_connection(db_path(base))
    ensure_schema(conn)
    return conn
