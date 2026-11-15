"""One history per competition: the repo engine's work folded into the Kaggle loop.

A competition could be worked by two engines side by side: the Kaggle loop
(``.overnight_<slug>/experiment_log.db``, rounds on the chosen compute) and a
repo project beside it (``.researchforge/researchforge.db``, benchmarked on this
machine). Two histories for one competition confused everyone, so the Kaggle
loop is the only engine for a competition. This module copies the repo
project's experiments that ran into the Kaggle loop's log (table
``imported_experiments``) and its untested hypotheses into the pool. The repo
project itself is only read, never changed.

Imported experiments keep the repo engine's own measurement (its benchmark,
metric and baseline), which is not the Kaggle loop's. They are shown in the one
history but never compared with rounds: never "best", never a parent, never the
baseline that decides "improved".
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, read_json

TABLE = "imported_experiments"
MERGE_FILE = "history_merge.json"
_NOT_RUN = ("planned", "approved", "cancelled")


def repo_dir_for(artifacts_dir: Path) -> Path | None:
    """The repo project beside this competition, if it has a database."""
    repo = artifacts_dir.parent / ".researchforge"
    return repo if (repo / "researchforge.db").is_file() else None


def _repo_metric(repo_dir: Path) -> tuple[str, str]:
    """(metric name, direction) from the repo project's newest contract."""
    try:
        with sqlite3.connect(f"file:{repo_dir / 'researchforge.db'}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT record FROM contracts ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        spec = (json.loads(row[0]) if row else {}).get("spec") or {}
    except (sqlite3.Error, ValueError, TypeError):
        return "", ""
    metric = (spec.get("objective") or {}).get("primary_metric") or {}
    return str(metric.get("name") or ""), str(metric.get("direction") or "")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            repo_id      TEXT PRIMARY KEY,
            num          INTEGER,
            title        TEXT,
            status       TEXT,
            raw_status   TEXT,
            metric       REAL,
            metric_name  TEXT,
            direction    TEXT,
            baseline     REAL,
            parents      TEXT,
            reason       TEXT,
            stage        TEXT,
            started_at   TEXT,
            completed_at TEXT,
            patch        TEXT,
            change       TEXT,
            imported_at  TEXT
        )
    """)


def _repo_mtime(repo_dir: Path) -> float:
    paths = [repo_dir / n for n in ("researchforge.db", "researchforge.db-wal")]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def import_repo_history(artifacts_dir: Path, *, force: bool = False) -> dict[str, Any] | None:
    """Copy the repo project's run experiments and untested ideas into this loop.

    Idempotent: experiments are keyed by their repo id, ideas by their text. Skips
    work when the repo database has not changed since the last import (unless
    ``force``). Returns what was done, or None when there is no repo project.
    """
    repo_dir = repo_dir_for(artifacts_dir)
    if repo_dir is None:
        return None
    merge = read_json(artifacts_dir / MERGE_FILE, {}) or {}
    mtime = _repo_mtime(repo_dir)
    if not force and isinstance(merge, dict) and merge.get("repo_mtime") == mtime:
        return {**merge, "unchanged": True}

    from researchforge.kaggle.dashboard_state import _read_repo

    repo = _read_repo(artifacts_dir.parent, repo_dir)
    metric_name, direction = _repo_metric(repo_dir)
    baseline = (repo.get("baselines") or [{}])[0].get("metric") if repo.get("baselines") else None
    now = datetime.now().isoformat(timespec="seconds")
    ran = [e for e in repo.get("experiments", []) if e.get("raw_status") not in _NOT_RUN]
    with sqlite3.connect(artifacts_dir / "experiment_log.db") as conn:
        _ensure_table(conn)
        for e in ran:
            conn.execute(
                f"INSERT OR REPLACE INTO {TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    e["id"],
                    e.get("num"),
                    e.get("title") or e["id"],
                    e.get("status"),
                    e.get("raw_status"),
                    e.get("metric"),
                    metric_name,
                    direction or e.get("metric_direction") or "",
                    baseline,
                    json.dumps(e.get("parents") or []),
                    e.get("reason") or "",
                    e.get("stage") or "",
                    e.get("started_at"),
                    e.get("completed_at"),
                    e.get("patch") or "",
                    (e.get("proposal") or {}).get("description") or "",
                    now,
                ),
            )
    ideas = _import_ideas(artifacts_dir, repo.get("hypotheses", []))
    result = {
        "repo_dir": str(repo_dir),
        "merged_at": now,
        "repo_mtime": mtime,
        "experiments": len(ran),
        "not_run": len(repo.get("experiments", [])) - len(ran),
        "ideas_added": ideas,
        "metric_name": metric_name,
        "baseline": baseline,
    }
    atomic_write_json(artifacts_dir / MERGE_FILE, result)
    return result


def _import_ideas(artifacts_dir: Path, hypotheses: list[dict[str, Any]]) -> int:
    """Untested repo hypotheses join the pool, so the one engine can try them."""
    from researchforge.kaggle.hypothesis_pool import POOL_FILE

    pool_path = artifacts_dir / POOL_FILE
    raw = read_json(pool_path, [])
    pool: list[dict[str, Any]] = (
        [h for h in raw if isinstance(h, dict)] if isinstance(raw, list) else []
    )
    known_ids = {str(h.get("repo_hypothesis_id")) for h in pool if h.get("repo_hypothesis_id")}
    known_text = {str(h.get("idea") or "").strip().lower() for h in pool}
    added = 0
    for h in hypotheses:
        if h.get("origin") != "imported" or h.get("tested") or h.get("status") == "rejected":
            continue  # drafts were never imported; tested and rejected ones are done
        title = str(h.get("title") or "").strip()
        if not title or h.get("id") in known_ids or title.lower() in known_text:
            continue
        pool.append(
            {
                "idea": title,
                "rationale": h.get("detail") or "",
                "source": "repo project",
                "repo_hypothesis_id": h.get("id"),
                "tested": False,
                "injected_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        known_text.add(title.lower())
        added += 1
    if added:
        atomic_write_json(pool_path, pool)
    return added


def imported_experiments(artifacts_dir: Path) -> list[dict[str, Any]]:
    """The imported rows, oldest first ([] before any import)."""
    db = artifacts_dir / "experiment_log.db"
    if not db.is_file():
        return []
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(f"SELECT * FROM {TABLE} ORDER BY num")]
    except sqlite3.Error:  # no table: nothing imported yet
        return []
    for r in rows:
        try:
            r["parents"] = [str(x) for x in json.loads(r.get("parents") or "[]")]
        except ValueError:
            r["parents"] = []
    return rows


def merge_status(artifacts_dir: Path) -> dict[str, Any] | None:
    """The last import, or None when there is no repo project beside this loop."""
    repo_dir = repo_dir_for(artifacts_dir)
    if repo_dir is None:
        return None
    merge = read_json(artifacts_dir / MERGE_FILE, {}) or {}
    merge = merge if isinstance(merge, dict) else {}
    return {**merge, "pending": merge.get("repo_mtime") != _repo_mtime(repo_dir)}


def earlier_work(artifacts_dir: Path, limit: int = 30) -> list[dict[str, Any]]:
    """What the repo engine tried, for prepare-round: so ideas are not repeated blindly."""
    out = []
    for r in imported_experiments(artifacts_dir)[-limit:]:
        out.append(
            {
                "id": r["repo_id"],
                "title": r.get("title"),
                "outcome": r.get("status"),
                "metric": r.get("metric"),
                "note": r.get("reason") or "",
            }
        )
    return out
