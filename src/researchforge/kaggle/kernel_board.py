"""The public notebooks of a competition, ranked by the leaderboard score they give.

``kernel_board.json`` is rebuilt by the kernel watcher every cycle (hourly) and read
by the dashboard and ``prepare-round``. Each notebook carries:

* ``kaggle_rank`` — its place in Kaggle's own "best score" ordering (the API gives
  the order but not the number), None when it is not in the top of that list;
* ``claimed_score`` — the LB score its author states, from the title or the notebook
  (kernel_claims), re-read whenever the notebook is re-run;
* ``last_run_time`` / ``updated_at`` — when it last ran, and when the watcher saw
  a newer run of a notebook it already knew;
* ``tried`` — our rounds that ran or adapted it, with their local and LB scores.

Rows are sorted by claimed score (best first, in the metric's direction), then by
Kaggle's order. ``our_best_lb`` and each row's ``gap`` say how far ahead it claims to be.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, read_json

BOARD_FILE = "kernel_board.json"
MAX_ROWS = 200
MAX_CLAIM_FETCHES = 25  # notebooks downloaded per cycle to read a claim the title lacks
_ROUND_IN_DESCRIPTION = re.compile(r"\bR(\d+)\b")


def load_board(artifacts_dir: Path) -> dict[str, Any]:
    board = read_json(artifacts_dir / BOARD_FILE, {}) or {}
    return board if isinstance(board, dict) else {}


def _rounds_lb(artifacts_dir: Path) -> dict[int, float]:
    """Public LB score of each submitted round ("RF R21: …" in the description)."""
    snap = read_json(artifacts_dir / "submissions.json", {}) or {}
    out: dict[int, float] = {}
    for s in snap.get("submissions", []) if isinstance(snap, dict) else []:
        if not isinstance(s, dict):
            continue
        m = _ROUND_IN_DESCRIPTION.search(str(s.get("description") or ""))
        try:
            score = float(str(s.get("publicscore", s.get("publicScore"))))
        except (TypeError, ValueError):
            continue
        if m:
            out.setdefault(int(m.group(1)), score)
    return out


def _tried(artifacts_dir: Path, refs: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Our rounds that ran or adapted each notebook (credited, or its ref in the code)."""
    db = artifacts_dir / "experiment_log.db"
    if not db.exists() or not refs:
        return {}
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT round_num, source_credit, patch_code, full_passed, full_metric, "
                "rejected_reason FROM experiments ORDER BY round_num"
            ).fetchall()
    except sqlite3.Error:
        return {}
    lb = _rounds_lb(artifacts_dir)
    out: dict[str, list[dict[str, Any]]] = {}
    for num, credit, code, passed, metric, reason in rows:
        for ref in refs:
            if ref == (credit or "").strip() or ref in (code or ""):
                out.setdefault(ref, []).append(
                    {
                        "round": num,
                        "local": metric if passed else None,
                        "lb": lb.get(num),
                        "note": "" if passed else str(reason or "failed")[:120],
                    }
                )
    return out


def update_board(
    artifacts_dir: Path,
    recent: list[Any],
    by_score: list[Any],
    *,
    direction: str,
    our_best_lb: float | None,
    updated_refs: set[str] | None = None,
    max_fetches: int = MAX_CLAIM_FETCHES,
) -> dict[str, Any]:
    """Merge this cycle's listings into the board and save it. Returns the board.

    ``by_score`` is Kaggle's best-score-first listing (its order is the rank);
    ``updated_refs`` are notebooks the watcher saw re-run since it last looked.
    """
    from researchforge.kaggle.kernel_claims import notebook_claim, title_claim

    now = datetime.now().isoformat(timespec="seconds")
    old = {r["ref"]: r for r in load_board(artifacts_dir).get("rows", []) if r.get("ref")}
    rank = {k.ref: i + 1 for i, k in enumerate(by_score)}
    listed = {k.ref: k for k in [*recent, *by_score]}
    rows: dict[str, dict[str, Any]] = {}
    for ref, row in old.items():  # remembered notebooks keep their row, without a rank
        rows[ref] = {**row, "kaggle_rank": None}
    for ref, k in listed.items():
        prev = old.get(ref, {})
        run = str(k.last_run_time or "")
        row = {
            **prev,
            "ref": ref,
            "title": k.title,
            "author": k.author,
            "votes": k.total_votes,
            "last_run_time": run,
            "kaggle_rank": rank.get(ref),
            "first_seen_at": prev.get("first_seen_at") or now,
        }
        if updated_refs and ref in updated_refs:
            row["updated_at"] = now
        if prev.get("claim_checked_run") != run:
            claim = title_claim(k.title)
            if claim is None and max_fetches > 0:
                max_fetches -= 1
                claim = notebook_claim(ref)
            row.update(
                claim or {"claimed_score": None, "claim_source": None, "claim_snippet": None}
            )
            row["claim_checked_run"] = run
        rows[ref] = row

    maximize = str(direction).lower().startswith("max")
    tried = _tried(artifacts_dir, set(rows))
    for ref, row in rows.items():
        row["tried"] = tried.get(ref, [])
        claim = row.get("claimed_score")
        row["gap"] = (
            round((claim - our_best_lb) if maximize else (our_best_lb - claim), 6)
            if isinstance(claim, (int, float)) and our_best_lb is not None
            else None
        )

    def order(row: dict[str, Any]) -> tuple[int, float, int]:
        claim = row.get("claimed_score")
        rank = int(row.get("kaggle_rank") or 10**6)
        if not isinstance(claim, (int, float)):
            return (1, 0.0, rank)
        return (0, -float(claim) if maximize else float(claim), rank)

    ranked = sorted(rows.values(), key=order)[:MAX_ROWS]
    board = {
        "fetched_at": now,
        "direction": "maximize" if maximize else "minimize",
        "our_best_lb": our_best_lb,
        "rows": ranked,
    }
    atomic_write_json(artifacts_dir / BOARD_FILE, board)
    return board


def ahead_of_us(artifacts_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    """Notebooks claiming a better LB than our best, best first (for the proposer)."""
    board = load_board(artifacts_dir)
    out = []
    for r in board.get("rows", []):
        if isinstance(r.get("gap"), (int, float)) and r["gap"] > 0:
            out.append(
                {
                    k: r.get(k)
                    for k in (
                        "ref",
                        "title",
                        "claimed_score",
                        "claim_source",
                        "gap",
                        "kaggle_rank",
                        "last_run_time",
                        "updated_at",
                        "tried",
                    )
                }
            )
        if len(out) >= limit:
            break
    return out
