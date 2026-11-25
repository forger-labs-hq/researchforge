"""The pending hypothesis pool: one ranking for every loop that draws from it.

``pending_hypotheses.json`` is filled by the kernel watcher, the arXiv sensor,
resynthesize and the user. Three loops read it — Claude's ``prepare-round``, the
API-key loop (``kaggle run``) and repo autorun's re-synthesis — and they must
agree on what to try next, so the order is decided here and nowhere else.

Order, most important first:

0. ideas the person pinned, in the order they set (``pinned_ideas.json``; the
   dashboard's Up next queue). Kept apart from the pool, which the watcher rewrites;
1. ideas the user added by hand;
2. notebooks claiming a public-LB score above our best submission, highest claim
   first. The claim is the author's word, but it is a leaderboard score, which a
   local benchmark often fails to predict (Enveda: local 0.915, LB 0.21, public
   notebooks 0.49+), so a notebook well ahead of us is the next thing to try;
3. how measurable the change is locally (``blind`` ideas need a Kaggle GPU run);
4. the watcher's relevance tag — extend what works, re-tread failures last.

A notebook claiming a public-LB score *below* our best submission is left out
(:func:`below_our_lb`): running it can at best reproduce a worse score. The person
can still pin it, and ideas they added themselves are never left out.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import location_role, read_json

POOL_FILE = "pending_hypotheses.json"
PINS_FILE = "pinned_ideas.json"  # idea keys the person pinned, first = tried first
_RELEVANCE_RANK = {"builds_on_best": 0, "new_direction": 1, "repeats_failure": 2}
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2, "blind": 3}


def load_pool(artifacts_dir: Path) -> list[dict[str, Any]]:
    pool = read_json(artifacts_dir / POOL_FILE, [])
    return [h for h in pool if isinstance(h, dict)] if isinstance(pool, list) else []


def idea_key(idea: str) -> str:
    """How an idea is matched everywhere (mark_tested, skip_idea, pins)."""
    return idea.strip().lower()[:60]


def pin_id(idea: str) -> str:
    """A short stable id for a pinned idea, for proposals to name it by."""
    import hashlib

    return "p-" + hashlib.sha1(idea_key(idea).encode()).hexdigest()[:6]


def pinned(artifacts_dir: Path) -> list[dict[str, Any]]:
    """The pinned ideas still waiting to be tried, in the person's order.

    Each is the pool entry plus ``pin_id`` and ``pin`` (its 1-based place). A pinned
    idea leaves the queue once it is tested or skipped.
    """
    keys = read_json(artifacts_dir / PINS_FILE, []) or []
    waiting = {
        idea_key(str(h.get("idea") or "")): h
        for h in load_pool(artifacts_dir)
        if h.get("idea") and not h.get("tested") and not h.get("skipped")
    }
    out: list[dict[str, Any]] = []
    for key in keys if isinstance(keys, list) else []:
        h = waiting.get(str(key))
        if h is not None:
            out.append({**h, "pin_id": pin_id(str(h["idea"])), "pin": len(out) + 1})
    return out


def _waiting_key(artifacts_dir: Path, name: str) -> str:
    """The key of the untested pool idea named by its text or ``pin_id``; ValueError if none."""
    name = name.strip()
    for h in load_pool(artifacts_dir):
        idea = str(h.get("idea") or "")
        if not idea or h.get("tested") or h.get("skipped"):
            continue
        if name == pin_id(idea) or idea_key(name) == idea_key(idea):
            return idea_key(idea)
    raise ValueError(f"no untested idea in the pool matches {name[:80]!r}")


def set_pins(artifacts_dir: Path, ideas: list[str]) -> list[dict[str, Any]]:
    """Pin exactly these ideas, in this order (an empty list unpins all). Returns pinned().

    An idea is named by its text or its ``pin_id``; one that is not waiting in the pool
    is refused, so a typo never silently pins nothing.
    """
    from researchforge.kaggle.workspace import atomic_write_json

    keys: list[str] = []
    for name in ideas:
        key = _waiting_key(artifacts_dir, name)
        if key not in keys:
            keys.append(key)
    atomic_write_json(artifacts_dir / PINS_FILE, keys)
    return pinned(artifacts_dir)


def pin(artifacts_dir: Path, idea: str, place: int | None = None) -> list[dict[str, Any]]:
    """Pin one idea at ``place`` (1-based; last when None), or move a pinned one there."""
    key = _waiting_key(artifacts_dir, idea)
    rest = [h["pin_id"] for h in pinned(artifacts_dir) if idea_key(h["idea"]) != key]
    at = len(rest) if place is None else max(0, min(len(rest), place - 1))
    return set_pins(artifacts_dir, [*rest[:at], key, *rest[at:]])


def unpin(artifacts_dir: Path, idea: str) -> list[dict[str, Any]]:
    key = _waiting_key(artifacts_dir, idea)
    return set_pins(
        artifacts_dir, [h["pin_id"] for h in pinned(artifacts_dir) if idea_key(h["idea"]) != key]
    )


def our_best_public_lb(artifacts_dir: Path, direction: str) -> float | None:
    """Our best scored public-LB submission from the ``submissions.json`` snapshot."""
    snap = read_json(artifacts_dir / "submissions.json", {}) or {}
    scores: list[float] = []
    for s in snap.get("submissions", []) if isinstance(snap, dict) else []:
        raw = s.get("publicscore", s.get("publicScore")) if isinstance(s, dict) else None
        try:
            scores.append(float(str(raw)))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    return max(scores) if str(direction).lower().startswith("max") else min(scores)


def claim_gap(h: dict[str, Any], our_best: float | None, direction: str) -> float | None:
    """How far a notebook's claimed LB score is ahead of ours (positive = ahead)."""
    claimed = h.get("claimed_score")
    if not isinstance(claimed, (int, float)) or our_best is None:
        return None
    gap = claimed - our_best if str(direction).lower().startswith("max") else our_best - claimed
    return round(gap, 6)


def _below(h: dict[str, Any], gap: float | None) -> bool:
    return gap is not None and gap < 0 and h.get("source") != "user"


def below_our_lb(artifacts_dir: Path, direction: str) -> set[str]:
    """Untested ideas left out of the ranking: notebooks claiming less than our best LB."""
    our_best = our_best_public_lb(artifacts_dir, direction)
    if our_best is None:
        return set()
    return {
        str(h.get("idea"))
        for h in load_pool(artifacts_dir)
        if not h.get("tested") and _below(h, claim_gap(h, our_best, direction))
    }


def rank_untested(
    artifacts_dir: Path, direction: str, *, blocked: set[str] | None = None
) -> list[dict[str, Any]]:
    """Untested, unblocked pool entries in trial order, each tagged for the reader.

    Adds ``metric_confidence``, ``confidence_note`` and, for notebook ideas with a
    claim, ``claim_vs_our_best`` and ``our_best_lb``.
    """
    from researchforge.intelligence.scanner import confidence_note, tag_metric_confidence

    if blocked is None:
        from researchforge.kaggle.control import blocked_ideas

        blocked = blocked_ideas(artifacts_dir)
    our_best = our_best_public_lb(artifacts_dir, direction)
    pins = {idea_key(h["idea"]): h for h in pinned(artifacts_dir)}
    ranked: list[tuple[tuple[int, float, int, int, int], dict[str, Any]]] = []
    for index, h in enumerate(load_pool(artifacts_dir)):
        idea = str(h.get("idea") or "")
        if h.get("tested") or h.get("skipped") or not idea or idea in blocked:
            continue
        confidence = str(tag_metric_confidence(idea, []))
        entry = {
            **h,
            "metric_confidence": confidence,
            "confidence_note": confidence_note(confidence),
        }
        gap = claim_gap(h, our_best, direction)
        if gap is not None:
            entry["claim_vs_our_best"] = gap
            entry["our_best_lb"] = our_best
        if _below(h, gap) and idea_key(idea) not in pins:
            continue
        tier, within = (1 if h.get("source") == "user" else 2), 0.0
        if gap is not None and gap > 0:
            within = -gap
        pinned_as = pins.get(idea_key(idea))
        if pinned_as is not None:
            entry["pin"], entry["pin_id"] = pinned_as["pin"], pinned_as["pin_id"]
            tier, within = 0, float(pinned_as["pin"])
        key = (
            tier,
            within,
            _CONFIDENCE_RANK.get(confidence, 1),
            _RELEVANCE_RANK.get(str(h.get("relevance") or ""), 1),
            index,
        )
        ranked.append((key, entry))
    ranked.sort(key=lambda pair: pair[0])
    return [entry for _, entry in ranked]


def describe(entry: dict[str, Any]) -> str:
    """One line for a prompt: the idea plus where it came from and what it claims."""
    parts = [str(entry.get("idea") or "")]
    tags = [str(entry.get("source") or "")]
    if entry.get("kernel_ref"):
        tags.append(f"notebook {entry['kernel_ref']}")
    if isinstance(entry.get("claimed_score"), (int, float)):
        claim = f"claims public LB {entry['claimed_score']}"
        if entry.get("claim_vs_our_best") is not None:
            claim += f" ({entry['claim_vs_our_best']:+.4f} vs our best {entry['our_best_lb']})"
        tags.append(claim + " — the author's claim, unverified")
    if entry.get("metric_confidence"):
        tags.append(f"local metric confidence: {entry['metric_confidence']}")
    return parts[0] + "  [" + "; ".join(t for t in tags if t) + "]"


def overnight_dirs_for(conn: sqlite3.Connection) -> list[Path]:
    """The Kaggle overnight dirs that share a root with this repo project's DB."""
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main" and row[2]:
            root = Path(row[2]).resolve().parent.parent  # <root>/.researchforge/<db>
            return sorted(
                p
                for p in root.glob(".overnight_*")
                if (p / POOL_FILE).is_file() and location_role(p) == "primary"
            )
    return []


def mark_tested(
    artifacts_dir: Path, proposal_data: dict[str, Any], round_num: int, outcome: str
) -> str | None:
    """Mark the pool idea a proposal tested. Returns the matched idea, if any.

    Matches the proposal's ``hypothesis`` field (or, failing that, its
    source_credit / title) against pool ideas on a case-insensitive 60-char
    prefix — exact intent, no fuzzy guessing.
    """
    from datetime import datetime

    from researchforge.kaggle.workspace import atomic_write_json

    pool_path = artifacts_dir / POOL_FILE
    pool = read_json(pool_path, None)
    if not isinstance(pool, list):
        return None
    keys = [
        str(proposal_data.get(k) or "").strip().lower()[:60]
        for k in ("hypothesis", "source_credit", "title")
    ]
    keys = [k for k in keys if k]
    for h in pool:
        idea = str(h.get("idea", "")).strip().lower()[:60]
        if idea and idea in keys and not h.get("tested"):
            h["tested"] = True
            h["tested_in_round"] = round_num
            h["tested_at"] = datetime.now().isoformat(timespec="seconds")
            h["tested_outcome"] = outcome
            atomic_write_json(pool_path, pool)
            return str(h.get("idea") or "") or None
    return None


def set_round_outcome(artifacts_dir: Path, round_num: int, outcome: str) -> str | None:
    """Correct the outcome of the idea a round tested (a late Kaggle result). Returns it."""
    from researchforge.kaggle.workspace import atomic_write_json

    pool_path = artifacts_dir / POOL_FILE
    pool = read_json(pool_path, None)
    if not isinstance(pool, list):
        return None
    for h in pool:
        if h.get("tested_in_round") == round_num:
            h["tested_outcome"] = outcome
            atomic_write_json(pool_path, pool)
            return str(h.get("idea") or "") or None
    return None


def skip_idea(artifacts_dir: Path, idea: str, reason: str) -> str | None:
    """Take one untested idea out of the ranking, with the reason. Returns the idea.

    For ideas the loop cannot run on its compute (they would need the person to
    do something on their computer), so later launches stop meeting them again.
    """
    from datetime import datetime

    from researchforge.kaggle.workspace import atomic_write_json

    pool_path = artifacts_dir / POOL_FILE
    pool = read_json(pool_path, None)
    if not isinstance(pool, list):
        return None
    key = idea.strip().lower()[:60]
    for h in pool:
        if not isinstance(h, dict) or h.get("tested"):
            continue
        if str(h.get("idea", "")).strip().lower()[:60] == key and key:
            h["skipped"] = True
            h["skip_reason"] = reason
            h["skipped_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_write_json(pool_path, pool)
            return str(h.get("idea") or "")
    return None


def setup_direction(artifacts_dir: Path) -> str:
    """The direction the loop judges by: the setup contract's primary metric."""
    setup = read_json(artifacts_dir / "overnight_setup.json", {}) or {}
    try:
        return str(setup["contract"]["objective"]["primary_metric"]["direction"])
    except (KeyError, TypeError):
        return "maximize"


def pool_summary(conn: sqlite3.Connection, limit: int = 5) -> list[dict[str, Any]]:
    """Untested pool ideas next to a repo project, per overnight dir, for status output."""
    out = []
    for d in overnight_dirs_for(conn):
        ranked = rank_untested(d, setup_direction(d))
        out.append(
            {
                "overnight_dir": str(d),
                "untested": len(ranked),
                "top": [describe(h) for h in ranked[:limit]],
            }
        )
    return out


def leads_instructions(conn: sqlite3.Connection, limit: int = 15) -> list[str]:
    """Synthesis prompt lines carrying the Kaggle pool next to this repo project.

    The watcher, arXiv sensor and user fill that pool for the overnight loop; a
    repo project on the same competition should draw on it too. The lines say
    plainly that these are leads: hypotheses still need evidence from the stored
    papers, and a notebook's claimed score is unverified.
    """
    lines: list[str] = []
    for d in overnight_dirs_for(conn):
        ranked = rank_untested(d, setup_direction(d))[:limit]
        if ranked:
            lines.extend(f"- {describe(h)}" for h in ranked)
    if not lines:
        return []
    return [
        "KAGGLE LEADS — untested ideas the Kaggle watcher, arXiv sensor and user "
        "collected for this competition, best first. They are leads, not evidence: "
        "a hypothesis drawn from one still needs evidence from the stored papers, "
        "and a notebook's claimed leaderboard score is its author's word.",
        *lines,
        "Prefer hypotheses that test a strong lead, unless the results above already rule it out.",
    ]
