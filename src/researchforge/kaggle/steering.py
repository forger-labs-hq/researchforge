"""Directions from the person: steer the research from the dashboard or a chat.

A direction is either **standing** ("never use external data"), which every proposal
is told until the person removes it, or **for the next round(s)** ("try a ViT backbone
next"), which stays until as many rounds as the person asked for (one by default) have
followed it. Both reach whoever proposes — the chat agent through ``prepare-round``'s
``directions`` and ``up_next``, an auto-run launch and the API-key loop through their
prompts — ahead of the idea ranking.

**Up next** is the person's queue: their next-round directions, newest first unless
they reorder them, then the ideas they pinned (``hypothesis_pool.pinned``), then the
engine's ranking. Its head must be followed: a proposal names it in ``follows`` or
gives the reason it cannot in ``declines`` (``check_proposal``), and execute-round
refuses one that does neither. Where two directions disagree, the one higher in the
queue wins.

The proposer answers each new direction once (``kaggle steer <slug> --reply <id> "…"``)
with what it will do; the reply shows under the direction on the dashboard. A
direction can never approve a submission or a plan: those keep their own buttons.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from researchforge.kaggle.workspace import atomic_write_json, read_json

FILE = "steering.json"
STEERED_FILE = "steered_rounds.json"  # {round: [Up next items it followed or declined]}
Kind = Literal["standing", "next"]
KINDS: tuple[Kind, ...] = ("standing", "next")
_MAX_TEXT = 2000


@dataclass
class Direction:
    id: str
    text: str
    kind: Kind
    at: str
    via: str = "dashboard"  # dashboard | chat | cli
    status: str = "open"  # open | read | applied | removed
    read_by_round: int | None = None
    read_at: str = ""
    applied_at: str = ""
    removed_at: str = ""
    replies: list[dict[str, str]] = field(default_factory=list)
    rounds_wanted: int = 1  # next-round directions: how many rounds must follow it
    followed_by: list[int] = field(default_factory=list)  # rounds that followed it
    declined: list[dict[str, Any]] = field(default_factory=list)  # {round, reason, at}
    # A fix for a submission Kaggle failed (kaggle.fixes): {round, error, attempt, ...}
    fixes: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.status in ("open", "read") or (
            self.kind == "standing" and self.status != "removed"
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load(artifacts_dir: Path) -> list[Direction]:
    raw = read_json(artifacts_dir / FILE, []) or []
    known = set(Direction.__dataclass_fields__)
    out = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("id") and item.get("text"):
            out.append(Direction(**{k: v for k, v in item.items() if k in known}))
    return out


def _save(artifacts_dir: Path, items: list[Direction]) -> None:
    atomic_write_json(artifacts_dir / FILE, [asdict(d) for d in items])


def _find(items: list[Direction], direction_id: str) -> Direction:
    hit = next((d for d in items if d.id == direction_id), None)
    if hit is None:
        raise KeyError(f"no direction {direction_id}")
    return hit


def add(
    artifacts_dir: Path,
    text: str,
    kind: str = "next",
    via: str = "dashboard",
    rounds: int = 1,
    fixes: dict[str, Any] | None = None,
) -> Direction:
    """A new direction goes to the top of the queue: the newest one is followed first."""
    text = text.strip()
    if not text:
        raise ValueError("write the direction first")
    if len(text) > _MAX_TEXT:
        raise ValueError(f"keep a direction under {_MAX_TEXT} characters")
    if kind not in KINDS:
        raise ValueError("kind must be standing or next")
    if not 1 <= rounds <= 50:
        raise ValueError("a direction can be for 1 to 50 rounds")
    items = load(artifacts_dir)
    d = Direction(
        id=f"d-{secrets.token_hex(3)}",
        text=text,
        kind=kind,
        at=_now(),
        via=via,
        rounds_wanted=rounds if kind == "next" else 1,
        fixes=dict(fixes or {}),
    )
    items.insert(0, d)
    _save(artifacts_dir, items)
    return d


def reorder(artifacts_dir: Path, ids: list[str]) -> list[Direction]:
    """Put these directions first, in this order; the rest keep theirs after them."""
    items = load(artifacts_dir)
    first = [_find(items, i) for i in dict.fromkeys(ids)]
    _save(artifacts_dir, first + [d for d in items if d not in first])
    return load(artifacts_dir)


def reply(artifacts_dir: Path, direction_id: str, text: str, by: str = "agent") -> Direction:
    """Record the proposer's (or the person's) answer to a direction."""
    from researchforge.kaggle import control

    text = text.strip()
    if not text:
        raise ValueError("the reply is empty")
    items = load(artifacts_dir)
    d = _find(items, direction_id)
    d.replies.append({"at": _now(), "by": by, "text": text[:_MAX_TEXT]})
    _save(artifacts_dir, items)
    if by != "person":
        control.notify(artifacts_dir, "interaction", f'On your direction "{d.text[:80]}": {text}')
    return d


def remove(artifacts_dir: Path, direction_id: str) -> Direction:
    items = load(artifacts_dir)
    d = _find(items, direction_id)
    d.status, d.removed_at = "removed", _now()
    _save(artifacts_dir, items)
    return d


def mark_round(artifacts_dir: Path, round_num: int) -> list[Direction]:
    """prepare-round for ``round_num``: mark new directions read. Returns the active ones.

    Reading is not following: a next-round direction stays until rounds follow it
    (``record_round``).
    """
    items = load(artifacts_dir)
    changed = False
    for d in items:
        if d.kind == "next" and d.status == "open":
            d.status, d.read_by_round, d.read_at, changed = "read", round_num, _now(), True
    if changed:
        _save(artifacts_dir, items)
    return [d for d in items if d.active]


def up_next(artifacts_dir: Path) -> list[dict[str, Any]]:
    """The person's queue in order: next-round directions, then pinned ideas."""
    from researchforge.kaggle.hypothesis_pool import pinned

    queue: list[dict[str, Any]] = [
        {
            "id": d.id,
            "kind": "fix" if d.fixes else "direction",
            "text": d.text,
            "rounds_left": max(1, d.rounds_wanted - len(d.followed_by)),
            **({"fixes_round": d.fixes.get("round")} if d.fixes else {}),
        }
        for d in load(artifacts_dir)
        if d.kind == "next" and d.active
    ]
    queue += [
        {"id": h["pin_id"], "kind": "pinned_idea", "text": str(h["idea"])}
        for h in pinned(artifacts_dir)
    ]
    for place, item in enumerate(queue, 1):
        item["place"] = place
    return queue


def _named(proposal: dict[str, Any], key: str) -> set[str]:
    raw = proposal.get(key)
    if isinstance(raw, dict):
        return {str(k) for k, v in raw.items() if str(v or "").strip()}
    if isinstance(raw, str):
        return {raw} if raw.strip() else set()
    return {str(x) for x in raw} if isinstance(raw, list) else set()


def follows(proposal: dict[str, Any], item: dict[str, Any]) -> bool:
    """Whether a proposal follows a queue item: names it, or (an idea) tests it."""
    from researchforge.kaggle.hypothesis_pool import idea_key

    if item["id"] in _named(proposal, "follows"):
        return True
    tested = str(proposal.get("hypothesis") or "")
    return (
        item["kind"] == "pinned_idea"
        and bool(tested)
        and (idea_key(tested) == idea_key(item["text"]))
    )


def check_proposal(artifacts_dir: Path, proposal: dict[str, Any]) -> str:
    """Why a proposal may not run (it ignores the head of Up next), or "" when it may."""
    queue = up_next(artifacts_dir)
    if not queue:
        return ""
    head = queue[0]
    if follows(proposal, head) or head["id"] in _named(proposal, "declines"):
        return ""
    what = {"direction": "direction", "fix": "fix item"}.get(head["kind"], "pinned idea")
    return (
        f"the proposal ignores the person's top {what} in Up next, {head['id']}: "
        f'"{head["text"][:160]}". Follow it ("follows": ["{head["id"]}"]'
        + (', and "hypothesis" set to that idea' if head["kind"] == "pinned_idea" else "")
        + f') or say why this round cannot ("declines": {{"{head["id"]}": "<reason>"}})'
    )


def record_round(artifacts_dir: Path, round_num: int, proposal: dict[str, Any]) -> list[str]:
    """A round started: count it for the directions it follows, keep the ones it declines.

    A next-round direction is applied once ``rounds_wanted`` rounds followed it. Returns
    the ids it followed.
    """
    queue = {item["id"]: item for item in up_next(artifacts_dir)}
    raw = proposal.get("declines")
    declines: dict[str, Any] = raw if isinstance(raw, dict) else {}
    _remember_round(artifacts_dir, round_num, proposal, list(queue.values()), declines)
    items = load(artifacts_dir)
    done = []
    for d in items:
        if not d.active or d.kind != "next" or d.id not in queue:
            continue
        if follows(proposal, queue[d.id]):
            if round_num not in d.followed_by:
                d.followed_by.append(round_num)
            if len(d.followed_by) >= d.rounds_wanted:
                d.status, d.applied_at = "applied", _now()
            done.append(d.id)
        elif str(declines.get(d.id) or "").strip():
            reason = str(declines[d.id]).strip()[:_MAX_TEXT]
            d.declined.append({"round": round_num, "reason": reason, "at": _now()})
    _save(artifacts_dir, items)
    return done


def active(artifacts_dir: Path) -> list[Direction]:
    return [d for d in load(artifacts_dir) if d.active]


def _remember_round(
    artifacts_dir: Path,
    round_num: int,
    proposal: dict[str, Any],
    queue: list[dict[str, Any]],
    declines: dict[str, Any],
) -> None:
    """Keep what the round did with each queue item, so the dashboard can mark it."""
    got = []
    for item in queue:
        entry = {"id": item["id"], "kind": item["kind"], "text": item["text"][:300]}
        if follows(proposal, item):
            got.append({**entry, "how": "followed"})
        elif str(declines.get(item["id"]) or "").strip():
            reason = str(declines[item["id"]]).strip()[:_MAX_TEXT]
            got.append({**entry, "how": "declined", "reason": reason})
    data = read_json(artifacts_dir / STEERED_FILE, {}) or {}
    data = data if isinstance(data, dict) else {}
    if got:
        data[str(round_num)] = got
    else:
        data.pop(str(round_num), None)
    atomic_write_json(artifacts_dir / STEERED_FILE, data)


def steered_rounds(artifacts_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Round -> the directions and pinned ideas it followed (or declined), oldest first.

    Rounds from before this was kept are taken from the directions themselves: the
    rounds that followed one, or (older still) the round that read it.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    data = read_json(artifacts_dir / STEERED_FILE, {}) or {}
    for key, got in data.items() if isinstance(data, dict) else []:
        if str(key).isdigit() and isinstance(got, list):
            out[int(key)] = [g for g in got if isinstance(g, dict)]
    for d in load(artifacts_dir):
        if d.kind != "next":
            continue
        rounds = list(d.followed_by) or (
            [d.read_by_round] if d.status == "applied" and d.read_by_round else []
        )
        kind = "fix" if d.fixes else "direction"
        entry = {"id": d.id, "kind": kind, "text": d.text[:300], "how": "followed"}
        for r in rounds:
            if not any(g.get("id") == d.id for g in out.get(r, [])):
                out.setdefault(r, []).append(entry)
    return out


def prompt_lines(artifacts_dir: Path) -> list[str]:
    """The standing directions, then Up next in order, as lines for a proposer's prompt."""
    lines = [
        f"[{d.id}] Standing direction: {d.text}"
        for d in active(artifacts_dir)
        if d.kind == "standing"
    ]
    for item in up_next(artifacts_dir):
        what = {"direction": "Direction", "fix": "Fix a failed submission"}.get(
            item["kind"], "Pinned idea"
        )
        head = (
            " (head of the queue: follow it or decline it with a reason)"
            if item["place"] == 1
            else ""
        )
        lines.append(f"Up next #{item['place']} [{item['id']}] {what}{head}: {item['text']}")
    return lines


FOLLOW_RULE = (
    "Up next is the person's queue, in their order. Its head comes first: put its id in "
    'the proposal\'s "follows" (for a pinned idea, also set "hypothesis" to that idea), or '
    'name it in "declines" with the reason this round cannot ({"<id>": "<reason>"}); '
    "execute-round refuses a proposal that does neither. Where two directions disagree, "
    "the one higher in the queue wins."
)


def for_round(items: list[Direction]) -> list[dict[str, Any]]:
    """prepare-round's ``directions``: what to follow and which still want a reply."""
    return [
        {
            "id": d.id,
            "kind": d.kind,
            "text": d.text,
            "reply_needed": not any(r.get("by") != "person" for r in d.replies),
            "rounds_left": max(1, d.rounds_wanted - len(d.followed_by))
            if d.kind == "next"
            else None,
        }
        for d in items
    ]
