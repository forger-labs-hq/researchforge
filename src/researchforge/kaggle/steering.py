"""Directions from the person: steer the research from the dashboard or a chat.

A direction is either **standing** ("never use external data"), which every proposal
is told until the person removes it, or **for the next round** ("try a ViT backbone
next"): the next ``prepare-round`` marks it read by that round, and the round after
closes it as applied. Both reach whoever proposes — the chat agent through
``prepare-round``'s ``person_notes`` and ``directions``, an auto-run launch through its
prompt — ahead of the idea ranking.

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


def add(artifacts_dir: Path, text: str, kind: str = "next", via: str = "dashboard") -> Direction:
    text = text.strip()
    if not text:
        raise ValueError("write the direction first")
    if len(text) > _MAX_TEXT:
        raise ValueError(f"keep a direction under {_MAX_TEXT} characters")
    if kind not in KINDS:
        raise ValueError("kind must be standing or next")
    items = load(artifacts_dir)
    d = Direction(id=f"d-{secrets.token_hex(3)}", text=text, kind=kind, at=_now(), via=via)
    items.append(d)
    _save(artifacts_dir, items)
    return d


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
    """prepare-round for ``round_num``: close what an earlier round read, mark the rest read.

    Returns the directions this round must follow.
    """
    items = load(artifacts_dir)
    changed = False
    for d in items:
        if d.kind != "next":
            continue
        if d.status == "read" and (d.read_by_round or 0) < round_num:
            d.status, d.applied_at, changed = "applied", _now(), True
        elif d.status == "open":
            d.status, d.read_by_round, d.read_at, changed = "read", round_num, _now(), True
    if changed:
        _save(artifacts_dir, items)
    return [d for d in items if d.active]


def active(artifacts_dir: Path) -> list[Direction]:
    return [d for d in load(artifacts_dir) if d.active]


def prompt_lines(artifacts_dir: Path) -> list[str]:
    """The active directions as lines for a proposer's prompt."""
    return [
        f"[{d.id}] {'Standing direction' if d.kind == 'standing' else 'For this round'}: {d.text}"
        for d in active(artifacts_dir)
    ]


def for_round(items: list[Direction]) -> list[dict[str, Any]]:
    """prepare-round's ``directions``: what to follow and which still want a reply."""
    return [
        {
            "id": d.id,
            "kind": d.kind,
            "text": d.text,
            "reply_needed": not any(r.get("by") != "person" for r in d.replies),
        }
        for d in items
    ]
