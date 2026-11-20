"""A submission Kaggle failed becomes a fix round at the top of Up next.

Kaggle can refuse a round long after it ran: a notebook that runs out of time on the
hidden test set (which can be larger than the public one), runs out of memory, or
writes a file in the wrong shape. The monitor reads Kaggle's reason with each
submissions check (``submission.leaderboard_by_round``) and calls :func:`queue_fixes`:

- the newest failed round for each distinct error gets one fix item at the top of Up
  next (older rounds that failed the same way are named in it, not fixed one by one);
- the item quotes Kaggle's error, says where the failed round's run log is, and carries
  the direction or pinned idea that round followed, so the fix counts as that work;
- the round that follows it prints ``FIX_CHECK: passed <evidence>`` when the error can
  no longer happen (``FIX_CHECK: failed <why>`` otherwise). Only a passing check is
  offered for submission (``submission.steered_candidates``), first in the bell;
- a fix that fails its check, or whose submission Kaggle fails again, is tried once
  more; after :data:`MAX_ATTEMPTS` the person is asked in the bell instead.

Nothing here submits: a fix round is offered like any other, and only the person's
yes sends it.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

MAX_ATTEMPTS = 2
# Failures older than this are history, not work: the loop has long moved on from them.
WINDOW_DAYS = 7
_TIMEOUT = re.compile(r"exceeded the allowed runtime|timed? ?out|time limit", re.I)
_MEMORY = re.compile(r"out of memory|memory limit|\bOOM\b|exceeded the allowed memory", re.I)


def short_error(error: str) -> str:
    """Kaggle's reason in a few words, for labels and offers."""
    if _TIMEOUT.search(error or ""):
        return "timed out on Kaggle"
    if _MEMORY.search(error or ""):
        return "ran out of memory on Kaggle"
    return "failed on Kaggle"


def _hint(error: str) -> str:
    if _TIMEOUT.search(error or ""):
        return (
            "Kaggle reruns the notebook on the hidden test set, which can be larger than "
            "the public one: measure how the run's time grows with the data, add a guard "
            "that stops work well inside the limit, and cut what costs the most (calls, "
            "tasks, epochs, model size)."
        )
    if _MEMORY.search(error or ""):
        return "Measure peak memory, then load less at once (batches, smaller dtypes, streaming)."
    return "Reproduce the failure from the run's log, fix its cause, and check it is gone."


def _log_hint(artifacts_dir: Path, round_num: int) -> str:
    from researchforge.kaggle.submission import round_outputs

    out = round_outputs(artifacts_dir, round_num)
    ref = str(out.get("kernel_ref") or "")
    if ref:
        slug = ref.split("/")[-1]
        saved = artifacts_dir / "kernel_outputs" / slug / f"{slug}.log"
        where = f"saved at {saved}" if saved.is_file() else f"`kaggle kernels output {ref}`"
        return f"its notebook {ref} (run log {where})"
    return "its run log"


def _finished(artifacts_dir: Path, round_num: int) -> bool:
    from researchforge.kaggle.submission import round_outputs

    if round_outputs(artifacts_dir, round_num).get("pending"):
        return False
    db = artifacts_dir / "experiment_log.db"
    if not db.exists():
        return False
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT completed_at FROM experiments WHERE round_num = ?", (round_num,)
        ).fetchone()
    return bool(row and row[0])


def fix_check(artifacts_dir: Path, round_num: int) -> dict[str, Any] | None:
    """What a round printed as its FIX_CHECK line (kept in its outputs), or None."""
    from researchforge.kaggle.submission import round_outputs

    got = round_outputs(artifacts_dir, round_num).get("fix_check")
    return got if isinstance(got, dict) else None


def fixed_by(artifacts_dir: Path, round_num: int) -> dict[str, Any] | None:
    """The fix item a round followed, or None when it followed none."""
    from researchforge.kaggle import steering

    for d in steering.load(artifacts_dir):
        if d.fixes and round_num in d.followed_by:
            return {"id": d.id, **d.fixes}
    return None


def unproven(artifacts_dir: Path, round_num: int) -> bool:
    """A fix round that has not shown its error is gone: never offered for submission."""
    if fixed_by(artifacts_dir, round_num) is None:
        return False
    check = fix_check(artifacts_dir, round_num)
    return not (check and check.get("passed"))


def queue_fixes(artifacts_dir: Path) -> list[str]:
    """Add fix items for new Kaggle failures and failed fixes. Returns a line for each."""
    from researchforge.kaggle import control, steering
    from researchforge.kaggle.submission import leaderboard_by_round

    lines: list[str] = []
    items = steering.load(artifacts_dir)
    handled = {
        int(n)
        for d in items
        if d.fixes
        for n in (d.fixes.get("round"), d.fixes.get("after_round"))
        if isinstance(n, int)
    }

    # Kaggle failed a submission: the newest round per distinct error.
    by_error: dict[str, list[int]] = {}
    lb = leaderboard_by_round(artifacts_dir)
    since = (datetime.now() - timedelta(days=WINDOW_DAYS)).isoformat()
    for num, v in lb.items():
        if v.get("status") == "error" and str(v.get("date") or "").replace(" ", "T") >= since:
            by_error.setdefault(str(v.get("error") or "")[:160], []).append(num)
    for rounds in by_error.values():
        newest = max(rounds)
        if newest in handled:
            continue
        before = fixed_by(artifacts_dir, newest)  # a fix round Kaggle failed too
        line = _queue(
            artifacts_dir,
            newest,
            str(lb[newest].get("error") or ""),
            others=sorted(r for r in rounds if r != newest),
            before=before,
            after_round=None,
        )
        handled.add(newest)
        lines.append(line)

    # A fix round ran but did not show its error is gone: once more, then ask.
    for d in steering.load(artifacts_dir):
        if not d.fixes or not d.followed_by:
            continue
        ran = d.followed_by[-1]
        if ran in handled or not _finished(artifacts_dir, ran):
            continue
        check = fix_check(artifacts_dir, ran)
        if check and check.get("passed"):
            continue
        why = (check or {}).get("evidence") or "it printed no FIX_CHECK line"
        lines.append(
            _queue(
                artifacts_dir,
                int(d.fixes.get("round") or ran),
                str(d.fixes.get("error") or ""),
                others=[],
                before={"id": d.id, **d.fixes},
                after_round=ran,
                last_try=f"R{ran} did not fix it: {why}",
            )
        )
        handled.add(ran)
    if lines:
        control.notify(artifacts_dir, "alert", "🔧 " + " · ".join(lines)[:300])
    return lines


def _queue(
    artifacts_dir: Path,
    round_num: int,
    error: str,
    *,
    others: list[int],
    before: dict[str, Any] | None,
    after_round: int | None,
    last_try: str = "",
) -> str:
    from researchforge.kaggle import control, steering

    attempt = int((before or {}).get("attempt") or 0) + 1
    origin = list((before or {}).get("origin") or _origin(artifacts_dir, round_num))
    target = int((before or {}).get("round") or round_num)
    what = short_error(error)
    if attempt > MAX_ATTEMPTS:
        control.request_interaction(
            artifacts_dir,
            f"R{target} {what} and {MAX_ATTEMPTS} fix rounds did not get past it "
            f"({last_try or error[:200]}). Try another fix, or leave it?",
            kind="other",
            options=["try another fix", "leave it"],
            default="leave it",
            answer_key=f"fix:{target}",
            source="fix rounds",
        )
        # Kept (removed from the queue at once) so the question is asked once, not every check.
        marker = steering.add(
            artifacts_dir,
            f"(asked you about R{target}: {what})",
            via="kaggle",
            fixes={"round": target, "after_round": after_round, "attempt": attempt, "asked": True},
        )
        steering.remove(artifacts_dir, marker.id)
        return f"R{target} {what} after {MAX_ATTEMPTS} fixes: asked you"
    steer = " ".join(f"It continues {o['what']}: {o['text'][:200]}" for o in origin)
    same = f" {', '.join(f'R{r}' for r in others)} failed the same way." if others else ""
    text = (
        f"Fix R{target}: {what}. Kaggle said: {error[:400]}{same} "
        + (f"{last_try}. " if last_try else "")
        + f"Read why in {_log_hint(artifacts_dir, after_round or target)}. {_hint(error)} "
        "Print one line `FIX_CHECK: passed <evidence the error cannot happen again>` "
        "(or `FIX_CHECK: failed <why>`): only a passing fix is offered for submission. "
        + (f"{steer} " if steer else "")
        + f"(attempt {attempt} of {MAX_ATTEMPTS})"
    )
    steering.add(
        artifacts_dir,
        text,
        via="kaggle",
        fixes={
            "round": target,
            "after_round": after_round,
            "error": error[:1000],
            "attempt": attempt,
            "origin": origin,
        },
    )
    return f"R{target} {what}: fix round {attempt} queued first in Up next"


def _origin(artifacts_dir: Path, round_num: int) -> list[dict[str, str]]:
    """The person's directions and pinned ideas the failed round followed."""
    from researchforge.kaggle.steering import steered_rounds

    return [
        {
            "id": str(g.get("id")),
            "text": str(g.get("text") or ""),
            "what": "your direction" if g.get("kind") == "direction" else "your pinned idea",
        }
        for g in steered_rounds(artifacts_dir).get(round_num, [])
        if g.get("how") == "followed" and g.get("kind") in ("direction", "pinned_idea")
    ]


def answered(artifacts_dir: Path) -> list[str]:
    """Act on "try another fix" answers: one more fix round each."""
    from researchforge.kaggle import control, steering

    lines = []
    asked = {
        int(d.fixes["round"])
        for d in steering.load(artifacts_dir)
        if d.fixes.get("asked") is True and isinstance(d.fixes.get("round"), int)
    }
    for q in control.load_interactions(artifacts_dir):
        key = str(q.answer_key or "")
        if not key.startswith("fix:") or q.status != "answered":
            continue
        if str(q.answer or "").strip().lower() != "try another fix":
            continue
        num = int(key[4:]) if key[4:].isdigit() else None
        if num is None or num not in asked:
            continue
        error = next(
            (
                str(d.fixes["error"])
                for d in steering.load(artifacts_dir)
                if d.fixes.get("round") == num and d.fixes.get("error")
            ),
            "",
        )
        lines.append(
            _queue(artifacts_dir, num, error, others=[], before={"round": num}, after_round=None)
        )
        items = steering.load(artifacts_dir)
        for d in items:
            if d.fixes.get("asked") is True and d.fixes.get("round") == num:
                d.fixes["asked"] = "answered"
        steering._save(artifacts_dir, items)
    return lines
