"""Submitting an approved round to Kaggle.

A submission is only ever made when the person answers "yes" to a round's
submission question (the dashboard's button or ``kaggle respond``). An
unattended run may not answer it. Each round records what it produced
(``round_outputs.json``): the Kaggle kernel and version it ran as, and the
submission file it wrote. A code competition is submitted as that kernel's
output; any other competition as the file.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, read_json

OUTPUTS_FILE = "round_outputs.json"
LOG_FILE = "submissions_made.jsonl"
REFUSED_FILE = "submissions_refused.jsonl"  # what Kaggle said when it refused one
_ROUND_IN_QUESTION = re.compile(r"\bSubmit R(\d+)\b")


def record_round_outputs(artifacts_dir: Path, round_num: int, outputs: dict[str, Any]) -> None:
    data = read_json(artifacts_dir / OUTPUTS_FILE, {}) or {}
    data = data if isinstance(data, dict) else {}
    data[str(round_num)] = outputs
    atomic_write_json(artifacts_dir / OUTPUTS_FILE, data)


def round_outputs(artifacts_dir: Path, round_num: int) -> dict[str, Any]:
    data = read_json(artifacts_dir / OUTPUTS_FILE, {}) or {}
    got = data.get(str(round_num)) if isinstance(data, dict) else None
    return got if isinstance(got, dict) else {}


def round_of(answer_key: str, question: str) -> int | None:
    """The round a submission question is about."""
    if answer_key.startswith("round:") and answer_key[6:].isdigit():
        return int(answer_key[6:])
    m = _ROUND_IN_QUESTION.search(question)
    return int(m.group(1)) if m else None


def submittable(artifacts_dir: Path, round_num: int) -> bool:
    """Whether a round left something Kaggle takes: a submission file it wrote.

    A notebook round counts only when its run wrote one: Kaggle refuses a notebook
    version without it (400 Bad Request), as with a notebook that ran out of time.
    The file then comes from the notebook's output, so the kept copy may be gone.
    """
    out = round_outputs(artifacts_dir, round_num)
    if out.get("pending"):
        return False
    file = str(out.get("submission_file") or "")
    return bool(file) and (bool(out.get("kernel_ref")) or Path(file).is_file())


def recover_notebook_file(artifacts_dir: Path, round_num: int) -> bool | None:
    """Look in a notebook round's Kaggle output for a submission file it never kept.

    Older rounds kept none, and a download can fail. True when one was found (and is
    now recorded), False when the run's output has none, None when Kaggle could not be
    read. The output listing comes in pages and Kaggle rate-limits it, so it is only
    complete enough to say "none" when the run's own log (``<kernel>.log``) came too.
    """
    from researchforge.kaggle.api import fetch_kernel_output

    out = round_outputs(artifacts_dir, round_num)
    ref = str(out.get("kernel_ref") or "")
    if not ref or out.get("pending"):
        return None
    if out.get("submission_file"):
        return True
    got = fetch_kernel_output(
        ref,
        artifacts_dir / "kernel_outputs" / ref.split("/")[-1],
        file_pattern=r"^submission\.",
        timeout=300,
    )
    written = sorted(p for p in got if p.name.startswith("submission."))
    if not written:
        return False if any(p.name == f"{ref.split('/')[-1]}.log" for p in got) else None
    record_round_outputs(artifacts_dir, round_num, {**out, "submission_file": str(written[0])})
    return True


def withdraw_unsubmittable(artifacts_dir: Path) -> list[int]:
    """Close open submission questions for notebook rounds that wrote no submission file.

    Approving one could only fail: Kaggle refuses a notebook version that wrote no
    submission. Older offers were made for any finished notebook. The monitor calls
    this between its cycles; returns the rounds whose questions it closed.
    """
    from researchforge.kaggle import control

    closed = []
    for item in control.open_interactions(artifacts_dir):
        if item.kind != "submission_approval":
            continue
        num = round_of(item.answer_key, item.question)
        if num is None or submittable(artifacts_dir, num):
            continue
        if recover_notebook_file(artifacts_dir, num) is not False:
            continue  # found one, Kaggle could not be read, or not a notebook round
        control.dismiss(
            artifacts_dir,
            item.id,
            f"withdrawn: R{num}'s notebook wrote no submission file, so Kaggle cannot score it",
        )
        closed.append(num)
    return closed


def _asked_or_submitted(artifacts_dir: Path) -> set[int]:
    from researchforge.kaggle import control

    rounds = {
        round_of(q.answer_key, q.question)
        for q in control.load_interactions(artifacts_dir)
        if q.kind == "submission_approval"
    }
    log = artifacts_dir / LOG_FILE
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError, AttributeError):
                rounds.add(int(json.loads(line).get("round")))
    return {r for r in rounds if r is not None}


def best_round(artifacts_dir: Path) -> tuple[int, str, float] | None:
    """(round, title, metric) of the best scored round in this competition's log."""
    import sqlite3

    db = artifacts_dir / "experiment_log.db"
    if not db.exists():
        return None
    meta = read_json(artifacts_dir / "competition_metadata.json", {}) or {}
    minimize = str(meta.get("metric_direction", "")).lower() == "minimize"
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT round_num, patch_description, full_metric, full_improved FROM experiments "
            "WHERE full_passed = 1 AND full_metric IS NOT NULL ORDER BY round_num"
        ).fetchall()
    if not rows:
        return None
    best = min(rows, key=lambda r: r[2]) if minimize else max(rows, key=lambda r: r[2])
    if not best[3]:  # the best score never counted as an improvement (e.g. the first one)
        return None
    return int(best[0]), str(best[1] or f"Round {best[0]}"), float(best[2])


def unscored_rounds(artifacts_dir: Path) -> list[tuple[int, str]]:
    """(round, title) of rounds that ran to the end with no score but left something to submit.

    A public notebook for a hidden-test competition prints no local score; Kaggle's
    leaderboard is the only measure of it, so it is offered for submission unscored.
    """
    import sqlite3

    from researchforge.kaggle.benchmark_runner import UNSCORED_PATTERN
    from researchforge.kaggle.experiment_loop import archived_rounds

    db = artifacts_dir / "experiment_log.db"
    if not db.exists():
        return []
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT round_num, patch_description, rejected_reason FROM experiments "
            "WHERE full_passed = 0 ORDER BY round_num"
        ).fetchall()
    hidden = archived_rounds(artifacts_dir)
    return [
        (int(num), str(title or f"Round {num}"))
        for num, title, reason in rows
        if num not in hidden
        and UNSCORED_PATTERN.search(str(reason or ""))
        and submittable(artifacts_dir, int(num))
    ]


def offer_missed(artifacts_dir: Path) -> int | None:
    """Ask about rounds never asked about (or submitted). Returns the first one asked.

    The best scored round, and every round that ran unscored but left a submission.
    The offer after a round can be missed: a round run by run-parallel or collected
    late, a crash between saving the round and asking, or a round that only later
    gained a file to submit. The monitor calls this between its cycles.
    """
    from researchforge.kaggle import control

    asked = _asked_or_submitted(artifacts_dir)
    setup = read_json(artifacts_dir / "overnight_setup.json", {}) or {}
    target = str(setup.get("selected_gpu") or "cpu")
    candidates: list[tuple[int, str, float | None]] = []
    best = best_round(artifacts_dir)
    if best is not None:
        candidates.append(best)
    candidates += [(num, title, None) for num, title in unscored_rounds(artifacts_dir)]
    first = None
    for num, title, metric in candidates:
        if num in asked:
            continue
        asked.add(num)
        if control.offer_submission(artifacts_dir, num, title, metric, target) is not None:
            first = first if first is not None else num
    return first


def is_code_competition(artifacts_dir: Path, slug: str) -> bool:
    """Kaggle's own answer, asked now; the saved metadata only when Kaggle cannot be asked.

    A wrong saved flag would send a file to a notebook-only competition (or the reverse),
    so the saved metadata is corrected when Kaggle disagrees.
    """
    from researchforge.kaggle.competition_reader import _fetch_competition_api

    path = artifacts_dir / "competition_metadata.json"
    meta = read_json(path, {}) or {}
    meta = meta if isinstance(meta, dict) else {}
    live = _fetch_competition_api(slug)
    if live is None:
        return bool(meta.get("is_code_competition"))
    if path.exists() and bool(meta.get("is_code_competition")) != live:
        atomic_write_json(path, {**meta, "is_code_competition": live})
    return live


def submit_command(artifacts_dir: Path, slug: str, round_num: int, message: str) -> list[str]:
    """The `kaggle competitions submit` argv for one round. ValueError if it cannot be made."""
    out = round_outputs(artifacts_dir, round_num)
    code = is_code_competition(artifacts_dir, slug)
    file = str(out.get("submission_file") or "")
    if code:
        if not out.get("kernel_ref"):
            raise ValueError(
                f"R{round_num} did not run as a Kaggle notebook, and this competition only "
                "accepts notebook submissions; nothing was submitted"
            )
        if not file and recover_notebook_file(artifacts_dir, round_num):
            file = str(round_outputs(artifacts_dir, round_num).get("submission_file") or "")
        if not file:
            raise ValueError(
                f"R{round_num}'s notebook wrote no submission file (it may have run out of "
                "time), and Kaggle only takes a notebook version that wrote one; "
                "nothing was submitted"
            )
        return [
            "kaggle", "competitions", "submit", slug,
            "-k", str(out["kernel_ref"]),
            "-v", str(out.get("kernel_version") or 1),
            "-f", Path(file).name if file else "submission.csv",
            "-m", message,
        ]  # fmt: skip
    if not file or not Path(file).is_file():
        raise ValueError(
            f"R{round_num} left no submission file"
            + (" (it ran before rounds kept their output)" if not out else "")
            + "; nothing was submitted"
        )
    return ["kaggle", "competitions", "submit", slug, "-f", file, "-m", message]


_REMAINING = re.compile(r"(\d+) submissions? remaining today", re.I)
_OUTDATED_CLI = re.compile(r"^Warning: Looks like you're using an outdated.*$\n?", re.M)


def _log_entries(artifacts_dir: Path) -> list[dict[str, Any]]:
    log = artifacts_dir / LOG_FILE
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(ValueError):
            entry = json.loads(line)
            if isinstance(entry, dict):
                out.append(entry)
    return out


def daily_allowance(artifacts_dir: Path) -> dict[str, Any] | None:
    """Submissions left today on Kaggle, whose day ends at 00:00 UTC.

    The lowest of what is known wins: Kaggle's count in its reply to our last
    submission today that gave one, less our submissions after it (Kaggle's reply
    does not always carry the count); and the competition's daily limit less today's
    submissions, ours or in the listing the monitor keeps (it lags a new one).
    ``{"remaining", "max", "after_round", "at", "resets_at"}`` (``resets_at`` in local
    time; ``max`` and ``after_round`` may be None), or None when unknown.
    """
    now = datetime.now(UTC)
    resets = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), UTC)
    snap = read_json(artifacts_dir / "submissions.json", {}) or {}
    max_daily = snap.get("max_daily") if isinstance(snap.get("max_daily"), int) else None
    ours: list[dict[str, Any]] = []  # today's, newest first
    for entry in reversed(_log_entries(artifacts_dir)):
        try:
            at = datetime.fromisoformat(str(entry["at"])).astimezone(UTC)
        except (KeyError, ValueError):
            break
        if at.date() != now.date():
            break
        ours.append(entry)
    found: list[tuple[int, Any, str]] = []  # (remaining, after_round, at)
    for later, entry in enumerate(ours):
        m = _REMAINING.search(str(entry.get("reply") or ""))
        if m:
            last = ours[0]
            found.append((max(0, int(m.group(1)) - later), last.get("round"), str(last["at"])))
            break
    if max_daily is not None:
        today = now.date().isoformat()
        listed = sum(
            1
            for row in snap.get("submissions") or []
            if isinstance(row, dict) and str(row.get("date") or "").startswith(today)
        )
        used = max(listed, len(ours))
        found.append((max(0, max_daily - used), None, str(snap.get("fetched_at") or "")))
    if not found:
        return None
    remaining, after_round, since = min(found, key=lambda f: f[0])
    return {
        "remaining": remaining,
        "max": max_daily,
        "after_round": after_round,
        "at": since,
        "resets_at": resets.astimezone().isoformat(timespec="minutes"),
    }


def _limit_reached(artifacts_dir: Path) -> str:
    left = daily_allowance(artifacts_dir)
    if left is None or left["remaining"] > 0:
        return ""
    resets = datetime.fromisoformat(left["resets_at"]).strftime("%H:%M")
    if left["after_round"] is None:
        used = f"all {left['max']} of today's submissions are used"
    else:
        used = f"none are left after R{left['after_round']} ({left['at'][11:16]})"
    return (
        f"Kaggle's daily submission limit is used up: {used}. It resets at 00:00 UTC "
        f"({resets} here); the question stays open, approve it again then"
    )


def submit_round(artifacts_dir: Path, slug: str, round_num: int, title: str) -> str:
    """Submit one round. Returns Kaggle's reply; ValueError (nothing logged) on failure.

    Refused before calling Kaggle when the round was already submitted, another
    submission for this project is in flight, or today's limit is used up.
    """
    done = next((e for e in _log_entries(artifacts_dir) if e.get("round") == round_num), None)
    if done is not None:
        raise ValueError(
            f"R{round_num} was already submitted ({done.get('at')}); nothing was submitted again"
        )
    limit = _limit_reached(artifacts_dir)
    if limit:
        raise ValueError(limit)
    lock = artifacts_dir / "submitting.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if datetime.now().timestamp() - lock.stat().st_mtime < 900:
            raise ValueError(
                "another submission for this competition is being sent; wait for it to finish"
            ) from None
        lock.unlink(missing_ok=True)  # left by a crash
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(round_num).encode())
        os.close(fd)
        return _submit(artifacts_dir, slug, round_num, title)
    finally:
        lock.unlink(missing_ok=True)


def _submit(artifacts_dir: Path, slug: str, round_num: int, title: str) -> str:
    message = f"RF R{round_num}: {title}"[:200]
    cmd = submit_command(artifacts_dir, slug, round_num, message)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"kaggle submit did not run: {exc}") from None
    reply = (proc.stdout + proc.stderr).strip()
    # The kaggle CLI exits 0 on some refusals (e.g. the daily limit): read its words too.
    refused = re.search(r"\b(error|forbidden|exceeded|not allowed|failed)\b", reply, re.I)
    if proc.returncode != 0 or (refused and "Successfully" not in reply):
        said = _OUTDATED_CLI.sub("", reply).strip()
        with (artifacts_dir / REFUSED_FILE).open("a", encoding="utf-8") as fh:
            refusal = {
                "at": datetime.now().isoformat(timespec="seconds"),
                "round": round_num,
                "command": cmd[:-2],
                "exit_code": proc.returncode,
                "reply": said[-2000:],
            }
            fh.write(json.dumps(refusal) + "\n")
        limit = _limit_reached(artifacts_dir)
        if limit and re.search(r"\b400\b|limit|exceeded", said, re.I):
            raise ValueError(limit)
        if re.search(r"\b400\b", said):
            raise ValueError(
                f"Kaggle refused R{round_num} (400 Bad Request, with no reason given). The "
                "usual causes: today's submission limit is used up (it resets at 00:00 UTC), or "
                "for a notebook submission, Kaggle cannot re-run that notebook version or it "
                "wrote no submission file. "
                f"Kaggle said: {said[-200:]}"
            )
        raise ValueError(f"Kaggle refused R{round_num}: {said[-300:] or proc.returncode}")
    entry = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "round": round_num,
        "command": cmd[:-2],  # without the message
        "reply": reply[-300:],
    }
    with (artifacts_dir / LOG_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    # List it now, so the Submissions page and the "listed" notice need no wait.
    with contextlib.suppress(Exception):
        from researchforge.kaggle.cli import _submission_check

        _submission_check(slug, artifacts_dir)
    return reply.splitlines()[-1] if reply else "submitted"


def answer(artifacts_dir: Path, slug: str, qid: str, value: str, comment: str = "") -> str:
    """Answer a submission question; "yes" submits first and is recorded only if it worked.

    Returns a line for the person. Raises ValueError (question left open) when the
    submission cannot be made, KeyError for an unknown question.
    """
    from researchforge.kaggle import control

    item = next((q for q in control.open_interactions(artifacts_dir) if q.id == qid), None)
    if item is None:
        raise KeyError(f"no open question {qid}")
    if value.strip().lower() != "yes":
        control.respond(artifacts_dir, qid, value, comment)
        return f"{qid} answered: {value} — nothing submitted"
    if os.environ.get("RESEARCHFORGE_AUTO_RUN") == "1":
        raise ValueError("an unattended run may not approve a submission; only the person can")
    round_num = round_of(item.answer_key, item.question)
    if round_num is None:
        raise ValueError(f"cannot tell which round {qid} is about; nothing was submitted")
    title = re.search(r"'(.*)'", item.question)
    reply = submit_round(artifacts_dir, slug, round_num, title.group(1) if title else "")
    control.respond(artifacts_dir, qid, "yes", comment)
    return f"R{round_num} submitted to Kaggle: {reply}"
