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
from types import SimpleNamespace
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, read_json

OUTPUTS_FILE = "round_outputs.json"
LOG_FILE = "submissions_made.jsonl"
REFUSED_FILE = "submissions_refused.jsonl"  # what Kaggle said when it refused one
_ROUND_IN_QUESTION = re.compile(r"\bSubmit R(\d+)\b")
_QUOTED_TITLE = re.compile(r"\bSubmit R\d+ '(.*?)' \(")


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


def _collected_without_file(artifacts_dir: Path, round_num: int) -> bool:
    """Whether collecting this notebook round found no submission file in its output.

    The collector downloads the run's log and any ``submission.*`` and records
    ``NO_SUBMISSION_REASON`` when the second is missing, so the record is enough when
    Kaggle will not list the output again (it rate-limits notebooks with large ones).
    """
    import sqlite3

    from researchforge.kaggle.benchmark_runner import NO_SUBMISSION_REASON

    db = artifacts_dir / "experiment_log.db"
    if not db.exists():
        return False
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT rejected_reason FROM experiments WHERE round_num = ?", (round_num,)
        ).fetchone()
    return bool(row) and str(row[0] or "").startswith(NO_SUBMISSION_REASON)


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
        found = recover_notebook_file(artifacts_dir, num)
        if found is None and _collected_without_file(artifacts_dir, num):
            found = False  # Kaggle rate-limits the listing; the collection already looked
        if found is not False:
            continue  # found one, Kaggle could not be read, or not a notebook round
        control.dismiss(
            artifacts_dir,
            item.id,
            f"withdrawn: R{num}'s notebook wrote no submission file, so Kaggle cannot score it",
        )
        closed.append(num)
    return closed


_ROUND_IN_DESCRIPTION = re.compile(r"\bR(\d+)\b|\bround_(\d+)\b")


def leaderboard_by_round(artifacts_dir: Path) -> dict[int, dict[str, Any]]:
    """What Kaggle made of each submitted round, from the watcher's submissions.json.

    A submission names its round in its description ("RF R21: …"; older ones
    "round_14"). Per round: ``public`` (the newest public score, or None), ``status``
    ("scored", "pending", "no_score" when Kaggle finished without one, "error"),
    ``error`` (Kaggle's reason), ``date`` and ``submissions`` (how many were sent).
    """
    snap = read_json(artifacts_dir / "submissions.json", {}) or {}
    subs = snap.get("submissions", []) if isinstance(snap, dict) else []
    out: dict[int, dict[str, Any]] = {}
    for x in sorted(
        (x for x in subs if isinstance(x, dict)), key=lambda x: str(x.get("date") or "")
    ):  # oldest first, so the newest submission of a round has the last word
        m = _ROUND_IN_DESCRIPTION.search(str(x.get("description") or ""))
        if not m:
            continue
        num = int(m.group(1) or m.group(2))
        raw = str(x.get("status") or "").lower().replace("submissionstatus.", "")
        try:
            public: float | None = float(str(x.get("publicscore", x.get("publicScore"))))
        except (TypeError, ValueError):
            public = None
        # Kaggle lists a notebook that ran out of time as "complete" with no score; only
        # its error text (kernel_watcher.submission_errors) says what happened.
        error = str(x.get("errorDescription") or x.get("errordescription") or "")
        status = (
            "scored"
            if public is not None
            else "error"
            if error or "error" in raw
            else "no_score"
            if "complete" in raw
            else "pending"
        )
        prev = out.get(num, {})
        row = {
            "public": public,
            "status": status,
            "error": error,
            "date": str(x.get("date") or ""),
            "submissions": int(prev.get("submissions", 0)) + 1,
        }
        if public is None and prev.get("public") is not None:
            # A later submission still scoring (or unscored) keeps the score it had.
            row = {**prev, "submissions": row["submissions"], "latest": status}
        out[num] = row
    return out


LEADERBOARD_NOTE = (
    "Our submitted rounds as Kaggle scored them: the local metric beside the public "
    "leaderboard score. The LB is what counts. Where the two disagree, the local benchmark "
    "does not predict the LB for that kind of change: weigh a new idea by how similar "
    "rounds did on the LB, not by local gains alone, and say so in the rationale. A round "
    "Kaggle failed (lb_status error, with Kaggle's reason in lb_error, e.g. a notebook that "
    "exceeded the allowed runtime on the hidden test set) will fail the same way again: a "
    "new round built on it must remove the cause (a shorter run, fewer calls or tasks, a "
    "time guard) and say how in its rationale."
)


def leaderboard_table(artifacts_dir: Path, limit: int = 15) -> list[dict[str, Any]]:
    """Submitted rounds, newest first: round, title, local metric, LB score and status."""
    import sqlite3

    lb = leaderboard_by_round(artifacts_dir)
    db = artifacts_dir / "experiment_log.db"
    local: dict[int, tuple[str, float | None]] = {}
    if lb and db.exists():
        with sqlite3.connect(db) as conn:
            for num, title, passed, metric in conn.execute(
                "SELECT round_num, patch_description, full_passed, full_metric FROM experiments"
            ):
                local[int(num)] = (str(title or ""), metric if passed else None)
    rows = []
    for num in sorted(lb, reverse=True)[:limit]:
        title, metric = local.get(num, ("", None))
        v = lb[num]
        rows.append(
            {
                "round": num,
                "title": title[:100],
                "local": metric,
                "lb": v.get("public"),
                "lb_status": v.get("status"),
                **({"lb_error": v["error"]} if v.get("error") else {}),
            }
        )
    return rows


def submitted_rounds(artifacts_dir: Path) -> set[int]:
    """Rounds sent to Kaggle (submissions_made.jsonl)."""
    rounds: set[int] = set()
    log = artifacts_dir / LOG_FILE
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError, AttributeError, TypeError):
                rounds.add(int(json.loads(line).get("round")))
    return rounds


def _asked_or_submitted(artifacts_dir: Path) -> set[int]:
    from researchforge.kaggle import control

    rounds = {
        round_of(q.answer_key, q.question)
        for q in control.load_interactions(artifacts_dir)
        if q.kind == "submission_approval"
    }
    return {r for r in rounds if r is not None} | submitted_rounds(artifacts_dir)


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
    notes: dict[int, str] = {}
    for num, title, metric, note in steered_candidates(artifacts_dir, best):
        notes[num] = note
        candidates.append((num, title, metric))
    first = None
    for num, title, metric in candidates:
        if num in asked:
            continue
        asked.add(num)
        offered = control.offer_submission(
            artifacts_dir, num, title, metric, target, note=notes.get(num, "")
        )
        if offered is not None:
            first = first if first is not None else num
    return first


def steered_candidates(
    artifacts_dir: Path, best: tuple[int, str, float] | None
) -> list[tuple[int, str, float | None, str]]:
    """(round, title, metric, note) of rounds that followed the person's steer and left a file.

    Offered whether or not they beat the best: the person asked for that work, so they
    judge it. The note says what the round followed and, when it fell short, by how much.
    """
    import sqlite3

    from researchforge.kaggle.benchmark_runner import UNSCORED_PATTERN
    from researchforge.kaggle.experiment_loop import archived_rounds
    from researchforge.kaggle.steering import steered_rounds

    followed = {
        n: [g for g in gs if g.get("how") == "followed"]
        for n, gs in steered_rounds(artifacts_dir).items()
    }
    followed = {n: gs for n, gs in followed.items() if gs}
    db = artifacts_dir / "experiment_log.db"
    if not followed or not db.exists():
        return []
    meta = read_json(artifacts_dir / "competition_metadata.json", {}) or {}
    minimize = str(meta.get("metric_direction", "")).lower() == "minimize"
    marks = ",".join("?" * len(followed))
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT round_num, patch_description, full_passed, full_metric, rejected_reason "
            f"FROM experiments WHERE round_num IN ({marks}) ORDER BY round_num",
            list(followed),
        ).fetchall()
    hidden = archived_rounds(artifacts_dir)
    out: list[tuple[int, str, float | None, str]] = []
    for num, title, passed, metric, reason in rows:
        scored = bool(passed) and metric is not None
        if num in hidden or not submittable(artifacts_dir, int(num)):
            continue
        if not scored and not UNSCORED_PATTERN.search(str(reason or "")):
            continue  # failed: whatever file it left is not a result to judge
        g = followed[num][0]
        if g.get("kind") == "fix":
            from researchforge.kaggle.fixes import fix_check, fixed_by, short_error

            fix, check = fixed_by(artifacts_dir, int(num)), fix_check(artifacts_dir, int(num))
            if not fix or not (check and check.get("passed")):
                continue  # a fix is offered only once it showed the error is gone
            note = (
                f"fixes R{fix.get('round')} ({short_error(str(fix.get('error') or ''))}); "
                f"check: {str(check.get('evidence') or 'passed')[:200]}"
            )
            out.append(
                (
                    int(num),
                    str(title or f"Round {num}"),
                    None if not scored else float(metric),
                    note,
                )
            )
            continue
        note = "followed your " + ("direction" if g.get("kind") == "direction" else "pinned idea")
        score = float(metric) if scored else None
        if score is not None and best is not None and best[0] != num:
            beat = score < best[2] if minimize else score > best[2]
            if not beat:
                note += f"; did not beat the best: {score:.4f} vs {best[2]:.4f} (R{best[0]})"
        out.append((int(num), str(title or f"Round {num}"), score, note))
    return out


def round_title(artifacts_dir: Path, round_num: int) -> str:
    """The round's own title from the experiment log ("" if unknown).

    Read from the log, not the offer's text: an offer quotes the title and adds notes
    of its own, so the text cannot say where the title ends.
    """
    import sqlite3

    db = artifacts_dir / "experiment_log.db"
    if not db.exists():
        return ""
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT patch_description FROM experiments WHERE round_num = ?", (round_num,)
        ).fetchone()
    return str(row[0] or "") if row else ""


def submit_now(artifacts_dir: Path, slug: str, round_num: int, comment: str = "") -> str:
    """Submit one round the person picked (Runs, DAG): its offer, answered yes.

    An open offer for the round is reused. Raises ValueError, leaving the offer
    open, when the round left nothing Kaggle takes or Kaggle refuses it.
    """
    from researchforge.kaggle import control

    title = round_title(artifacts_dir, round_num) or f"Round {round_num}"
    item = control.ask_submission(
        artifacts_dir,
        round_num,
        f"Submit R{round_num} '{title}' (picked by you) to Kaggle?",
        source="you",
        notify_person=False,  # they are the one submitting it
    )
    return answer(artifacts_dir, slug, item.id, "yes", comment)


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


# A submission through Kaggle's Python API: the CLI prints only "400 Client Error" when
# Kaggle refuses one, while the response body says why. A notebook (kernel given) or a file.
_SUBMIT_SNIPPET = """
import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
slug, kernel, version, file_name, message = sys.argv[1:6]
api = KaggleApi()
api.authenticate()
try:
    if kernel:
        api.competition_submit_code(
            file_name, message, slug, kernel, int(version) if version else None, quiet=True
        )
    else:
        api.competition_submit(file_name, message, slug, quiet=True)
    print("RF_OK")
except Exception as exc:
    resp = getattr(exc, "response", None)
    body = getattr(resp, "text", "") if resp is not None else ""
    print("RF_ERR " + json.dumps({"error": str(exc), "body": str(body)[:2000]}))
"""


def still_scoring(artifacts_dir: Path) -> str:
    """The submission Kaggle is still scoring ("R65, submitted 21 h ago"), or ""."""
    snap = read_json(artifacts_dir / "submissions.json", {}) or {}
    for x in snap.get("submissions", []) if isinstance(snap, dict) else []:
        if not isinstance(x, dict) or "pending" not in str(x.get("status", "")).lower():
            continue
        m = _ROUND_IN_DESCRIPTION.search(str(x.get("description") or ""))
        what = f"R{m.group(1) or m.group(2)}" if m else "an earlier submission"
        try:
            sent = datetime.fromisoformat(str(x.get("date") or "")[:19])
            hours = (datetime.now(UTC).replace(tzinfo=None) - sent).total_seconds() / 3600
            return f"{what}, submitted {hours:.0f} h ago"
        except ValueError:
            return what
    return ""


def _send(cmd: list[str], slug: str) -> tuple[int, str]:
    """Send the submission once. Returns (exit code, what Kaggle said)."""
    if "-f" in cmd:
        from researchforge.kaggle.competition_reader import run_on_kaggle_python

        def arg(flag: str) -> str:
            return cmd[cmd.index(flag) + 1] if flag in cmd else ""

        out = run_on_kaggle_python(
            _SUBMIT_SNIPPET, slug, arg("-k"), arg("-v"), arg("-f"), arg("-m"), timeout=600
        )
        for line in reversed(out.splitlines()):
            if line.startswith("RF_OK"):
                return 0, f"Successfully submitted to {slug}"
            if line.startswith("RF_ERR "):
                err = json.loads(line[7:])
                body = str(err.get("body") or "").strip()
                return 1, str(err.get("error") or "") + (
                    f" | Kaggle's reason: {body}" if body else ""
                )
        # The API could not be reached at all (no `kaggle` Python): nothing was sent yet.
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"kaggle submit did not run: {exc}") from None
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _submit(artifacts_dir: Path, slug: str, round_num: int, title: str) -> str:
    message = f"RF R{round_num}: {title}"[:200]
    cmd = submit_command(artifacts_dir, slug, round_num, message)
    code, reply = _send(cmd, slug)
    proc = SimpleNamespace(returncode=code)
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
        reason = said.split("Kaggle's reason:")[-1].strip() if "Kaggle's reason:" in said else ""
        busy = still_scoring(artifacts_dir)
        if busy and re.search(r"\b400\b", said):
            # Seen on Kaggle competitions that rerun each submission for hours: a new one is
            # refused until the last is scored.
            raise ValueError(
                f"Kaggle refused R{round_num} while it is still scoring {busy}. This "
                "competition takes one submission at a time: approve again once that one has "
                "its score (the offer stays open)."
                + (f" Kaggle said: {reason[:300]}" if reason else "")
            )
        if reason:
            raise ValueError(f"Kaggle refused R{round_num}: {reason[:400]}")
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
    quoted = _QUOTED_TITLE.search(item.question)  # for a round missing from the log
    title = round_title(artifacts_dir, round_num) or (quoted.group(1) if quoted else "")
    reply = submit_round(artifacts_dir, slug, round_num, title)
    control.respond(artifacts_dir, qid, "yes", comment)
    return f"R{round_num} submitted to Kaggle: {reply}"
