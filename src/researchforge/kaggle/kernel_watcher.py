"""Public notebook watcher — runs on a schedule and injects new ideas.

The watcher does three things:
  1. Poll for NEW public kernels since the last check
  2. Extract techniques/ideas (never copies code — extracts concepts)
  3. Inject new hypotheses into the experiment DAG, notify if a
     kernel with a meaningfully better score appeared

Runs as a local background process (laptop must stay on).
Schedule is stored in .overnight_<slug>/watcher_config.json.

Key rule: the watcher NEVER tests ideas using public LB scores.
Every idea is validated only on the user's own CV folds.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchforge.ai.providers import AiProvider


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass
class WatcherConfig:
    """Persisted watcher configuration."""

    competition_slug: str
    interval_hours: float  # how often to poll
    enabled: bool = True
    last_checked_at: str = ""
    seen_kernel_refs: list[str] = field(default_factory=list)
    seen_submission_ids: list[str] = field(default_factory=list)  # notified submissions
    # ref -> the last run time seen: a notebook re-run since then is looked at again.
    kernel_runs: dict[str, str] = field(default_factory=dict)
    notification_threshold_delta: float = 0.005  # notify if a new kernel's LB is this far ahead

    def save(self, artifacts_dir: Path) -> None:
        from researchforge.kaggle.workspace import atomic_write_json

        atomic_write_json(artifacts_dir / "watcher_config.json", asdict(self))

    @classmethod
    def load(cls, artifacts_dir: Path) -> WatcherConfig | None:
        path = artifacts_dir / "watcher_config.json"
        if not path.exists():
            return None
        with path.open() as fh:
            data = json.load(fh)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class WatcherResult:
    """What the watcher found in one poll cycle."""

    checked_at: str
    new_kernels_found: int
    new_ideas_extracted: list[str]
    better_kernel: dict[str, Any] | None  # {title, ref, lb_score, delta} if significantly better
    notification_sent: bool
    updated_kernels_found: int = 0  # known notebooks re-run since last seen


# ---------------------------------------------------------------------------
# Main watch cycle
# ---------------------------------------------------------------------------


def run_watch_cycle(
    artifacts_dir: Path,
    ai_provider: AiProvider | None = None,
    baseline_score: float | None = None,
    dry_run: bool = False,
    check_submissions: bool = True,
) -> WatcherResult:
    """Run one poll cycle — fetch new kernels, extract ideas, inject hypotheses.

    Args:
        artifacts_dir: .overnight_<slug>/ directory
        ai_provider: AI for idea extraction (falls back to title parsing if None)
        baseline_score: Current best known score (to detect significantly better kernels)
        dry_run: If True, don't persist anything (just report)
        check_submissions: Also poll submission status (the monitor runs that
            as its own sensor and passes False)

    Returns:
        WatcherResult describing what was found
    """
    from researchforge.kaggle.api import list_competition_kernels
    from researchforge.kaggle.notebook_baseline import _extract_lb_score

    config = WatcherConfig.load(artifacts_dir)
    if config is None:
        return WatcherResult(
            checked_at=datetime.now().isoformat(),
            new_kernels_found=0,
            new_ideas_extracted=[],
            better_kernel=None,
            notification_sent=False,
        )

    slug = config.competition_slug
    checked_at = datetime.now().isoformat()

    notification_sent = False
    if check_submissions and not dry_run:
        # Independent of kernels: a submission can resolve on a quiet day.
        completed = _check_new_completed_submissions(
            slug, config.seen_submission_ids, artifacts_dir
        )
        for sub in completed:
            config.seen_submission_ids.append(sub["id"])
            notification_sent = (
                _notify_submission_complete(slug, sub, artifacts_dir) or notification_sent
            )
        if completed:
            config.save(artifacts_dir)

    from researchforge.kaggle.hypothesis_pool import our_best_public_lb, setup_direction

    # Newest first (catches new notebooks) and Kaggle's best-score order (its rank).
    # API errors propagate so callers (the monitor's sensor) can record them.
    recent = list_competition_kernels(slug, sort_by="dateCreated", page_size=50)
    popular = list_competition_kernels(slug, sort_by="scoreDescending", page_size=50)
    # Notebooks re-run since they were last seen: their authors keep improving them.
    recently_run = list_competition_kernels(slug, sort_by="dateRun", page_size=50)

    # Deduplicate
    all_kernels = {k.ref: k for k in recent + popular + recently_run}
    if not dry_run:
        # Votes and claimed scores change; keep the pool's links current.
        annotate_kernel_ideas(artifacts_dir, list(all_kernels.values()))

    # New notebooks, and known ones with a newer run than the one last seen
    seen = set(config.seen_kernel_refs)
    new_kernels = [k for k in all_kernels.values() if k.ref not in seen]
    updated = [
        k
        for k in all_kernels.values()
        if k.ref in seen
        and config.kernel_runs.get(k.ref)
        and str(k.last_run_time or "") > config.kernel_runs[k.ref]
    ]

    # Compare with our best leaderboard score; the setup baseline when nothing is scored.
    direction = setup_direction(artifacts_dir)
    our_best = our_best_public_lb(artifacts_dir, direction)
    reference = our_best if our_best is not None else baseline_score
    board: dict[str, Any] = {}
    if not dry_run:
        from researchforge.kaggle.kernel_board import update_board

        board = update_board(
            artifacts_dir,
            recent + recently_run,
            popular,
            direction=direction,
            our_best_lb=reference,
            updated_refs={k.ref for k in updated},
        )
        for k in all_kernels.values():
            config.kernel_runs[k.ref] = str(k.last_run_time or "")

    if not new_kernels and not updated:
        if not dry_run:
            config.last_checked_at = checked_at
            config.save(artifacts_dir)
        return WatcherResult(
            checked_at=checked_at,
            new_kernels_found=0,
            new_ideas_extracted=[],
            better_kernel=None,
            notification_sent=notification_sent,
        )

    # Extract ideas from new and re-run notebooks
    new_ideas = _extract_ideas_from_kernels(new_kernels + updated, ai_provider)

    # Did a new or re-run notebook claim a score well ahead of ours? (the biggest lead)
    better_kernel = None
    if reference is not None:
        threshold = config.notification_threshold_delta
        sign = 1 if direction.startswith("max") else -1
        claims = {r["ref"]: r.get("claimed_score") for r in board.get("rows", [])}
        for k in new_kernels + updated:
            lb = claims.get(k.ref)
            lb = lb if isinstance(lb, (int, float)) else _extract_lb_score(k.title)
            # Signed gap: positive means the notebook is ahead, whichever way the metric runs.
            if lb is not None and sign * (lb - reference) > threshold:
                delta = round(sign * (lb - reference), 4)
                if better_kernel is None or delta > float(better_kernel["delta"]):
                    better_kernel = {
                        "title": k.title,
                        "ref": k.ref,
                        "author": k.author,
                        "lb_score": lb,
                        "delta": delta,
                        "updated": k in updated,
                    }

    # Inject new ideas into the pending hypothesis pool
    if not dry_run:
        _inject_ideas(artifacts_dir, new_ideas, new_kernels + updated)
        annotate_kernel_ideas(artifacts_dir, list(all_kernels.values()))
        config.seen_kernel_refs.extend([k.ref for k in new_kernels])
        config.last_checked_at = checked_at
        config.save(artifacts_dir)

        # Send notification if something notably better appeared
        if better_kernel:
            notification_sent = _notify_better_kernel(slug, better_kernel, artifacts_dir)

    return WatcherResult(
        checked_at=checked_at,
        new_kernels_found=len(new_kernels),
        updated_kernels_found=len(updated),
        new_ideas_extracted=new_ideas,
        better_kernel=better_kernel,
        notification_sent=notification_sent,
    )


# ---------------------------------------------------------------------------
# Submission status checking
# ---------------------------------------------------------------------------


def fetch_submissions(slug: str, timeout: int = 30) -> tuple[list[dict[str, Any]], str]:
    """Return ``(rows, error)`` from ``kaggle competitions submissions <slug> --csv``.

    Rows use lower-cased CSV headers (ref, filename, date, description, status,
    publicscore, privatescore). ``error`` is "" on success.
    """
    import csv
    import io
    import subprocess

    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", slug, "--csv"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return [], "kaggle CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return [], f"kaggle CLI timed out after {timeout}s"
    except OSError as exc:
        return [], str(exc)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip().splitlines()
        return [], err[-1][:200] if err else f"kaggle CLI exit {result.returncode}"
    lines = [
        ln
        for ln in result.stdout.splitlines()
        if ln.strip() and not ln.strip().lower().startswith("warning")
    ]
    if len(lines) < 2:
        return [], ""
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = [
        {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()} for row in reader
    ]
    # The CSV leaves out why a submission failed; a notebook that ran out of time on
    # Kaggle is listed "complete" with no score, and only the API says "exceeded the
    # allowed runtime".
    errors = submission_errors(slug)
    for row in rows:
        if errors.get(str(row.get("ref") or "")):
            row["errordescription"] = errors[str(row["ref"])]
    return rows, ""


_ERRORS_SNIPPET = """
import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions(sys.argv[1]) or []
print(json.dumps({str(s.ref): s.error_description for s in subs if s.error_description}))
"""


def submission_errors(slug: str) -> dict[str, str]:
    """{submission ref: Kaggle's error text} for this competition's failed submissions."""
    import json

    from researchforge.kaggle.competition_reader import run_on_kaggle_python

    out = run_on_kaggle_python(_ERRORS_SNIPPET, slug).strip().splitlines()
    try:
        got = json.loads(out[-1]) if out else {}
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in got.items()} if isinstance(got, dict) else {}


def save_submissions_snapshot(artifacts_dir: Path, rows: list[dict[str, Any]], error: str) -> None:
    """Persist the latest submission listing to submissions.json for the dashboard."""
    from researchforge.kaggle.workspace import atomic_write_json, read_json

    path = artifacts_dir / "submissions.json"
    prev = read_json(path, {}) or {}
    snapshot = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "error": error,
        # Keep the last good listing when a fetch fails.
        "submissions": rows if not error else prev.get("submissions", []),
        # Submissions Kaggle allows per day; asked once, so "N of M left today" can show.
        "max_daily": prev.get("max_daily") or _max_daily(artifacts_dir),
    }
    atomic_write_json(path, snapshot)


def _max_daily(artifacts_dir: Path) -> int | None:
    from researchforge.kaggle.competition_reader import competition_facts

    value = competition_facts(artifacts_dir.name.removeprefix(".overnight_")).get("max_daily")
    return int(value) if value else None


def _check_new_completed_submissions(
    slug: str,
    already_seen: list[str],
    artifacts_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return submissions that just became 'complete' and haven't been notified yet.

    Calls `kaggle competitions submissions <slug> --csv` — no Claude Code
    permissions needed; runs as a direct subprocess under the launchd watcher.
    When ``artifacts_dir`` is given the full listing is also snapshotted.

    Each returned dict: {id, filename, date, publicscore, privatescore, status}
    """
    rows, error = fetch_submissions(slug)
    if artifacts_dir is not None:
        with contextlib.suppress(OSError):
            save_submissions_snapshot(artifacts_dir, rows, error)
    completed = []
    for row in rows:
        # Status is "SubmissionStatus.COMPLETE" — check case-insensitively
        status = row.get("status", "").lower()
        if "complete" in status and "error" not in status:
            sub_id = row.get("ref", "")
            if sub_id and sub_id not in already_seen:
                row["id"] = sub_id
                completed.append(row)
    return completed


def _notify_submission_complete(slug: str, sub: dict[str, Any], artifacts_dir: Path) -> bool:
    """Send ntfy push when a Kaggle submission resolves to 'complete'."""
    return _notify_submission_event(slug, sub, artifacts_dir, "scored")


def _notify_submission_event(
    slug: str, sub: dict[str, Any], artifacts_dir: Path, event: str
) -> bool:
    """Push one submission event: "listed" (Kaggle is scoring it), "scored" or "failed"."""
    try:
        from researchforge.notifications.service import Channel, Notification, send_notification

        score = sub.get("publicscore", "") or sub.get("public_score", "")
        private = sub.get("privatescore", "") or sub.get("private_score", "")
        filename = sub.get("filename", "submission")
        date = sub.get("date", "")[:16]  # trim to YYYY-MM-DD HH:MM

        score_str = f"Public LB: {score}" if score else "Score pending"
        private_str = f" | Private: {private}" if private else ""
        msg = {
            "listed": f"{filename} received — Kaggle is scoring it ({date})",
            "failed": f"{filename} failed on Kaggle ({date})"
            + (f": {sub['errordescription'][:200]}" if sub.get("errordescription") else ""),
        }.get(event, f"{filename} resolved — {score_str}{private_str} ({date})")
        title = {
            "listed": f"Kaggle submission received: {slug}",
            "failed": f"Kaggle submission failed: {slug}",
        }.get(event, f"Kaggle submission complete: {slug}")

        # Resolve topic from setup file
        import json as _json
        import os as _os

        topic = _os.environ.get("NTFY_TOPIC", "")
        if not topic:
            setup_path = artifacts_dir / "overnight_setup.json"
            if setup_path.exists():
                topic = _json.loads(setup_path.read_text()).get("ntfy_topic", "")

        import os

        old = os.environ.get("NTFY_TOPIC")
        if topic:
            os.environ["NTFY_TOPIC"] = topic
        try:
            return send_notification(
                Notification(
                    title=title,
                    message=msg,
                    channel=Channel.NTFY,
                    tags=[{"listed": "hourglass", "failed": "x"}.get(event, "trophy")],
                    priority="default" if event == "listed" else "high",
                )
            )
        finally:
            if topic:
                if old is None:
                    os.environ.pop("NTFY_TOPIC", None)
                else:
                    os.environ["NTFY_TOPIC"] = old
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Idea extraction
# ---------------------------------------------------------------------------


_IDEA_EXTRACT_SYSTEM = """\
You are a Kaggle expert. Given a list of new public notebook titles for a competition,
extract 3-8 distinct, concrete technique ideas that might improve the model.
Return ONLY a JSON array of short technique strings — no prose.
Focus on actionable ML techniques, not notebook presentation style.
Example: ["2.5D MIL with CoAtNet-2", "Mixup augmentation", "5-fold stratified CV with seed 42"]
Never suggest copying code — only techniques.
"""


def _extract_ideas_from_kernels(kernels: Any, ai_provider: AiProvider | None) -> list[str]:
    """Extract technique ideas from new kernel titles."""
    if not kernels:
        return []

    titles_text = "\n".join(f"- {k.title} (by {k.author})" for k in kernels[:20])

    if ai_provider is not None:
        try:
            raw = ai_provider.generate(_IDEA_EXTRACT_SYSTEM, titles_text, max_tokens=512)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            ideas = json.loads(raw)
            if isinstance(ideas, list):
                return [str(i) for i in ideas]
        except Exception:
            pass

    # Fallback: use titles directly
    return [k.title for k in kernels[:5] if k.title]


# ---------------------------------------------------------------------------
# Hypothesis injection
# ---------------------------------------------------------------------------


_STOPWORDS = frozenset(
    [
        "with",
        "from",
        "that",
        "this",
        "into",
        "using",
        "use",
        "based",
        "model",
        "models",
        "data",
        "feature",
        "features",
        "train",
        "training",
        "add",
        "try",
        "more",
        "better",
        "improve",
        "kernel",
        "notebook",
        "baseline",
        "score",
        "public",
    ]
)


def _terms(text: str) -> set[str]:
    import re

    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3} - _STOPWORDS


def idea_relevance(artifacts_dir: Path, idea: str) -> dict[str, Any]:
    """How a new idea relates to the current experiment path, from experiment_log.db.

    builds_on_best   — shares terms with rounds that improved (most useful next)
    repeats_failure  — mostly overlaps rounds that failed or did not improve
    new_direction    — unrelated to anything tried so far
    """
    import sqlite3

    words = _terms(idea)
    db = artifacts_dir / "experiment_log.db"
    improved: list[set[str]] = []
    failed: list[set[str]] = []
    if db.exists() and words:
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                from researchforge.kaggle.experiment_loop import archived_rounds

                hidden = archived_rounds(db.parent)
                for num, desc, ok in conn.execute(
                    "SELECT round_num, patch_description, full_improved FROM experiments"
                ):
                    if num not in hidden:
                        (improved if ok else failed).append(_terms(desc or ""))
        except sqlite3.Error:
            pass

    def overlap(group: list[set[str]]) -> float:
        return max((len(words & g) / len(words) for g in group if g), default=0.0)

    best, worst = overlap(improved), overlap(failed)
    if best >= 0.2:
        label = "builds_on_best"
    elif worst >= 0.5:
        label = "repeats_failure"
    else:
        label = "new_direction"
    return {
        "relevance": label,
        "overlap_improved": round(best, 2),
        "overlap_failed": round(worst, 2),
    }


def _inject_ideas(artifacts_dir: Path, ideas: list[str], kernels: Any) -> None:
    """Append new ideas to the pending hypothesis pool, tagged with their relevance."""
    pool_path = artifacts_dir / "pending_hypotheses.json"
    existing: list[dict[str, Any]] = []
    if pool_path.exists():
        try:
            existing = json.loads(pool_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []

    existing_ideas = {h.get("idea", "").lower() for h in existing}
    by_title = {str(k.title).strip().lower(): k for k in kernels}
    added = []
    for idea in ideas:
        if idea.lower() not in existing_ideas:
            # Title-derived ideas (no AI provider) map to exactly one notebook.
            own = by_title.get(idea.strip().lower())
            added.append(
                {
                    "idea": idea,
                    "source": "kernel_watcher",
                    "source_refs": [own.ref] if own else [k.ref for k in kernels[:3]],
                    "injected_at": datetime.now().isoformat(),
                    "tested": False,
                    **(_kernel_fields(own) if own else {}),
                    **idea_relevance(artifacts_dir, idea),
                }
            )

    if added:
        from researchforge.kaggle.workspace import atomic_write_json

        atomic_write_json(pool_path, existing + added)


MAX_NOTEBOOK_CLAIM_FETCHES = 25  # notebooks downloaded per cycle to read their LB claim


def _kernel_fields(kernel: Any) -> dict[str, Any]:
    """What the dashboard shows about the notebook an idea came from."""
    return {
        "kernel_ref": kernel.ref,
        "kernel_title": kernel.title,
        "kernel_votes": kernel.total_votes,
    }


def annotate_kernel_ideas(
    artifacts_dir: Path, kernels: Any, *, max_fetches: int = MAX_NOTEBOOK_CLAIM_FETCHES
) -> int:
    """Link pool entries to their own notebook: ref, votes and the LB score it claims.

    Repairs ideas injected before this was recorded. Matches on the notebook
    title, which is what a title-derived idea is. The claim comes from the title,
    else from the notebook itself — re-read only when the notebook has been re-run
    since (`claim_checked_run`). Returns how many entries changed.
    """
    from researchforge.kaggle.kernel_claims import notebook_claim, title_claim

    pool_path = artifacts_dir / "pending_hypotheses.json"
    if not pool_path.exists():
        return 0
    try:
        pool = json.loads(pool_path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    by_title = {str(k.title).strip().lower(): k for k in kernels}
    changed = 0
    for h in pool:
        if not isinstance(h, dict) or h.get("source") != "kernel_watcher":
            continue
        own = by_title.get(str(h.get("idea", "")).strip().lower())
        if own is None:
            continue
        fields = _kernel_fields(own)
        run = str(own.last_run_time or "")
        if h.get("claim_checked_run") != run:
            claim = title_claim(own.title)
            if claim is None and max_fetches > 0:
                max_fetches -= 1
                claim = notebook_claim(own.ref)
            fields.update(
                claim or {"claimed_score": None, "claim_source": None, "claim_snippet": None}
            )
            fields["claim_checked_run"] = run
        if any(h.get(k) != v for k, v in fields.items()) or h.get("source_refs") != [own.ref]:
            h.update(fields)
            h["source_refs"] = [own.ref]
            changed += 1
    if changed:
        from researchforge.kaggle.workspace import atomic_write_json

        atomic_write_json(pool_path, pool)
    return changed


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------


def _notify_better_kernel(
    slug: str, kernel: dict[str, Any], artifacts_dir: Path | None = None
) -> bool:
    """Send notification when a new kernel beats the current baseline.

    Uses the per-competition ntfy_topic from overnight_setup.json when available,
    falling back to NTFY_TOPIC env var.
    """
    try:
        from researchforge.notifications.service import notify_kernel_found

        # Resolve topic: setup file > env var
        topic = ""
        if artifacts_dir:
            setup_path = artifacts_dir / "overnight_setup.json"
            if setup_path.exists():
                import json as _json

                data = _json.loads(setup_path.read_text())
                topic = data.get("ntfy_topic", "")

        return notify_kernel_found(
            slug=slug,
            title=kernel.get("title", ""),
            lb_score=kernel.get("lb_score"),
            delta=kernel.get("delta"),
            topic=topic,
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Background scheduler loop
# ---------------------------------------------------------------------------


def run_watcher_loop(
    artifacts_dir: Path,
    ai_provider: AiProvider | None = None,
    baseline_score: float | None = None,
) -> None:
    """Run the watcher loop indefinitely (blocking — run in a background thread/process).

    Polls at the interval specified in watcher_config.json.
    """
    while True:
        config = WatcherConfig.load(artifacts_dir)
        if config is None or not config.enabled:
            break

        try:
            result = run_watch_cycle(artifacts_dir, ai_provider, baseline_score)
        except Exception as exc:  # API outage must not kill the loop
            print(
                f"[watcher] {datetime.now().isoformat(timespec='seconds')}: error: {exc}",
                flush=True,
            )
            time.sleep(config.interval_hours * 3600)
            continue

        if result.new_kernels_found > 0:
            print(
                f"[watcher] {result.checked_at}: "
                f"{result.new_kernels_found} new kernels, "
                f"{len(result.new_ideas_extracted)} ideas injected"
            )
            if result.better_kernel:
                print(
                    f"[watcher] ⚡ Better kernel found: "
                    f"{result.better_kernel['title']} "
                    f"(LB {result.better_kernel['lb_score']:.4f})"
                )

        sleep_seconds = config.interval_hours * 3600
        time.sleep(sleep_seconds)
