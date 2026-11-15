"""Scheduler configuration for overnight experiments.

Covers three orthogonal scheduling concerns:
  1. Kernel watcher     — how often to poll for new public notebooks
  2. Experiment trigger — when to run the next experiment batch
  3. Morning briefing   — when to send the daily summary notification

Stored in .overnight_<slug>/scheduler_config.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# SchedulerConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """All scheduling config for one overnight competition run."""

    competition_slug: str

    # ── 1. Kernel watcher ──────────────────────────────────────────────────
    watcher_enabled: bool = True
    watcher_interval_hours: float = 6.0  # 4 / 6 / 12 / 24
    watcher_seen_kernel_refs: list[str] = field(default_factory=list)
    watcher_last_checked_at: str = ""
    watcher_notification_threshold: float = 0.005  # notify if new kernel LB > baseline + this

    # ── 2. Experiment trigger ───────────────────────────────────────────────
    # "now"       → start experiments immediately
    # "at_time"   → wait until experiment_start_time (HH:MM local)
    # "manual"    → user calls prepare-round / execute-round manually
    # "after_watcher" → trigger a round whenever watcher finds a better kernel
    experiment_start_mode: str = "now"
    experiment_start_time: str = ""  # "23:00" — only for "at_time"
    experiment_triggered_at: str = ""  # set when auto-trigger fires

    # ── 3. Morning briefing ─────────────────────────────────────────────────
    # "on_completion" → send ntfy immediately when experiments finish
    # "at_time"       → send at briefing_time each day
    # "both"          → send both (immediate + daily summary)
    # "never"         → no scheduled briefing
    briefing_mode: str = "on_completion"
    briefing_time: str = "07:00"  # HH:MM local time, for "at_time" / "both"
    briefing_only_if_improved: bool = False  # if True, skip briefing when nothing improved
    briefing_last_sent_at: str = ""  # to avoid double-sending

    # ── Misc ────────────────────────────────────────────────────────────────
    ntfy_topic: str = ""  # copied from setup for daemon use
    enabled: bool = True

    # ── Sensors (see researchforge.kaggle.sensors) ──────────────────────────
    disabled_sensors: list[str] = field(default_factory=list)
    enabled_sensors: list[str] = field(default_factory=list)  # opt-in built-ins
    custom_sensors: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, artifacts_dir: Path) -> None:
        from researchforge.kaggle.workspace import atomic_write_json

        atomic_write_json(artifacts_dir / "scheduler_config.json", asdict(self))

    @classmethod
    def load(cls, artifacts_dir: Path) -> SchedulerConfig | None:
        path = artifacts_dir / "scheduler_config.json"
        if not path.exists():
            return None
        with path.open() as fh:
            data = json.load(fh)
        # Drop unknown keys for forward-compat
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_watcher_config(cls, watcher_config: Any, competition_slug: str) -> SchedulerConfig:
        """Migrate from legacy WatcherConfig."""
        return cls(
            competition_slug=competition_slug,
            watcher_enabled=getattr(watcher_config, "enabled", True),
            watcher_interval_hours=getattr(watcher_config, "interval_hours", 6.0),
            watcher_seen_kernel_refs=getattr(watcher_config, "seen_kernel_refs", []),
            watcher_last_checked_at=getattr(watcher_config, "last_checked_at", ""),
        )

    def next_watcher_check_at(self) -> datetime | None:
        """Return the next time the watcher should run, or None if not scheduled."""
        if not self.watcher_enabled or not self.watcher_last_checked_at:
            return datetime.now()  # run immediately
        try:
            last = datetime.fromisoformat(self.watcher_last_checked_at)
            return last + timedelta(hours=self.watcher_interval_hours)
        except ValueError:
            return datetime.now()

    def watcher_is_due(self) -> bool:
        nxt = self.next_watcher_check_at()
        return nxt is None or datetime.now() >= nxt

    def briefing_is_due(self) -> bool:
        """Return True if a time-based briefing should fire now."""
        if self.briefing_mode not in ("at_time", "both"):
            return False
        if not self.briefing_time:
            return False
        now = datetime.now()
        try:
            h, m = map(int, self.briefing_time.split(":"))
        except ValueError:
            return False
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        # Within 10 minutes of the target time
        if abs((now - target).total_seconds()) > 600:
            return False
        # Don't double-send: check last sent
        if self.briefing_last_sent_at:
            try:
                last = datetime.fromisoformat(self.briefing_last_sent_at)
                if (now - last).total_seconds() < 3600:  # not within the same hour
                    return False
            except ValueError:
                pass
        return True

    def experiment_should_start(self) -> bool:
        """Return True if the experiment loop should start now."""
        if self.experiment_start_mode == "now":
            return not bool(self.experiment_triggered_at)
        if self.experiment_start_mode == "at_time" and self.experiment_start_time:
            if self.experiment_triggered_at:
                return False  # already triggered
            now = datetime.now()
            try:
                h, m = map(int, self.experiment_start_time.split(":"))
            except ValueError:
                return False
            target = now.replace(hour=h, minute=m, second=0, microsecond=0)
            return now >= target
        return False  # manual mode


# ---------------------------------------------------------------------------
# Scheduler daemon — the background process
# ---------------------------------------------------------------------------


def run_scheduler_daemon(
    slug: str,
    artifacts_dir_str: str,
    ai_provider: Any = None,
) -> None:
    """Background daemon that drives the watcher + briefing schedule.

    Runs indefinitely. Launched by _start_scheduler_background() in cli.py.
    Does NOT drive the experiment loop — that's Claude Code's job.
    Does NOT make proposals — that's Claude's job.

    What it DOES do:
      - Kernel watcher: polls on schedule, injects ideas, notifies if better kernel
      - Morning briefing: sends ntfy at configured time
    """
    artifacts_dir = Path(artifacts_dir_str)
    poll_sleep = 60  # check every minute whether something is due

    while True:
        config = SchedulerConfig.load(artifacts_dir)
        if config is None or not config.enabled:
            break

        # Kernel watcher
        if config.watcher_enabled and config.watcher_is_due():
            _run_watcher_cycle(slug, artifacts_dir, config, ai_provider)

        # Morning briefing (time-based)
        if config.briefing_is_due():
            _send_morning_briefing(slug, artifacts_dir, config)

        time.sleep(poll_sleep)


# ---------------------------------------------------------------------------
# Watcher cycle
# ---------------------------------------------------------------------------


def _run_watcher_cycle(
    slug: str,
    artifacts_dir: Path,
    config: SchedulerConfig,
    ai_provider: Any,
) -> None:
    """Run one kernel-watcher poll cycle."""
    try:
        from researchforge.kaggle.api import list_competition_kernels
        from researchforge.kaggle.kernel_watcher import _extract_ideas_from_kernels, _inject_ideas
        from researchforge.kaggle.notebook_baseline import _extract_lb_score

        recent = list_competition_kernels(slug, sort_by="dateCreated", page_size=50)
        popular = list_competition_kernels(slug, sort_by="scoreDescending", page_size=20)
        all_kernels = {k.ref: k for k in recent + popular}
        seen = set(config.watcher_seen_kernel_refs)
        new_kernels = [k for k in all_kernels.values() if k.ref not in seen]

        now = datetime.now().isoformat()
        config.watcher_last_checked_at = now

        if not new_kernels:
            config.save(artifacts_dir)
            return

        # Extract ideas and inject into hypothesis pool
        new_ideas = _extract_ideas_from_kernels(new_kernels, ai_provider)
        _inject_ideas(artifacts_dir, new_ideas, new_kernels)

        config.watcher_seen_kernel_refs.extend([k.ref for k in new_kernels])
        config.save(artifacts_dir)

        # Check if any new kernel has a significantly better score
        baseline_score = _load_baseline_score(artifacts_dir)
        for k in new_kernels:
            lb = _extract_lb_score(k.title)
            if (
                lb
                and baseline_score
                and lb > baseline_score + config.watcher_notification_threshold
            ):
                _notify_better_kernel_found(slug, k.title, lb, baseline_score, config)
                break

        print(
            f"[scheduler] watcher: {len(new_kernels)} new kernels, {len(new_ideas)} ideas injected",
            flush=True,
        )
    except Exception as exc:
        print(f"[scheduler] watcher error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Morning briefing
# ---------------------------------------------------------------------------


def _send_morning_briefing(
    slug: str,
    artifacts_dir: Path,
    config: SchedulerConfig,
) -> None:
    """Build and send the morning briefing notification."""
    try:
        summary_path = artifacts_dir / "overnight_summary.json"
        resynth_path = artifacts_dir / "resynthesize_summary.json"

        if not summary_path.exists():
            return

        import json as _json

        summary = _json.loads(summary_path.read_text())
        stats = summary.get("statistics", {})
        metadata = summary.get("metadata", {})
        best = summary.get("best_result")

        rounds = stats.get("total_rounds_attempted", 0)
        improved = stats.get("rounds_improved_over_baseline", 0)
        baseline_cv = metadata.get("baseline_cv", 0)
        best_metric = best.get("metric_value") if best else None
        best_desc = best.get("patch_description", "") if best else ""

        if config.briefing_only_if_improved and improved == 0:
            return

        # Build message
        if best_metric and baseline_cv:
            delta = best_metric - baseline_cv
            pct = abs(delta / baseline_cv) * 100 if baseline_cv else 0
            direction = "↑" if delta > 0 else "↓"
            result_line = f"Best: {best_metric:.4f} ({direction}{pct:.1f}%) — {best_desc[:40]}"
        else:
            result_line = "No improvement found."

        learned = []
        if resynth_path.exists():
            resynth = _json.loads(resynth_path.read_text())
            learned = resynth.get("what_we_learned", [])[:2]

        message_lines = [
            f"Competition: {slug}",
            f"Tried: {rounds}, improved: {improved}",
            result_line,
        ]
        if learned:
            message_lines.append(f"Learned: {learned[0]}")

        from researchforge.notifications.service import Notification, send_notification

        sent = send_notification(
            Notification(
                title=f"ResearchForge morning report — {slug}",
                message=" · ".join(message_lines),
                tags=["sun"] if improved > 0 else ["moon"],
            )
        )

        if sent:
            config.briefing_last_sent_at = datetime.now().isoformat()
            config.save(artifacts_dir)
            print("[scheduler] morning briefing sent", flush=True)

    except Exception as exc:
        print(f"[scheduler] briefing error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_baseline_score(artifacts_dir: Path) -> float | None:
    setup_path = artifacts_dir / "overnight_setup.json"
    if not setup_path.exists():
        return None
    try:
        import json as _json

        data = _json.loads(setup_path.read_text())
        br = data.get("baseline_result") or {}
        return br.get("cv_score")
    except Exception:
        return None


def _notify_better_kernel_found(
    slug: str, title: str, lb: float, baseline: float, config: SchedulerConfig
) -> None:
    try:
        from researchforge.notifications.service import Notification, send_notification

        delta = lb - baseline
        send_notification(
            Notification(
                title=f"ResearchForge: better kernel found — {slug}",
                message=f"{title} (LB {lb:.4f}, +{delta:.4f} vs your baseline). Idea injected.",
                tags=["bulb"],
                priority="high",
            )
        )
    except Exception:
        pass
