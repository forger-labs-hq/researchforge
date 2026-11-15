"""Notification service.

Sends push notifications via ntfy (primary) or Telegram (optional).

Configuration (in priority order):
  1. NTFY_TOPIC env var  — set in shell or .env
  2. ~/.config/researchforge/notify.json  — persisted by `researchforge notify setup`

With no topic set, nothing is sent: a shared default topic would let anyone read the alerts.

Setup (one-time, survives across Claude Code sessions):
  researchforge notify setup --topic <your-ntfy-topic>
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Channel(StrEnum):
    NTFY = "ntfy"
    TELEGRAM = "telegram"


@dataclass
class Notification:
    title: str
    message: str
    channel: Channel = Channel.NTFY
    priority: str = "default"  # low / default / high / urgent  (ntfy levels)
    tags: list[str] = field(default_factory=list)  # ntfy emoji tags e.g. ["white_check_mark"]

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []


# ---------------------------------------------------------------------------
_NOTIFICATION_LOG = Path.home() / ".config" / "researchforge" / "notification_log.jsonl"


def _append_notification_log(notif: Notification, sent: bool) -> None:
    """Append one line to the notification log for the Timeline view."""
    try:
        import json as _json

        _NOTIFICATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "title": notif.title,
            "message": notif.message,
            "tags": notif.tags or [],
            "priority": notif.priority,
            "sent": sent,
        }
        with _NOTIFICATION_LOG.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry) + "\n")
    except Exception:
        pass  # never crash the main flow


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def send_notification(notif: Notification) -> bool:
    """Send a notification. Returns True if sent successfully."""
    if notif.channel == Channel.NTFY:
        sent = _send_ntfy(notif)
    elif notif.channel == Channel.TELEGRAM:
        sent = _send_telegram(notif)
    else:
        sent = False
    _append_notification_log(notif, sent)
    return sent


def notify_new_hypotheses(
    count: int,
    hypothesis_ids: list[str],
    project_name: str = "",
    topic: str = "",
) -> bool:
    """Convenience: notify when new hypotheses are imported or generated.

    Args:
        count: Number of new hypotheses added.
        hypothesis_ids: List of new hypothesis IDs (e.g. ["hyp-004", "hyp-005"]).
        project_name: Optional project/competition name for context.
        topic: Override NTFY_TOPIC for this notification.
    """
    if count == 0:
        return False
    ids_str = ", ".join(hypothesis_ids[:5])
    if len(hypothesis_ids) > 5:
        ids_str += f" … (+{len(hypothesis_ids) - 5} more)"
    context = f" for {project_name}" if project_name else ""
    msg = f"{count} new hypothesis(es) ready{context}: {ids_str}"

    old = os.environ.get("NTFY_TOPIC")
    if topic:
        os.environ["NTFY_TOPIC"] = topic
    try:
        return send_notification(
            Notification(
                title=f"ResearchForge: {count} new hypothesis(es)",
                message=msg,
                tags=["bulb"],
                priority="default",
            )
        )
    finally:
        if topic:
            if old is None:
                os.environ.pop("NTFY_TOPIC", None)
            else:
                os.environ["NTFY_TOPIC"] = old


def notify_kernel_found(
    slug: str,
    title: str,
    lb_score: float | None,
    delta: float | None,
    topic: str = "",
) -> bool:
    """Convenience: notify when a better public kernel is found for a competition."""
    score_str = f"LB {lb_score:.4f}" if lb_score is not None else "score unknown"
    delta_str = f", {delta:.4f} ahead of our best LB" if delta is not None else ""
    msg = (
        f"Public notebook for {slug}: {title} (claims {score_str}{delta_str}). "
        "It is on the dashboard's Public notebooks board; its ideas are in the pool."
    )

    old = os.environ.get("NTFY_TOPIC")
    if topic:
        os.environ["NTFY_TOPIC"] = topic
    try:
        return send_notification(
            Notification(
                title=f"ResearchForge: better kernel found ({slug})",
                message=msg,
                tags=["bulb"],
            )
        )
    finally:
        if topic:
            if old is None:
                os.environ.pop("NTFY_TOPIC", None)
            else:
                os.environ["NTFY_TOPIC"] = old


def notify_submission_approved(
    experiment_id: str,
    title: str,
    metric_name: str,
    delta_str: str,
    topic: str = "",
) -> bool:
    """Convenience: notify when an experiment is approved for Kaggle submission."""
    msg = f"{title[:60]} · {metric_name} {delta_str} · Ready to submit"

    old = os.environ.get("NTFY_TOPIC")
    if topic:
        os.environ["NTFY_TOPIC"] = topic
    try:
        return send_notification(
            Notification(
                title=f"RF submission approved: {experiment_id}",
                message=msg,
                tags=["rocket"],
                priority="high",
            )
        )
    finally:
        if topic:
            if old is None:
                os.environ.pop("NTFY_TOPIC", None)
            else:
                os.environ["NTFY_TOPIC"] = old


def notify_loop_complete(
    competition_slug: str,
    rounds_attempted: int,
    rounds_improved: int,
    best_metric: float | None,
    baseline_metric: float | None,
    submissions_left: int | None = None,
) -> bool:
    """Convenience: send the standard loop-finished notification."""
    improvement_str = ""
    if best_metric is not None and baseline_metric not in (None, 0):
        delta = best_metric - baseline_metric
        pct = abs(delta / baseline_metric) * 100
        direction = "↑" if delta > 0 else "↓"
        improvement_str = f" ({direction}{pct:.2f}%)"

    lines = [
        f"Competition: {competition_slug}",
        f"Tried: {rounds_attempted}, improved: {rounds_improved}",
        f"Best metric: {best_metric:.5f}{improvement_str}"
        if best_metric is not None
        else "No improvement found",
    ]
    if submissions_left is not None:
        lines.append(f"Submissions left today: {submissions_left}")

    msg = " · ".join(lines)
    title = f"ResearchForge: {competition_slug} done"

    return send_notification(
        Notification(
            title=title,
            message=msg,
            tags=["white_check_mark"] if rounds_improved > 0 else ["x"],
        )
    )


def notify_error(competition_slug: str, error: str) -> bool:
    """Convenience: send an error notification."""
    return send_notification(
        Notification(
            title=f"ResearchForge: {competition_slug} error",
            message=error[:500],
            priority="high",
            tags=["warning"],
        )
    )


# ---------------------------------------------------------------------------
# ntfy backend
# ---------------------------------------------------------------------------

_NOTIFY_CONFIG_PATH = Path.home() / ".config" / "researchforge" / "notify.json"


def _load_notify_config() -> dict[str, Any]:
    """Read ~/.config/researchforge/notify.json — returns {} on any error."""
    try:
        if _NOTIFY_CONFIG_PATH.exists():
            import json as _json

            data: dict[str, Any] = _json.loads(_NOTIFY_CONFIG_PATH.read_text(encoding="utf-8"))
            return data
    except Exception:
        pass
    return {}


def save_notify_config(topic: str, server: str = "", *, clear_topic: bool = False) -> None:
    """Persist ntfy settings to ~/.config/researchforge/notify.json.

    Called by `researchforge notify setup` and by overnight setup when NTFY_TOPIC
    is present, so the topic survives across processes and Claude Code subshells.
    """
    import json as _json

    _NOTIFY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_notify_config()
    if topic:
        cfg["ntfy_topic"] = topic
    elif clear_topic:
        cfg.pop("ntfy_topic", None)
    if server:
        cfg["ntfy_server"] = server
    _NOTIFY_CONFIG_PATH.write_text(_json.dumps(cfg, indent=2), encoding="utf-8")


def _ntfy_topic() -> str:
    """Resolve the ntfy topic, or "" when none is set (then nothing is sent).

    Priority (highest first):
      1. NTFY_TOPIC env var  (set per-session or in .env)
      2. ~/.config/researchforge/notify.json → ntfy_topic
    """
    env = os.environ.get("NTFY_TOPIC", "").strip()
    if env:
        return env
    cfg = _load_notify_config()
    stored = cfg.get("ntfy_topic", "").strip()
    if stored:
        return str(stored)
    return ""


def _ntfy_server() -> str:
    """Resolve the ntfy server URL.

    Priority: NTFY_SERVER env → config file → https://ntfy.sh
    """
    env = os.environ.get("NTFY_SERVER", "").strip()
    if env:
        return env.rstrip("/")
    cfg = _load_notify_config()
    stored = cfg.get("ntfy_server", "").strip()
    if stored:
        return str(stored).rstrip("/")
    return "https://ntfy.sh"


def _send_ntfy(notif: Notification) -> bool:
    """POST to ntfy.sh using the JSON body API (UTF-8 safe, no header-encoding limits).

    The header-based API encodes Title/Tags as latin-1, which crashes on any
    non-ASCII character (emojis, check marks, etc.).  The JSON body API sends
    everything as UTF-8 and is ntfy's recommended approach for rich messages.
    """
    import json as _json

    topic = _ntfy_topic()
    if not topic:
        return False
    url = _ntfy_server()  # POST to server root; topic goes in the body

    # ntfy priority mapping: low=1, default=3, high=4, urgent=5
    _priority_map = {"low": 1, "default": 3, "high": 4, "urgent": 5}
    priority_int = _priority_map.get(notif.priority, 3)

    payload: dict[str, Any] = {
        "topic": topic,
        "message": notif.message,
        "priority": priority_int,
    }
    if notif.title:
        payload["title"] = notif.title
    if notif.tags:
        payload["tags"] = notif.tags

    body = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Build SSL context with certifi CA bundle (required on macOS)
    context = None
    try:
        import ssl

        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    try:
        with urllib.request.urlopen(req, timeout=10, context=context) as resp:
            return bool(resp.status < 300)
    except (urllib.error.URLError, OSError):
        return False


# ---------------------------------------------------------------------------
# Telegram backend
# ---------------------------------------------------------------------------


def _send_telegram(notif: Notification) -> bool:
    """POST to Telegram Bot API."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    text = f"*{notif.title}*\n{notif.message}"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return bool(resp.status < 300)
    except (urllib.error.URLError, OSError):
        return False


import json  # noqa: E402 — imported here to avoid circular issue with module top
