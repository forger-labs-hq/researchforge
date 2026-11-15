"""Sensor contract shared by the monitor (producer) and the dashboard (consumer).

Config — what is *enabled* — lives in ``scheduler_config.json``
(:class:`~researchforge.kaggle.scheduler.SchedulerConfig`):

* ``disabled_sensors``: built-in sensor names the monitor must skip
* ``enabled_sensors``: built-ins that are off by default (``DEFAULT_OFF``) and
  were turned on
* ``custom_sensors``: user-added ``{name, command, timeout_seconds, enabled}``;
  the monitor runs ``command`` in the project root each cycle and records the
  exit status

Runtime state — what *happened* — lives in ``monitor_state.json`` under
``sensors.<name> = {last_run_at, status, detail, error}`` plus cycle metadata
(``last_cycle_at``, ``next_cycle_at``, ``monitor_pid``, ``poll_hours``).

The dashboard renders exactly these two files. A sensor with no recorded run
is shown as "never run", never as healthy.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, read_json

BUILTIN_SENSORS: list[dict[str, str]] = [
    {
        "name": "kernel_watcher_sensor",
        "kind": "Kernel sensor",
        "description": (
            "Every hour: new and re-run public notebooks → their ideas into the pool, the "
            "notebook board ranked by claimed LB; alerts when one claims more than our best LB"
        ),
    },
    {
        "name": "submission_status_sensor",
        "kind": "Run status sensor",
        "description": "Checks submissions every 15 minutes; notifies as soon as one is scored",
    },
    {
        "name": "hypothesis_pool_sensor",
        "kind": "Pool sensor",
        "description": "Notifies when the pending hypothesis pool grows by 3 or more",
    },
    {
        "name": "experiment_heartbeat_sensor",
        "kind": "Heartbeat sensor",
        "description": "Alerts when execute_round.lock is older than the stale threshold",
    },
    {
        "name": "arxiv_scanner_sensor",
        "kind": "Daily sensor",
        "description": "Scans arXiv for related papers and injects technique ideas",
    },
    {
        "name": "morning_briefing_sensor",
        "kind": "Schedule",
        "description": "Sends the daily summary notification at the briefing time",
    },
    {
        "name": "auto_run_sensor",
        "kind": "Auto-run",
        "description": (
            "Runs experiment rounds on untested ideas with Claude Code, Cursor or an API key "
            "(off until you turn it on; never submits)"
        ),
    },
]
BUILTIN_NAMES = {s["name"] for s in BUILTIN_SENSORS}
# Built-ins that do something costly on their own stay off until turned on.
DEFAULT_OFF = {"auto_run_sensor"}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,63}$")


def load_config(artifacts_dir: Path, slug: str) -> Any:
    """Load (or create in memory) the SchedulerConfig, migrating legacy toggles."""
    from researchforge.kaggle.scheduler import SchedulerConfig

    try:
        cfg = SchedulerConfig.load(artifacts_dir)
    except (OSError, ValueError, TypeError):
        cfg = None
    if cfg is None:
        cfg = SchedulerConfig(competition_slug=slug)
        legacy = read_json(artifacts_dir / "monitor_state.json", {})
        if isinstance(legacy, dict):
            cfg.disabled_sensors = [
                s for s in legacy.get("disabled_sensors", []) if s in BUILTIN_NAMES
            ]
    return cfg


def is_enabled(cfg: Any, name: str) -> bool:
    if name in DEFAULT_OFF:
        return name in getattr(cfg, "enabled_sensors", [])
    if name in BUILTIN_NAMES:
        return name not in cfg.disabled_sensors
    return any(c.get("name") == name and c.get("enabled", True) for c in cfg.custom_sensors)


def set_enabled(artifacts_dir: Path, slug: str, name: str, enabled: bool) -> None:
    cfg = load_config(artifacts_dir, slug)
    if name in DEFAULT_OFF:
        on = [s for s in cfg.enabled_sensors if s != name]
        cfg.enabled_sensors = [*on, name] if enabled else on
    elif name in BUILTIN_NAMES:
        disabled = [s for s in cfg.disabled_sensors if s != name]
        if not enabled:
            disabled.append(name)
        cfg.disabled_sensors = disabled
    else:
        for c in cfg.custom_sensors:
            if c.get("name") == name:
                c["enabled"] = enabled
                break
        else:
            raise KeyError(f"unknown sensor: {name}")
    cfg.save(artifacts_dir)


def add_custom(
    artifacts_dir: Path, slug: str, name: str, command: str, timeout_seconds: int = 120
) -> None:
    if not _NAME_RE.match(name):
        raise ValueError("sensor name must be 2-64 chars of a-z, 0-9, '_' or '-'")
    if name in BUILTIN_NAMES:
        raise ValueError(f"{name} is a built-in sensor")
    if not command.strip():
        raise ValueError("command is required")
    cfg = load_config(artifacts_dir, slug)
    cfg.custom_sensors = [c for c in cfg.custom_sensors if c.get("name") != name]
    cfg.custom_sensors.append(
        {
            "name": name,
            "command": command,
            "enabled": True,
            "timeout_seconds": max(5, min(int(timeout_seconds), 3600)),
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    cfg.save(artifacts_dir)


def remove_custom(artifacts_dir: Path, slug: str, name: str) -> bool:
    cfg = load_config(artifacts_dir, slug)
    before = len(cfg.custom_sensors)
    cfg.custom_sensors = [c for c in cfg.custom_sensors if c.get("name") != name]
    if len(cfg.custom_sensors) == before:
        return False
    cfg.save(artifacts_dir)
    return True


# ---------------------------------------------------------------------------
# Runtime state (monitor_state.json)
# ---------------------------------------------------------------------------


def record(
    state: dict[str, Any], name: str, status: str, detail: str = "", error: str = ""
) -> None:
    """Record one sensor evaluation. status: ok | error | disabled | skipped."""
    sensors = state.setdefault("sensors", {})
    prev = sensors.get(name, {})
    entry = {
        "last_run_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "detail": detail[:300],
        "error": error[:500],
    }
    if status == "error":
        entry["last_error_at"] = entry["last_run_at"]
    elif prev.get("last_error_at"):
        entry["last_error_at"] = prev["last_error_at"]
    if status == "ok":
        entry["last_ok_at"] = entry["last_run_at"]
    elif prev.get("last_ok_at"):
        entry["last_ok_at"] = prev["last_ok_at"]
    sensors[name] = entry


def save_state(artifacts_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(artifacts_dir / "monitor_state.json", state)


def run_custom(spec: dict[str, Any], cwd: Path) -> tuple[str, str, str]:
    """Run one custom sensor command. Returns (status, detail, error)."""
    try:
        proc = subprocess.run(
            spec["command"],
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=int(spec.get("timeout_seconds", 120)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "error", "", f"timed out after {spec.get('timeout_seconds', 120)}s"
    except OSError as exc:
        return "error", "", str(exc)
    out = (proc.stdout or "").strip().splitlines()
    err = (proc.stderr or "").strip().splitlines()
    if proc.returncode == 0:
        return "ok", out[-1] if out else "exit 0", ""
    return "error", out[-1] if out else "", f"exit {proc.returncode}: {err[-1] if err else ''}"
