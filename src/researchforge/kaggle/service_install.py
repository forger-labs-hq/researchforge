"""Background-service install for the monitor on Linux (systemd --user) and Windows
(Task Scheduler). macOS uses launchd, in ``kaggle scheduler-install`` itself.

The generated files and commands are pure functions so they can be tested on any OS;
``install`` / ``uninstall`` run them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def unit_name(slug: str) -> str:
    return f"researchforge-watcher-{slug}"


def task_name(slug: str) -> str:
    return f"ResearchForge\\watcher-{slug}"


def systemd_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def systemd_units(
    slug: str, python_exe: str, launcher: Path, hours: float, log_path: Path
) -> dict[str, str]:
    """A oneshot service plus a timer that fires it every ``hours`` (and once at boot)."""
    minutes = max(1, int(hours * 60))
    service = f"""[Unit]
Description=ResearchForge monitor for {slug}

[Service]
Type=oneshot
ExecStart="{python_exe}" "{launcher}"
StandardOutput=append:{log_path}
StandardError=append:{log_path}
"""
    timer = f"""[Unit]
Description=Run the ResearchForge monitor for {slug} every {hours}h

[Timer]
OnBootSec=2min
OnUnitActiveSec={minutes}min
Persistent=true

[Install]
WantedBy=timers.target
"""
    name = unit_name(slug)
    return {f"{name}.service": service, f"{name}.timer": timer}


def schtasks_create_args(slug: str, python_exe: str, launcher: Path, hours: float) -> list[str]:
    """`schtasks` arguments for a task that repeats every ``hours`` for the current user."""
    minutes = max(1, int(hours * 60))
    if minutes % 60 == 0 and minutes // 60 <= 23:
        schedule = ["/SC", "HOURLY", "/MO", str(minutes // 60)]
    else:
        schedule = ["/SC", "MINUTE", "/MO", str(min(minutes, 1439))]
    return [
        "schtasks",
        "/Create",
        "/F",
        "/TN",
        task_name(slug),
        "/TR",
        f'"{python_exe}" "{launcher}"',
        *schedule,
    ]


def install(
    platform: str, slug: str, python_exe: str, launcher: Path, hours: float, log_path: Path
) -> tuple[bool, str]:
    """Install the service. Returns (ok, human-readable detail)."""
    if platform.startswith("linux"):
        target = systemd_dir()
        target.mkdir(parents=True, exist_ok=True)
        for filename, text in systemd_units(slug, python_exe, launcher, hours, log_path).items():
            (target / filename).write_text(text, encoding="utf-8")
        for cmd in (
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "--now", f"{unit_name(slug)}.timer"],
        ):
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                return False, f"{' '.join(cmd)} failed: {res.stderr.strip()}"
        return True, f"systemd user timer {unit_name(slug)}.timer (units in {target})"
    if platform.startswith("win"):
        res = subprocess.run(
            schtasks_create_args(slug, python_exe, launcher, hours),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return False, f"schtasks failed: {res.stderr.strip() or res.stdout.strip()}"
        return True, f"Task Scheduler task {task_name(slug)}"
    return False, f"background services are not supported on {platform}"


def uninstall(platform: str, slug: str) -> tuple[bool, str]:
    """Remove the service. Returns (removed, detail)."""
    if platform.startswith("linux"):
        name = unit_name(slug)
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", f"{name}.timer"],
            capture_output=True,
            check=False,
        )
        removed = False
        for suffix in (".service", ".timer"):
            path = systemd_dir() / f"{name}{suffix}"
            if path.exists():
                path.unlink()
                removed = True
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, check=False)
        return removed, f"systemd user timer {name}.timer"
    if platform.startswith("win"):
        res = subprocess.run(
            ["schtasks", "/Delete", "/F", "/TN", task_name(slug)],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.returncode == 0, f"Task Scheduler task {task_name(slug)}"
    return False, f"background services are not supported on {platform}"
