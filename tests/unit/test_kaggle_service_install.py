"""Monitor background service on Linux (systemd --user) and Windows (Task Scheduler)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from researchforge.kaggle import service_install


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


class _Recorder:
    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr="boom")


def test_systemd_units_repeat_at_the_interval_and_log_to_file(tmp_path: Path) -> None:
    units = service_install.systemd_units(
        "titanic", "/usr/bin/python3", tmp_path / "l.py", 6, tmp_path / "x.log"
    )
    service = units["researchforge-watcher-titanic.service"]
    timer = units["researchforge-watcher-titanic.timer"]
    assert 'ExecStart="/usr/bin/python3"' in service
    assert f"StandardOutput=append:{tmp_path / 'x.log'}" in service
    assert "OnUnitActiveSec=360min" in timer and "WantedBy=timers.target" in timer


@pytest.mark.parametrize(
    ("hours", "schedule"),
    [(6, ["/SC", "HOURLY", "/MO", "6"]), (0.5, ["/SC", "MINUTE", "/MO", "30"])],
)
def test_schtasks_schedule(hours: float, schedule: list[str], tmp_path: Path) -> None:
    args = service_install.schtasks_create_args("titanic", "C:\\py.exe", tmp_path / "l.py", hours)
    assert args[:6] == ["schtasks", "/Create", "/F", "/TN", "ResearchForge\\watcher-titanic", "/TR"]
    assert args[-4:] == schedule


def test_linux_install_writes_units_and_enables_timer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rec = _Recorder()
    monkeypatch.setattr(service_install.subprocess, "run", rec)
    ok, detail = service_install.install(
        "linux", "titanic", "/usr/bin/python3", tmp_path / "l.py", 6, tmp_path / "x.log"
    )
    assert ok, detail
    unit_dir = service_install.systemd_dir()
    assert (unit_dir / "researchforge-watcher-titanic.timer").exists()
    assert ["systemctl", "--user", "enable", "--now", "researchforge-watcher-titanic.timer"] in (
        rec.calls
    )

    removed, _ = service_install.uninstall("linux", "titanic")
    assert removed and not (unit_dir / "researchforge-watcher-titanic.timer").exists()


def test_install_reports_failures_instead_of_pretending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(service_install.subprocess, "run", _Recorder(returncode=1))
    ok, detail = service_install.install(
        "win32", "titanic", "C:\\py.exe", tmp_path / "l.py", 6, tmp_path / "x.log"
    )
    assert not ok and "schtasks failed" in detail
    ok, detail = service_install.install(
        "sunos5", "titanic", "/py", tmp_path / "l.py", 6, tmp_path / "x.log"
    )
    assert not ok and "not supported" in detail
