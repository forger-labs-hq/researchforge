"""Repo auto-run pieces that need no project: settings, approval, the driver wrapper."""

from pathlib import Path
from typing import Any

import pytest

from researchforge.autorun import unattended as ua


def test_defaults_are_off_and_ask(tmp_path: Path) -> None:
    cfg = ua.RepoAutoRunConfig.load(tmp_path)
    assert cfg.enabled is False
    assert cfg.approval == "ask"
    assert not cfg.standing


def test_settings_are_validated_and_clamped(tmp_path: Path) -> None:
    cfg = ua.update_settings(tmp_path, {"rounds_per_launch": 99, "enabled": "on"})
    assert cfg.rounds_per_launch == 20
    assert cfg.enabled is True
    with pytest.raises(ValueError, match="driver must be"):
        ua.update_settings(tmp_path, {"driver": "gpt"})
    with pytest.raises(ValueError, match="unknown setting"):
        ua.update_settings(tmp_path, {"approval": "standing"})  # only set_approval may
    with pytest.raises(ValueError, match="HH:MM"):
        ua.update_settings(tmp_path, {"briefing_at": "7am"})


def test_switching_driver_drops_the_old_model(tmp_path: Path) -> None:
    ua.update_settings(tmp_path, {"driver": "claude", "model": "opus"})
    assert ua.update_settings(tmp_path, {"driver": "cursor"}).model == ""


def test_standing_approval_needs_the_typed_word(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='type "approve"'):
        ua.set_approval(tmp_path, "standing", typed="yes")
    assert ua.set_approval(tmp_path, "standing", typed=" Approve ").standing
    cfg = ua.set_approval(tmp_path, "ask")
    assert not cfg.standing
    assert cfg.standing_approved_at == ""


def test_a_folder_without_a_project_is_not_launched(tmp_path: Path) -> None:
    reasons = ua.gates(tmp_path, ua.RepoAutoRunConfig.load(tmp_path))
    assert reasons == ["not a ResearchForge project — run researchforge init here first"]


class _Failing:
    name = "claude-cli/default"

    def __init__(self, error: str) -> None:
        self.error, self.calls = error, 0

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        self.calls += 1
        raise RuntimeError(self.error)


def test_a_usage_limit_is_noted_and_later_calls_skip_the_driver() -> None:
    inner = _Failing("claude exited 1: You've hit your usage limit · resets 5pm")
    watch = ua._LimitWatch(inner)
    with pytest.raises(ua.UsageLimitHit):
        watch.generate("s", "u")
    assert watch.hit is not None and watch.hit[1].hour == 17
    with pytest.raises(ua.UsageLimitHit):
        watch.generate("s", "u")
    assert inner.calls == 1


def test_other_errors_pass_through() -> None:
    watch = ua._LimitWatch(_Failing("network down"))
    with pytest.raises(RuntimeError, match="network down"):
        watch.generate("s", "u")
    assert watch.hit is None


def test_alerts_use_the_project_topic_and_folder_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle import control
    from researchforge.notifications import service

    root = tmp_path / "my repo"
    ua.update_settings(root, {"ntfy_topic": "rf-repo-topic"})
    assert control.notify_topic(ua.rf_dir(root)) == ("rf-repo-topic", "project")
    sent: list[Any] = []
    monkeypatch.setattr(service, "send_notification", lambda n: sent.append(n) or True)
    assert control.notify(ua.rf_dir(root), "alert", "hello")
    assert sent[0].title == "ResearchForge: my repo"


def test_service_names_are_stable_and_distinct(tmp_path: Path) -> None:
    a, b = tmp_path / "Arc AGI 3", tmp_path / "other" / "Arc AGI 3"
    assert ua.service_key(a) == ua.service_key(a)
    assert ua.service_key(a) != ua.service_key(b)
    assert ua.service_label(a).startswith("com.researchforge.repo.arc-agi-3-")


def test_dashboard_repo_auto_actions(tmp_path: Path) -> None:
    from researchforge.kaggle.dashboard import DashboardService

    service = DashboardService(tmp_path)
    project = {"root": str(tmp_path), "repo": {"auto_run": {}}, "overnight": None}
    out = service.action_repo_auto(
        project, {"op": "settings", "changes": {"enabled": True, "max_rounds_per_day": 2}}
    )
    assert out["config"]["enabled"] is True
    with pytest.raises(ValueError, match='type "approve"'):
        service.action_repo_auto(project, {"op": "approval", "mode": "standing", "typed": ""})
    out = service.action_repo_auto(
        project, {"op": "approval", "mode": "standing", "typed": "approve"}
    )
    assert out["config"]["approval"] == "standing"
    with pytest.raises(ValueError, match="not starting"):
        service.action_repo_auto(project, {"op": "start"})
    with pytest.raises(ValueError, match="repository projects"):
        service.action_repo_auto({**project, "overnight": {"x": 1}}, {"op": "start"})


def test_an_unattended_run_cannot_give_a_standing_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from researchforge.cli import app

    (tmp_path / ".researchforge").mkdir()
    (tmp_path / ".researchforge" / "researchforge.db").write_bytes(b"")
    monkeypatch.setenv("RESEARCHFORGE_AUTO_RUN", "1")
    out = CliRunner().invoke(
        app, ["auto-run", "approval", "standing", "--path", str(tmp_path)], input="approve\n"
    )
    assert out.exit_code == 1
    assert "person's decision" in out.output


def _recording_run(calls: list[list[str]]) -> Any:
    import subprocess

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return run


def test_the_launchd_service_runs_the_monitor_every_poll_minutes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ua.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    calls: list[list[str]] = []
    monkeypatch.setattr(ua.subprocess, "run", _recording_run(calls))
    root = tmp_path / "proj"
    ua.update_settings(root, {"poll_minutes": 20})

    assert "every 20 min" in ua.install_service(root)
    plist = ua._plist(root).read_text()
    assert "<key>StartInterval</key><integer>1200</integer>" in plist
    assert "<key>RunAtLoad</key><true/>" in plist
    launcher = ua._write_launcher(root).read_text()
    assert f"'auto-run', 'monitor', '--once', '--path', {str(root)!r}" in launcher
    assert calls[-1][:3] == ["launchctl", "load", "-w"]

    assert ua.uninstall_service(root).startswith("removed com.researchforge.repo.")
    assert not ua._plist(root).exists()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_other_platforms_use_the_shared_service_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    from researchforge.kaggle import service_install

    monkeypatch.setattr(ua.sys, "platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    got: list[tuple[Any, ...]] = []
    monkeypatch.setattr(service_install, "install", lambda *a: got.append(a) or (True, "installed"))
    root = tmp_path / "proj"
    ua.update_settings(root, {"poll_minutes": 15})
    assert ua.install_service(root) == "installed"
    plat, name, _python, launcher, hours, _log = got[0]
    assert (plat, name, hours) == (platform, f"repo-{ua.service_key(root)}", 0.25)
    assert launcher.name == f"repo_monitor_{ua.service_key(root)}.py"
    # The shared installer turns 0.25h into a 15-minute systemd timer / schtasks repeat.
    units = service_install.systemd_units(name, "python", launcher, hours, tmp_path / "x.log")
    assert any("OnUnitActiveSec=15min" in text for text in units.values())
    args = service_install.schtasks_create_args(name, "python", launcher, hours)
    assert args[-4:] == ["/SC", "MINUTE", "/MO", "15"]


def test_a_launch_gets_the_project_topic_and_no_competition_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", "/somewhere/.overnight_x")
    ua.update_settings(tmp_path, {"ntfy_topic": "rf-repo-topic", "driver": "byok"})
    env = ua.child_env(tmp_path, ua.RepoAutoRunConfig.load(tmp_path))
    assert env["NTFY_TOPIC"] == "rf-repo-topic"
    assert "RESEARCHFORGE_OVERNIGHT_DIR" not in env
