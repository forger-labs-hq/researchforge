"""Choosing where experiments run: every name works, and a bad choice stops setup."""

from __future__ import annotations

from pathlib import Path

import pytest

from researchforge.kaggle import gpu_detector, orchestrator
from researchforge.kaggle.cloud_gpu_config import CloudGPUConfig
from researchforge.kaggle.facts_gatherer import CompetitionFacts, _check_remote
from researchforge.kaggle.gpu_detector import GPUBackend, GPUDetection, resolve_compute_target
from researchforge.kaggle.orchestrator import OvernightError, OvernightOrchestrator
from researchforge.kaggle.ssh_config import SSHConfig


def _orchestrator(d: Path) -> OvernightOrchestrator:
    orch = OvernightOrchestrator.__new__(OvernightOrchestrator)
    orch.slug = "comp"
    orch.artifacts_dir = d
    orch.data_dir = d / "data"
    return orch


def _detection() -> GPUDetection:
    backends = [GPUBackend.KAGGLE_GPU, GPUBackend.COLAB_GPU, GPUBackend.CPU]
    return GPUDetection("comp", backends, "", recommended=GPUBackend.KAGGLE_GPU)


def test_gather_facts_ids_and_flag_names_mean_the_same_backend() -> None:
    assert resolve_compute_target("local") == GPUBackend.CPU
    assert resolve_compute_target("kaggle_gpu") == resolve_compute_target("Kaggle")
    assert resolve_compute_target("competition") == GPUBackend.COMPETITION_GPU
    assert resolve_compute_target("cloud_gpu") == GPUBackend.CLOUD_GPU
    assert resolve_compute_target("a100") is None


def test_setup_refuses_an_unknown_target_before_doing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _too_far() -> bool:
        raise AssertionError("setup must stop before checking credentials")

    monkeypatch.setattr(orchestrator, "validate_credentials", _too_far)
    with pytest.raises(OvernightError, match="Unknown compute target 'a100'"):
        _orchestrator(tmp_path).setup(gpu_preference="a100")


def test_remote_targets_need_their_config_first(tmp_path: Path) -> None:
    orch = _orchestrator(tmp_path)
    with pytest.raises(OvernightError, match="ssh-setup comp"):
        orch.forced_compute("ssh")
    with pytest.raises(OvernightError, match="cloud-gpu-setup comp"):
        orch.forced_compute("cloud")

    SSHConfig(host="me@box").save(tmp_path)
    CloudGPUConfig(provider="vast", gpu_type="RTX 4090").save(tmp_path)
    assert orch.forced_compute("ssh") == GPUBackend.SSH_SERVER
    assert orch.forced_compute("cloud_gpu") == GPUBackend.CLOUD_GPU
    assert orch.forced_compute("auto") is None and orch.forced_compute(None) is None


def test_setup_recommends_what_gather_facts_recommended(tmp_path: Path) -> None:
    detection = _detection()
    _orchestrator(tmp_path)._prefer_facts_recommendation(detection)
    assert detection.recommended == GPUBackend.KAGGLE_GPU  # no facts: unchanged

    CompetitionFacts(slug="comp", recommended_compute="local").save(tmp_path)
    _orchestrator(tmp_path)._prefer_facts_recommendation(detection)
    assert detection.recommended == GPUBackend.CPU


def test_interactive_menu_lists_every_target_recommendation_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    detection = _detection()
    detection.recommended = GPUBackend.CPU
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "5")
    picked = gpu_detector.ask_user_gpu_preference(detection)
    out = capsys.readouterr().out
    menu = [ln.strip() for ln in out.splitlines() if ") " in ln]
    assert menu[0].startswith("1) This Machine") and menu[0].endswith("(recommended)")
    assert [ln.split(" — ")[0] for ln in menu[3:]] == ["4) Ssh Server", "5) Cloud Gpu"]
    assert picked == GPUBackend.CLOUD_GPU


def test_facts_report_configured_remote_targets(tmp_path: Path) -> None:
    ssh, cloud = _check_remote(tmp_path)
    assert ssh.viable == "unknown" and "ssh-setup" in ssh.reason
    assert cloud.viable == "unknown" and "--list-gpus" in cloud.reason

    SSHConfig(host="me@box").save(tmp_path)
    CloudGPUConfig(provider="vast", gpu_type="RTX 4090", max_price_per_hr=0.5).save(tmp_path)
    ssh, cloud = _check_remote(tmp_path)
    assert ssh.viable is True and "me@box" in ssh.reason
    assert cloud.viable is True and cloud.reason == "Configured: vast, RTX 4090, up to $0.5/h."


def test_gpu_on_a_set_up_project_switches_compute_or_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from typer.testing import CliRunner

    from researchforge.cli import app

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("RESEARCHFORGE_OVERNIGHT_DIR", raising=False)
    monkeypatch.setenv("RESEARCHFORGE_NO_DASHBOARD", "1")
    monkeypatch.chdir(tmp_path)
    d = tmp_path / ".overnight_comp"
    d.mkdir()
    setup = d / "overnight_setup.json"
    setup.write_text(json.dumps({"competition_slug": "comp", "selected_gpu": "kaggle_gpu"}))

    refused = CliRunner().invoke(app, ["kaggle", "overnight", "comp", "--gpu", "cloud"])
    assert refused.exit_code == 1 and "cloud-gpu-setup comp" in refused.output
    assert json.loads(setup.read_text())["selected_gpu"] == "kaggle_gpu"

    switched = CliRunner().invoke(app, ["kaggle", "overnight", "comp", "--gpu", "local"])
    assert switched.exit_code == 0, switched.output
    assert "kaggle_gpu → cpu" in switched.output and "Resuming comp" in switched.output
    assert json.loads(setup.read_text())["selected_gpu"] == "cpu"
