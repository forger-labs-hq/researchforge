"""vast.ai (REST) and AWS EC2 (aws CLI) cloud GPU providers — request shapes and errors."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from urllib.parse import unquote

import pytest

from researchforge.kaggle import cloud_gpu_config as cg


def test_vast_rents_cheapest_matching_offer_and_destroys(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, Any]] = []

    def fake(method: str, path: str, api_key: str, body: dict[str, Any] | None = None) -> Any:
        calls.append((method, path, body))
        if path.startswith("/bundles/"):
            return {"offers": [{"id": 111, "dph_total": 0.31}, {"id": 222, "dph_total": 0.5}]}
        if path.startswith("/asks/"):
            return {"success": True, "new_contract": 9001}
        if path == "/instances/9001/" and method == "GET":
            return {
                "instances": {
                    "actual_status": "running",
                    "ssh_host": "ssh4.vast.ai",
                    "ssh_port": 2222,
                }
            }
        return {}

    monkeypatch.setattr(cg, "_vast_req", fake)
    monkeypatch.setenv("VAST_API_KEY", "k")
    cfg = cg.CloudGPUConfig(provider="vast", gpu_type="RTX 4090", max_price_per_hr=0.4)
    host, port, pod = cg.provision_and_wait(cfg)
    assert (host, port, pod) == ("ssh4.vast.ai", 2222, "9001")
    query = json.loads(unquote(calls[0][1].split("q=", 1)[1]))
    assert query["gpu_name"] == {"eq": "RTX 4090"} and query["dph_total"] == {"lte": 0.4}
    assert calls[1][0] == "PUT" and calls[1][1] == "/asks/111/"
    cg.terminate_instance(cfg, pod)
    assert ("DELETE", "/instances/9001/", None) in calls


def test_vast_without_offers_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cg, "_vast_req", lambda *a, **k: {"offers": []})
    monkeypatch.setenv("VAST_API_KEY", "k")
    with pytest.raises(RuntimeError, match="No rentable vast.ai offer"):
        cg.provision_vast(cg.CloudGPUConfig(provider="vast", max_price_per_hr=0.1))


def test_aws_requires_ami_key_and_security_group() -> None:
    with pytest.raises(RuntimeError, match="--ssh-key-name"):
        cg.provision_aws(cg.CloudGPUConfig(provider="aws", base_image="ami-1"))


def test_aws_launch_wait_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        out: dict[str, Any] = {}
        if "run-instances" in cmd:
            out = {"Instances": [{"InstanceId": "i-abc"}]}
        elif "describe-instances" in cmd:
            out = {"Reservations": [{"Instances": [{"PublicIpAddress": "3.3.3.3"}]}]}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/aws")
    cfg = cg.CloudGPUConfig(
        provider="aws",
        base_image="ami-1",
        ssh_key_name="my-key",
        security_group_id="sg-1",
        region="us-east-1",
    )
    assert cg.provision_and_wait(cfg) == ("3.3.3.3", 22, "i-abc")
    launch = next(c for c in seen if "run-instances" in c)
    assert launch[launch.index("--instance-type") + 1] == "g5.xlarge"
    assert launch[-2:] == ["--region", "us-east-1"]
    cg.terminate_instance(cfg, "i-abc")
    assert any("terminate-instances" in c for c in seen)
