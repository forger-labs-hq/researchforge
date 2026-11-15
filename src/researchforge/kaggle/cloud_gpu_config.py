"""Cloud GPU provider configuration for overnight experiments.

Supports RunPod, Lambda Labs, vast.ai and AWS EC2. The lifecycle for each experiment is:
  provision → wait-for-ssh → rsync-code → run → rsync-results → terminate

Setup:
    export RUNPOD_API_KEY=your_key
    researchforge kaggle cloud-gpu-setup <slug> --provider runpod

The config is saved to .overnight_<slug>/cloud_gpu_config.json.
On each execute-round, a fresh pod is provisioned, used, and terminated.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class CloudGPUConfig:
    """Provider-agnostic cloud GPU configuration."""

    provider: str = "runpod"  # runpod | lambda | vast | aws
    api_key_env: str = ""  # env var that holds the API key, e.g. RUNPOD_API_KEY
    gpu_type: str = ""  # e.g. "NVIDIA RTX A4000" (RunPod) or "gpu_1x_a10" (Lambda)
    base_image: str = ""  # Docker image (RunPod only); Lambda uses fixed AMI
    region: str = ""  # Lambda region, e.g. "us-west-2"
    python_path: str = "python"  # Python executable inside the instance
    rsync_data: bool = False  # Copy competition data to the instance
    max_price_per_hr: float = 0.0  # 0 = no limit
    ssh_key_name: str = ""  # AWS: EC2 key pair name
    security_group_id: str = ""  # AWS: a group that allows inbound SSH

    # ------------------------------------------------------------------ #

    def save(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / "cloud_gpu_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, artifacts_dir: Path) -> CloudGPUConfig | None:
        path = artifacts_dir / "cloud_gpu_config.json"
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            return None

    def api_key(self) -> str:
        """Resolve the API key: the environment, then the auto-run env file.

        The env file is where the dashboard's Keys page saves it, and the only place a
        launchd monitor or an auto-run launch can find it.
        """
        from researchforge.kaggle.auto_run import load_env_file

        stored = load_env_file()[0]
        names = [self.api_key_env] if self.api_key_env else []
        names += {
            "runpod": ["RUNPOD_API_KEY"],
            "lambda": ["LAMBDA_API_KEY"],
            "vast": ["VAST_API_KEY"],
        }.get(self.provider, [])
        key = next((v for v in (os.environ.get(n) or stored.get(n, "") for n in names) if v), "")
        if not key and self.provider == "aws":
            # AWS uses its own credential chain through the aws CLI; no key to pass.
            import shutil

            key = "aws-cli" if shutil.which("aws") else ""
        return key

    def default_gpu_type(self) -> str:
        if self.gpu_type:
            return self.gpu_type
        if self.provider == "runpod":
            return "NVIDIA RTX A4000"  # ~$0.19/hr, widely available
        if self.provider == "lambda":
            return "gpu_1x_a10"  # Lambda A10
        if self.provider == "vast":
            return "RTX 4090"
        if self.provider == "aws":
            return "g5.xlarge"  # 1x A10G
        return ""

    def default_image(self) -> str:
        if self.base_image:
            return self.base_image
        # A widely-available PyTorch image with Python 3.10 + CUDA
        return "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


# ---------------------------------------------------------------------------
# RunPod provider
# ---------------------------------------------------------------------------

_RUNPOD_GQL = "https://api.runpod.io/graphql"


def _runpod_gql(api_key: str, query: str) -> dict[str, Any]:
    """Execute a GraphQL request against the RunPod API."""
    payload = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        _RUNPOD_GQL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data: dict[str, Any] = json.loads(resp.read())
        return data
    except urllib.error.HTTPError as exc:
        body = exc.read(500).decode(errors="replace")
        raise RuntimeError(f"RunPod API HTTP {exc.code}: {body}") from exc


def provision_runpod(cfg: CloudGPUConfig) -> dict[str, Any]:
    """Create a RunPod pod and return {pod_id, host, port}.

    The pod runs sshd on port 22 with key-based auth using the machine's
    default public key (RunPod injects ~/.ssh/authorized_keys from the account).
    """
    api_key = cfg.api_key()
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY not set — export it before using cloud_gpu")

    gpu = cfg.default_gpu_type().replace('"', '\\"')
    image = cfg.default_image().replace('"', '\\"')

    # RunPod GraphQL mutation — creates a pod with SSH on port 22
    mutation = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{
        cloudType: SECURE
        gpuCount: 1
        containerDiskInGb: 40
        minVcpuCount: 4
        minMemoryInGb: 16
        gpuTypeId: "{gpu}"
        imageName: "{image}"
        name: "rf-experiment"
        ports: "22/tcp"
        startSsh: true
      }}) {{
        id
        desiredStatus
        machine {{ podHostId host port }}
      }}
    }}
    """
    result = _runpod_gql(api_key, mutation)
    errors = result.get("errors")
    if errors:
        raise RuntimeError(f"RunPod error: {errors[0].get('message', errors)}")

    pod = result["data"]["podFindAndDeployOnDemand"]
    pod_id = pod["id"]
    return {"pod_id": pod_id, "provider": "runpod"}


def wait_runpod_ready(pod_id: str, api_key: str, timeout_s: int = 300) -> tuple[str, int]:
    """Poll until the pod is running and SSH-accessible. Returns (host, port)."""
    deadline = time.monotonic() + timeout_s
    query = f"""
    query {{
      pod(input: {{ podId: "{pod_id}" }}) {{
        id runtime {{ uptimeInSeconds gpus {{ id gpuUtilPercent }} }}
        desiredStatus
        machine {{ podHostId host port }}
      }}
    }}
    """
    while time.monotonic() < deadline:
        result = _runpod_gql(api_key, query)
        pod = result.get("data", {}).get("pod", {})
        runtime = pod.get("runtime")
        machine = pod.get("machine") or {}
        host = machine.get("host", "")
        port = machine.get("port", 22) or 22

        if runtime and host:
            # Pod is running and has network info
            return host, int(port)

        time.sleep(15)

    raise TimeoutError(f"RunPod pod {pod_id} did not become ready within {timeout_s}s")


def terminate_runpod(pod_id: str, api_key: str) -> None:
    """Terminate (delete) a RunPod pod."""
    mutation = f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}'
    try:
        _runpod_gql(api_key, mutation)
    except Exception as exc:
        # A pod that did not terminate keeps billing — say so loudly.
        print(
            f"⚠️  RunPod pod {pod_id} may still be running (terminate failed: {exc}). "
            "Check https://www.runpod.io/console/pods",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Lambda Labs provider
# ---------------------------------------------------------------------------

_LAMBDA_BASE = "https://cloud.lambdalabs.com/api/v1"


def _lambda_req(
    method: str, path: str, api_key: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    import base64

    token = base64.b64encode(f"{api_key}:".encode()).decode()
    url = f"{_LAMBDA_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result: dict[str, Any] = json.loads(resp.read())
        return result
    except urllib.error.HTTPError as exc:
        body_text = exc.read(500).decode(errors="replace")
        raise RuntimeError(f"Lambda API HTTP {exc.code}: {body_text}") from exc


def provision_lambda(cfg: CloudGPUConfig) -> dict[str, Any]:
    """Launch a Lambda Labs instance and return {instance_id}."""
    api_key = cfg.api_key()
    if not api_key:
        raise RuntimeError("LAMBDA_API_KEY not set — export it before using cloud_gpu")

    instance_type = cfg.default_gpu_type()
    region = cfg.region or "us-west-2"

    # Get available instance types to validate
    types_resp = _lambda_req("GET", "/instance-types", api_key)
    available = {
        k: v
        for k, v in (types_resp.get("data") or {}).items()
        if v.get("regions_with_capacity_available")
    }
    if instance_type not in available:
        # Fall back to cheapest available
        cheapest = min(
            available.keys(), key=lambda k: available[k].get("price_cents_per_hour", 9999)
        )
        instance_type = cheapest

    body = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [],  # Uses account-level SSH keys
        "quantity": 1,
        "name": "rf-experiment",
    }
    resp = _lambda_req("POST", "/instance-operations/launch", api_key, body)
    instance_ids = resp.get("data", {}).get("instance_ids", [])
    if not instance_ids:
        raise RuntimeError(f"Lambda launch returned no instance IDs: {resp}")
    return {"pod_id": instance_ids[0], "provider": "lambda"}


def wait_lambda_ready(instance_id: str, api_key: str, timeout_s: int = 300) -> tuple[str, int]:
    """Poll until the Lambda instance is running. Returns (ip, 22)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = _lambda_req("GET", f"/instances/{instance_id}", api_key)
        inst = resp.get("data") or {}
        status = inst.get("status", "")
        ip = inst.get("ip") or ""
        if status == "active" and ip:
            return ip, 22
        time.sleep(15)
    raise TimeoutError(f"Lambda instance {instance_id} did not become ready within {timeout_s}s")


def terminate_lambda(instance_id: str, api_key: str) -> None:
    """Terminate a Lambda instance."""
    try:
        _lambda_req(
            "POST", "/instance-operations/terminate", api_key, {"instance_ids": [instance_id]}
        )
    except Exception as exc:
        # An instance that did not terminate keeps billing — say so loudly.
        print(
            f"⚠️  Lambda instance {instance_id} may still be running (terminate failed: {exc}). "
            "Check https://cloud.lambdalabs.com/instances",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# vast.ai provider (REST API, https://console.vast.ai/api/v0)
# ---------------------------------------------------------------------------

_VAST_API = "https://console.vast.ai/api/v0"


def _vast_req(
    method: str, path: str, api_key: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{_VAST_API}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result: dict[str, Any] = json.loads(resp.read() or b"{}")
        return result
    except urllib.error.HTTPError as exc:
        body_text = exc.read(500).decode(errors="replace")
        raise RuntimeError(f"vast.ai API HTTP {exc.code}: {body_text}") from exc


def _vast_offers(cfg: CloudGPUConfig, api_key: str) -> list[dict[str, Any]]:
    query: dict[str, Any] = {
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
    }
    if cfg.gpu_type:
        query["gpu_name"] = {"eq": cfg.gpu_type}
    if cfg.max_price_per_hr > 0:
        query["dph_total"] = {"lte": cfg.max_price_per_hr}
    from urllib.parse import quote

    resp = _vast_req("GET", f"/bundles/?q={quote(json.dumps(query))}", api_key)
    offers: list[dict[str, Any]] = resp.get("offers") or []
    return offers


def provision_vast(cfg: CloudGPUConfig) -> dict[str, Any]:
    """Rent the cheapest matching vast.ai offer and return {pod_id}."""
    api_key = cfg.api_key()
    if not api_key:
        raise RuntimeError("VAST_API_KEY not set — export it before using cloud_gpu")
    offers = _vast_offers(cfg, api_key)
    if not offers:
        raise RuntimeError(
            f"No rentable vast.ai offer for {cfg.gpu_type or 'any GPU'}"
            + (f" under ${cfg.max_price_per_hr}/hr" if cfg.max_price_per_hr > 0 else "")
        )
    body = {"client_id": "me", "image": cfg.default_image(), "disk": 32, "runtype": "ssh"}
    resp = _vast_req("PUT", f"/asks/{offers[0]['id']}/", api_key, body)
    instance_id = resp.get("new_contract")
    if not resp.get("success") or not instance_id:
        raise RuntimeError(f"vast.ai rent failed: {resp}")
    return {"pod_id": str(instance_id), "provider": "vast"}


def wait_vast_ready(instance_id: str, api_key: str, timeout_s: int = 600) -> tuple[str, int]:
    """Poll until the vast.ai instance is running with SSH. Returns (host, port)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = _vast_req("GET", f"/instances/{instance_id}/", api_key)
        inst = resp.get("instances") or {}
        if inst.get("actual_status") == "running" and inst.get("ssh_host"):
            return str(inst["ssh_host"]), int(inst.get("ssh_port") or 22)
        time.sleep(15)
    raise TimeoutError(f"vast.ai instance {instance_id} did not become ready within {timeout_s}s")


def terminate_vast(instance_id: str, api_key: str) -> None:
    try:
        _vast_req("DELETE", f"/instances/{instance_id}/", api_key)
    except Exception as exc:
        print(
            f"⚠️  vast.ai instance {instance_id} may still be running (destroy failed: {exc}). "
            "Check https://cloud.vast.ai/instances/",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# AWS EC2 provider (through the aws CLI — uses your normal AWS credentials)
# ---------------------------------------------------------------------------


def _aws(args: list[str], region: str) -> dict[str, Any]:
    import subprocess

    cmd = ["aws", "ec2", *args, "--output", "json"] + (["--region", region] if region else [])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"aws {' '.join(args[:1])} failed: {res.stderr.strip()[:300]}")
    out: dict[str, Any] = json.loads(res.stdout or "{}")
    return out


def provision_aws(cfg: CloudGPUConfig) -> dict[str, Any]:
    """Launch one EC2 GPU instance and return {pod_id}. Needs an AMI, key pair and SSH group."""
    missing = [
        name
        for name, value in (
            ("--image (a Deep Learning AMI id)", cfg.base_image),
            ("--ssh-key-name", cfg.ssh_key_name),
            ("--security-group-id", cfg.security_group_id),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("AWS needs " + ", ".join(missing) + " in cloud-gpu-setup")
    resp = _aws(
        [
            "run-instances",
            "--image-id",
            cfg.base_image,
            "--instance-type",
            cfg.default_gpu_type(),
            "--key-name",
            cfg.ssh_key_name,
            "--security-group-ids",
            cfg.security_group_id,
            "--count",
            "1",
            "--instance-initiated-shutdown-behavior",
            "terminate",
            "--tag-specifications",
            "ResourceType=instance,Tags=[{Key=Name,Value=rf-experiment}]",
        ],
        cfg.region,
    )
    instances = resp.get("Instances") or []
    if not instances:
        raise RuntimeError(f"aws run-instances returned no instance: {resp}")
    return {"pod_id": instances[0]["InstanceId"], "provider": "aws"}


def wait_aws_ready(instance_id: str, region: str) -> tuple[str, int]:
    import subprocess

    wait = ["aws", "ec2", "wait", "instance-running", "--instance-ids", instance_id]
    subprocess.run(wait + (["--region", region] if region else []), check=False, timeout=900)
    resp = _aws(["describe-instances", "--instance-ids", instance_id], region)
    for reservation in resp.get("Reservations") or []:
        for inst in reservation.get("Instances") or []:
            if inst.get("PublicIpAddress"):
                return str(inst["PublicIpAddress"]), 22
    raise RuntimeError(f"EC2 instance {instance_id} has no public IP (check the subnet/VPC)")


def terminate_aws(instance_id: str, region: str) -> None:
    try:
        _aws(["terminate-instances", "--instance-ids", instance_id], region)
    except Exception as exc:
        print(
            f"⚠️  EC2 instance {instance_id} may still be running (terminate failed: {exc}). "
            "Check the EC2 console.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Unified provider dispatch
# ---------------------------------------------------------------------------


def provision_and_wait(cfg: CloudGPUConfig) -> tuple[str, int, str]:
    """Provision an instance and wait for SSH. Returns (host, port, pod_id).

    Raises RuntimeError if provisioning fails or times out.
    """
    api_key = cfg.api_key()
    if not api_key:
        raise RuntimeError(
            f"No API key found for provider '{cfg.provider}'. "
            f"Set {cfg.api_key_env or 'RUNPOD_API_KEY / LAMBDA_API_KEY / VAST_API_KEY'} "
            "(AWS needs the aws CLI installed and configured)."
        )

    if cfg.provider == "runpod":
        info = provision_runpod(cfg)
        host, port = wait_runpod_ready(info["pod_id"], api_key)
    elif cfg.provider == "lambda":
        info = provision_lambda(cfg)
        host, port = wait_lambda_ready(info["pod_id"], api_key)
    elif cfg.provider == "vast":
        info = provision_vast(cfg)
        host, port = wait_vast_ready(info["pod_id"], api_key)
    elif cfg.provider == "aws":
        info = provision_aws(cfg)
        host, port = wait_aws_ready(info["pod_id"], cfg.region)
    else:
        raise RuntimeError(
            f"Unsupported cloud GPU provider: '{cfg.provider}'. Use runpod, lambda, vast or aws."
        )

    return host, port, info["pod_id"]


def terminate_instance(cfg: CloudGPUConfig, pod_id: str) -> None:
    """Terminate the instance (best-effort — never raises)."""
    api_key = cfg.api_key()
    try:
        if cfg.provider == "runpod":
            terminate_runpod(pod_id, api_key)
        elif cfg.provider == "lambda":
            terminate_lambda(pod_id, api_key)
        elif cfg.provider == "vast":
            terminate_vast(pod_id, api_key)
        elif cfg.provider == "aws":
            terminate_aws(pod_id, cfg.region)
    except Exception:
        pass


def list_gpu_types(provider: str, api_key: str) -> list[dict[str, Any]]:
    """Return available GPU types for the given provider.

    Each dict has: name, price_per_hr (float), availability (bool).
    """
    try:
        if provider == "runpod":
            query = """
            query {
              gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice(input: { gpuCount: 1 }) { minimumBidPrice uninterruptablePrice }
              }
            }
            """
            result = _runpod_gql(api_key, query)
            types = result.get("data", {}).get("gpuTypes", []) or []
            return [
                {
                    "name": t["id"],
                    "display": t.get("displayName", t["id"]),
                    "memory_gb": t.get("memoryInGb", 0),
                    "price_per_hr": (t.get("lowestPrice") or {}).get("uninterruptablePrice") or 0.0,
                    "secure_cloud": t.get("secureCloud", False),
                }
                for t in types
                if t.get("secureCloud")
            ]

        if provider == "vast":
            offers = _vast_offers(CloudGPUConfig(provider="vast"), api_key)
            return [
                {
                    "name": o.get("gpu_name", ""),
                    "display": f"{o.get('gpu_name', '')} · {o.get('geolocation', '')}",
                    "memory_gb": round((o.get("gpu_ram") or 0) / 1024),
                    "price_per_hr": float(o.get("dph_total") or 0.0),
                }
                for o in offers[:40]
            ]

        if provider == "lambda":
            resp = _lambda_req("GET", "/instance-types", api_key)
            data = resp.get("data") or {}
            return [
                {
                    "name": k,
                    "display": k,
                    "price_per_hr": v.get("price_cents_per_hour", 0) / 100,
                    "available": bool(v.get("regions_with_capacity_available")),
                }
                for k, v in data.items()
            ]
    except Exception:
        pass
    return []
