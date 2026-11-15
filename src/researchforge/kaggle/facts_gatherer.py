"""Gather all facts about a competition before asking the user anything.

Produces a single JSON document (.overnight_<slug>/facts.json) that Claude
reads to decide exactly which questions to ask and in what order.

Zero user input required. Runs in ~30s.

Output schema (facts.json):
  slug, title, task_type, metric, metric_direction, metric_direction_source,
  deadline_hours,
  is_code_competition, internet_allowed_in_kernels,
  data_size_gb, starter_notebooks,
  local { free_disk_gb, gpu_type, gpu_name, viable, reason },
  kaggle_gpu { credentials_exist, username, quota_hours_remaining, viable },
  colab { viable, manual_step },
  ssh { viable, reason, requires }   (viable=True once ssh-setup has run),
  cloud_gpu { viable, reason, requires }   (viable=True once cloud-gpu-setup has run),
  top_kernel { ref, title, author, lb_score },
  existing_ideas_count,
  recommended_compute, recommended_reason,
  auto_run_driver, auto_run_reason, auto_run_ready, auto_run_fix
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class ComputeOption:
    viable: bool | str  # True / False / "unknown"
    reason: str = ""
    requires: list[str] = field(default_factory=list)


@dataclass
class LocalOption(ComputeOption):
    free_disk_gb: float = 0.0
    gpu_type: str = "none"  # "cuda" | "mps" | "none"
    gpu_name: str = ""


@dataclass
class KaggleGpuOption(ComputeOption):
    credentials_exist: bool = False
    username: str = ""
    quota_hours_remaining: float = 0.0


@dataclass
class TopKernel:
    ref: str
    title: str
    author: str
    lb_score: float | None


@dataclass
class StarterNotebook:
    ref: str
    title: str
    author: str
    is_official: bool  # posted by competition organiser


@dataclass
class CompetitionFacts:
    """Everything gathered before asking the user a single question."""

    slug: str
    title: str = ""
    task_type: str = ""
    metric: str = ""
    metric_direction: str = ""
    metric_direction_source: str = ""  # leaderboard | ai | keyword | guess | user
    deadline_hours: float | None = None
    is_code_competition: bool = False
    internet_allowed_in_kernels: bool = True
    data_size_gb: float = 0.0
    starter_notebooks: list[StarterNotebook] = field(default_factory=list)
    top_kernel: TopKernel | None = None
    existing_ideas_count: int = 0
    local: LocalOption = field(default_factory=lambda: LocalOption(viable=False))
    kaggle_gpu: KaggleGpuOption = field(default_factory=lambda: KaggleGpuOption(viable=False))
    colab: ComputeOption = field(default_factory=lambda: ComputeOption(viable=True))
    ssh: ComputeOption = field(default_factory=lambda: ComputeOption(viable="unknown"))
    cloud_gpu: ComputeOption = field(default_factory=lambda: ComputeOption(viable="unknown"))
    recommended_compute: str = ""
    recommended_reason: str = ""
    # Auto-run: the driver that matches how the user runs this (see auto_run).
    auto_run_driver: str = ""
    auto_run_reason: str = ""
    auto_run_ready: bool = False
    auto_run_fix: str = ""
    gathered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / "facts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, artifacts_dir: Path) -> CompetitionFacts | None:
        path = artifacts_dir / "facts.json"
        if not path.exists():
            return None
        with path.open() as fh:
            data = json.load(fh)
        return _from_dict(data)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def gather_facts(slug: str, artifacts_dir: Path, force_refresh: bool = False) -> CompetitionFacts:
    """Collect all competition facts. Loads from cache if available."""
    if not force_refresh:
        cached = CompetitionFacts.load(artifacts_dir)
        if cached is not None:
            # Refresh if older than 1 hour
            try:
                age_seconds = (
                    datetime.now() - datetime.fromisoformat(cached.gathered_at)
                ).total_seconds()
                if age_seconds < 3600:
                    return cached
            except (ValueError, AttributeError):
                pass

    facts = CompetitionFacts(slug=slug)

    # --- Competition metadata ---
    _enrich_from_metadata(facts)

    # --- Local compute ---
    facts.local = _check_local(facts.data_size_gb)

    # --- Kaggle GPU ---
    facts.kaggle_gpu = _check_kaggle_gpu()

    # --- Colab (always viable, manual step) ---
    facts.colab = _check_colab()

    # --- SSH and cloud GPU (unknown until configured) ---
    facts.ssh, facts.cloud_gpu = _check_remote(artifacts_dir)

    # --- Top kernel + starter notebooks ---
    _enrich_kernels(facts)

    # --- Existing idea sources ---
    pool_path = artifacts_dir / "pending_hypotheses.json"
    if pool_path.exists():
        try:
            facts.existing_ideas_count = len(json.loads(pool_path.read_text()))
        except Exception:
            facts.existing_ideas_count = 0

    # --- Recommendation ---
    facts.recommended_compute, facts.recommended_reason = _recommend_compute(facts)
    _suggest_auto_run(facts, artifacts_dir)

    facts.save(artifacts_dir)
    return facts


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------


def _enrich_from_metadata(facts: CompetitionFacts) -> None:
    """Populate title, metric, deadline, data size from Kaggle CLI + page."""

    from researchforge.kaggle.competition_reader import (
        _detect_code_competition,
        _estimate_data_size,
        _fetch_public_page,
    )

    slug = facts.slug

    # Use existing reader but don't cache in a real artifacts dir yet
    try:
        page_text = _fetch_public_page(slug)
        facts.is_code_competition = _detect_code_competition(slug, page_text)
        facts.data_size_gb = _estimate_data_size(slug)
        # Internet in kernels: check if competition rules mention "internet disabled"
        if page_text:
            import re

            if re.search(r"(?i)internet\s*(access\s*)?(is\s*)?disabled|no\s*internet", page_text):
                facts.internet_allowed_in_kernels = False
    except Exception:
        pass

    # Deadline from CLI
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "list", "--search", slug, "--csv"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode == 0:
            lines = [
                ln
                for ln in result.stdout.strip().splitlines()
                if ln.strip() and not ln.strip().lower().startswith("warning")
            ]
            if len(lines) >= 2:
                parts = lines[1].split(",")
                headers = [h.strip() for h in lines[0].split(",")]
                row = dict(zip(headers, [p.strip() for p in parts], strict=False))
                facts.title = row.get("ref", slug).split("/")[-1].replace("-", " ").title()
                if row.get("deadline"):
                    try:
                        deadline_dt = datetime.fromisoformat(row["deadline"].replace("Z", "+00:00"))
                        facts.deadline_hours = round(
                            (deadline_dt.replace(tzinfo=None) - datetime.utcnow()).total_seconds()
                            / 3600,
                            1,
                        )
                    except Exception:
                        pass
    except Exception:
        pass

    # Metric from competition_reader heuristic
    from researchforge.kaggle.competition_reader import CompetitionMetadata, _heuristic_enrich

    m = CompetitionMetadata(slug=slug)
    _heuristic_enrich(m)
    facts.metric = m.metric_name
    facts.metric_direction = m.metric_direction
    facts.metric_direction_source = m.metric_direction_source
    from researchforge.kaggle.api import leaderboard_direction

    measured = leaderboard_direction(slug)
    if measured:
        facts.metric_direction, facts.metric_direction_source = measured, "leaderboard"
    facts.task_type = m.task_type


# ---------------------------------------------------------------------------
# Local compute check
# ---------------------------------------------------------------------------


def _check_local(data_size_gb: float) -> LocalOption:
    disk = shutil.disk_usage(Path.home())
    free_gb = disk.free / 1e9

    # GPU detection
    gpu_type = "none"
    gpu_name = ""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_type = "mps"
            gpu_name = "Apple Silicon"
    except ImportError:
        pass

    viable = data_size_gb == 0 or data_size_gb <= free_gb * 0.8
    if viable:
        reason = f"Data fits locally ({data_size_gb:.0f} GB, {free_gb:.0f} GB free)"
        if gpu_type != "none":
            reason += f". {gpu_name} GPU available."
    else:
        reason = f"Data ({data_size_gb:.0f} GB) exceeds free disk ({free_gb:.0f} GB)"

    return LocalOption(
        viable=viable,
        reason=reason,
        free_disk_gb=round(free_gb, 1),
        gpu_type=gpu_type,
        gpu_name=gpu_name,
    )


def _suggest_auto_run(facts: CompetitionFacts, artifacts_dir: Path) -> None:
    """Which auto-run driver to offer: the API key setup will run on, else this agent."""
    from researchforge.ai.providers import get_provider
    from researchforge.kaggle import auto_run as ar

    if (artifacts_dir / "overnight_setup.json").is_file():
        driver, why = ar.suggested_driver(artifacts_dir)
    elif get_provider() is not None:
        driver, why = "byok", "an AI API key is set, so setup will run this project headless on it"
    else:
        driver, why = ar.suggested_driver(artifacts_dir)
    ready = ar.readiness(ar.AutoRunConfig(driver=driver))
    facts.auto_run_driver, facts.auto_run_reason = driver, why
    facts.auto_run_ready, facts.auto_run_fix = ready.ready, ready.fix


def _check_colab() -> ComputeOption:
    """Colab runs unattended through Google's Colab CLI once it is installed and signed in."""
    from researchforge.kaggle import colab

    st = colab.status()
    if st["signed_in"]:
        return ComputeOption(
            viable=True,
            reason=f"Runs unattended through Google's Colab CLI ({st['detail']}).",
        )
    if st["installed"]:
        return ComputeOption(
            viable="unknown",
            reason="Google's Colab CLI is installed but not signed in; sign in once to run "
            "rounds unattended (or open each notebook and click Run All).",
            requires=["colab_sign_in"],
        )
    return ComputeOption(
        viable="unknown",
        reason="Needs Google's Colab CLI (installed for you if you pick Colab) and a one-time "
        "sign-in; without it you open each notebook and click Run All.",
        requires=["colab_cli", "colab_sign_in"],
    )


def _check_remote(artifacts_dir: Path) -> tuple[ComputeOption, ComputeOption]:
    """SSH and cloud-GPU options: ready if already configured, else what they need."""
    from researchforge.kaggle.cloud_gpu_config import CloudGPUConfig
    from researchforge.kaggle.ssh_config import SSHConfig

    ssh_cfg = SSHConfig.load(artifacts_dir)
    ssh = (
        ComputeOption(viable=True, reason=f"Configured: {ssh_cfg.host}.")
        if ssh_cfg
        else ComputeOption(
            viable="unknown",
            reason="Your own GPU server. Run `kaggle ssh-setup` with its host and key first.",
            requires=["ssh_host", "ssh_key_path"],
        )
    )
    cloud_cfg = CloudGPUConfig.load(artifacts_dir)
    if cloud_cfg:
        gpu = cloud_cfg.gpu_type or "cheapest available GPU"
        cap = f", up to ${cloud_cfg.max_price_per_hr}/h" if cloud_cfg.max_price_per_hr else ""
        cloud = ComputeOption(viable=True, reason=f"Configured: {cloud_cfg.provider}, {gpu}{cap}.")
    else:
        cloud = ComputeOption(
            viable="unknown",
            reason=(
                "Pay-per-hour GPU on RunPod, Lambda, vast.ai or AWS; one instance per "
                "experiment, terminated after. Needs a provider API key, then "
                "`kaggle cloud-gpu-setup` (--list-gpus shows types and prices)."
            ),
            requires=["provider", "api_key_env_var", "gpu_type"],
        )
    return ssh, cloud


# ---------------------------------------------------------------------------
# Kaggle GPU check
# ---------------------------------------------------------------------------


def _check_kaggle_gpu() -> KaggleGpuOption:
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    if not creds_path.exists():
        return KaggleGpuOption(
            viable=False,
            credentials_exist=False,
            reason="Kaggle credentials not found (will set up during overnight flow)",
            requires=["kaggle_json"],
        )

    try:
        creds = json.loads(creds_path.read_text())
        username = creds.get("username", "")
    except Exception:
        username = ""

    # Quota: Kaggle doesn't expose remaining quota directly via CLI.
    # Approximate by checking if the account can list kernels.
    quota_hours = 30.0  # default assumption for free tier
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        viable = result.returncode == 0
    except Exception:
        viable = False

    return KaggleGpuOption(
        viable=viable,
        credentials_exist=True,
        username=username,
        quota_hours_remaining=quota_hours,
        reason=(
            f"Kaggle free GPU available (~{quota_hours:.0f}h/week quota). "
            "Data stays on Kaggle's servers."
        ),
    )


# ---------------------------------------------------------------------------
# Kernels enrichment
# ---------------------------------------------------------------------------


def _enrich_kernels(facts: CompetitionFacts) -> None:
    """Find top kernel and starter notebooks."""
    from researchforge.kaggle.api import KaggleAPIError, list_competition_kernels
    from researchforge.kaggle.notebook_baseline import _extract_lb_score

    try:
        kernels = list_competition_kernels(facts.slug, sort_by="scoreDescending", page_size=30)
    except KaggleAPIError:
        return

    # Top kernel by score
    for k in kernels:
        lb = _extract_lb_score(k.title)
        if lb is not None:
            facts.top_kernel = TopKernel(ref=k.ref, title=k.title, author=k.author, lb_score=lb)
            break
    if facts.top_kernel is None and kernels:
        k = kernels[0]
        facts.top_kernel = TopKernel(ref=k.ref, title=k.title, author=k.author, lb_score=None)

    # Starter notebooks: kernels with "starter", "baseline", "tutorial" in title
    import re

    starter_patterns = (
        r"(?i)\b(starter|baseline|tutorial|getting\s*started|introduction|beginner|simple|basic)\b"
    )
    for k in kernels:
        if re.search(starter_patterns, k.title):
            # Consider official if author matches competition slug pattern
            is_official = facts.slug.split("-")[0].lower() in k.author.lower()
            facts.starter_notebooks.append(
                StarterNotebook(ref=k.ref, title=k.title, author=k.author, is_official=is_official)
            )
            if len(facts.starter_notebooks) >= 3:
                break


# ---------------------------------------------------------------------------
# Compute recommendation
# ---------------------------------------------------------------------------


def _recommend_compute(facts: CompetitionFacts) -> tuple[str, str]:
    """Return (compute_id, reason) for the best compute option."""
    # Code competition → must use Kaggle GPU (data on their servers)
    if facts.is_code_competition and facts.kaggle_gpu.viable:
        return (
            "kaggle_gpu",
            "Code competition — experiments must run as Kaggle notebooks. "
            "Data stays on Kaggle's servers, no download needed.",
        )

    # Large data that doesn't fit locally → Kaggle GPU
    if not facts.local.viable and facts.kaggle_gpu.viable:
        return (
            "kaggle_gpu",
            f"Data ({facts.data_size_gb:.0f} GB) too large for local disk. "
            "Kaggle GPU uses data already on their servers.",
        )

    # Local GPU available and data fits → local
    if facts.local.viable and facts.local.gpu_type != "none":
        return (
            "local",
            f"Local {facts.local.gpu_name} GPU available and data fits ({facts.data_size_gb:.1f} "
            "GB).",
        )

    # Local CPU and data fits → local (slower but works)
    if facts.local.viable:
        return (
            "local",
            f"Data fits locally ({facts.data_size_gb:.1f} GB). "
            "Will run on CPU — consider Kaggle GPU for faster experiments.",
        )

    # Kaggle GPU as fallback
    if facts.kaggle_gpu.viable:
        return "kaggle_gpu", "Kaggle free GPU available."

    # Colab as last automated option
    return "colab", "Colab GPU available — you will need to click Run in browser."


# ---------------------------------------------------------------------------
# Deserialisation helper
# ---------------------------------------------------------------------------


def _from_dict(data: dict[str, Any]) -> CompetitionFacts:
    """Reconstruct CompetitionFacts from JSON dict."""
    top = data.pop("top_kernel", None)
    starters = data.pop("starter_notebooks", [])
    local_d = data.pop("local", {})
    kaggle_d = data.pop("kaggle_gpu", {})
    colab_d = data.pop("colab", {})
    ssh_d = data.pop("ssh", {})
    cloud_d = data.pop("cloud_gpu", {})

    facts = CompetitionFacts(
        **{k: v for k, v in data.items() if k in CompetitionFacts.__dataclass_fields__}
    )

    if top:
        facts.top_kernel = TopKernel(**top)
    facts.starter_notebooks = [StarterNotebook(**s) for s in starters]
    facts.local = LocalOption(**local_d) if local_d else LocalOption(viable=False)
    facts.kaggle_gpu = KaggleGpuOption(**kaggle_d) if kaggle_d else KaggleGpuOption(viable=False)
    facts.colab = ComputeOption(**colab_d) if colab_d else ComputeOption(viable=True)
    facts.ssh = ComputeOption(**ssh_d) if ssh_d else ComputeOption(viable="unknown")
    facts.cloud_gpu = ComputeOption(**cloud_d) if cloud_d else ComputeOption(viable="unknown")
    return facts
