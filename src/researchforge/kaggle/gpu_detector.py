"""GPU detection from Kaggle competition metadata.

Parses competition description to detect:
  - Specialized GPU (competition-provided)
  - Kaggle free GPU availability
  - Colab GPU suggestion
  - Falls back to CPU if needed
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import StrEnum


class GPUBackend(StrEnum):
    """Available GPU/compute backends."""

    COMPETITION_GPU = "competition_gpu"  # Specialized GPU from competition
    KAGGLE_GPU = "kaggle_gpu"  # Kaggle free GPU quota
    COLAB_GPU = "colab_gpu"  # Google Colab GPU
    SSH_SERVER = "ssh_server"  # User-owned remote GPU/CPU server
    CLOUD_GPU = "cloud_gpu"  # Pay-per-hour cloud GPU (RunPod, Lambda)
    CPU = "cpu"  # Fallback to CPU


# Every name a user, the skill or gather-facts may use for a compute target.
# ``local`` and ``cpu`` both mean this machine; experiments there use its GPU
# (CUDA or Apple MPS) when it has one.
COMPUTE_TARGETS: dict[str, GPUBackend] = {
    "local": GPUBackend.CPU,
    "cpu": GPUBackend.CPU,
    "kaggle": GPUBackend.KAGGLE_GPU,
    "kaggle_gpu": GPUBackend.KAGGLE_GPU,
    "competition": GPUBackend.COMPETITION_GPU,
    "competition_gpu": GPUBackend.COMPETITION_GPU,
    "colab": GPUBackend.COLAB_GPU,
    "colab_gpu": GPUBackend.COLAB_GPU,
    "ssh": GPUBackend.SSH_SERVER,
    "ssh_server": GPUBackend.SSH_SERVER,
    "cloud": GPUBackend.CLOUD_GPU,
    "cloud_gpu": GPUBackend.CLOUD_GPU,
}
COMPUTE_CHOICES = "local, kaggle, colab, ssh, cloud, competition"


def resolve_compute_target(name: str) -> GPUBackend | None:
    """The backend a compute name means, or None if the name is not one."""
    return COMPUTE_TARGETS.get(name.strip().lower())


@dataclass
class GPUDetection:
    """Result of GPU detection for a competition."""

    competition_slug: str
    detected_backends: list[GPUBackend]  # Ranked by preference
    explanation: str  # Human-readable reason
    recommended: GPUBackend  # Top choice


def detect_gpu_from_competition(slug: str) -> GPUDetection:
    """Detect available GPU from competition metadata.

    Parses the competition description and pages for GPU mentions.
    Returns ranked list of available GPU backends.
    """
    explanation_parts = []
    backends = []

    # Try to get competition info
    competition_info = _get_competition_info(slug)

    # Check for specialized GPU mention
    if competition_info and _mentions_gpu(competition_info):
        backends.append(GPUBackend.COMPETITION_GPU)
        explanation_parts.append(f"Competition provides specialized GPU for {slug}")

    # Kaggle GPU is always available (with quota)
    backends.append(GPUBackend.KAGGLE_GPU)
    explanation_parts.append("Kaggle free GPU available (weekly quota)")

    # Colab is always a fallback option
    backends.append(GPUBackend.COLAB_GPU)
    explanation_parts.append("Google Colab GPU available as fallback")

    # CPU is ultimate fallback
    backends.append(GPUBackend.CPU)
    explanation_parts.append("CPU fallback available")

    return GPUDetection(
        competition_slug=slug,
        detected_backends=backends,
        explanation=" → ".join(explanation_parts),
        recommended=backends[0] if backends else GPUBackend.CPU,
    )


def _get_competition_info(slug: str) -> str | None:
    """Fetch competition metadata via Kaggle CLI.

    Returns:
        Competition description/metadata, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "info", slug, "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _mentions_gpu(text: str) -> bool:
    """Check if text mentions GPU/accelerator/hardware.

    Uses heuristics to detect GPU mentions in competition description.
    """
    patterns = [
        r"\bgpu\b",
        r"\baccelerator",
        r"\bspecialized\s+hardware",
        r"\bnvidia",
        r"\btpu\b",
        r"\bgpus\b",
        r"\bgpu\s+support",
    ]

    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def ask_user_gpu_preference(detection: GPUDetection) -> GPUBackend:
    """Prompt user to select GPU backend based on detection.

    In non-interactive mode (Claude Code shell) returns the recommended
    backend immediately so the setup flow is never blocked.
    """
    import sys

    if not sys.stdin.isatty():
        # Non-interactive: use the recommended backend and print to stderr
        # so Claude can see the choice and ask the user via its own UI if needed.
        print(
            f"\n\u26a1  GPU auto-selected (non-interactive): "
            f"{detection.recommended.value.replace('_', ' ').title()}",
            file=sys.stderr,
        )
        print(
            f"   Pass --gpu <target> to choose a different one (options: {COMPUTE_CHOICES})",
            file=sys.stderr,
        )
        return detection.recommended

    print(f"\n{'=' * 70}")
    print(f"GPU Detection for: {detection.competition_slug}")
    print(f"{'=' * 70}")
    print(f"\n{detection.explanation}\n")
    print("Available options:")

    options = [detection.recommended] + [
        b for b in detection.detected_backends if b != detection.recommended
    ]
    options += [b for b in (GPUBackend.SSH_SERVER, GPUBackend.CLOUD_GPU) if b not in options]
    notes = {
        GPUBackend.COMPETITION_GPU: "accelerator the competition provides",
        GPUBackend.KAGGLE_GPU: "free, limited weekly quota; data stays on Kaggle",
        GPUBackend.COLAB_GPU: "free; you open the notebook and click Run",
        GPUBackend.SSH_SERVER: "your own server; run `kaggle ssh-setup` first",
        GPUBackend.CLOUD_GPU: "pay per hour; run `kaggle cloud-gpu-setup` first",
        GPUBackend.CPU: "this machine, using its GPU if it has one",
    }
    for i, backend in enumerate(options, 1):
        marker = " \u2713 (recommended)" if backend == detection.recommended else ""
        label = "This machine" if backend == GPUBackend.CPU else backend.value.replace("_", " ")
        print(f"  {i}) {label.title()} — {notes[backend]}{marker}")

    while True:
        try:
            choice = input(f"\nSelect compute (1-{len(options)}) [1]: ").strip() or "1"
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected = options[idx]
                print(f"\u2713 Selected: {selected.value.replace('_', ' ').title()}")
                return selected
        except (ValueError, IndexError):
            pass
        print("Invalid selection. Try again.")
