"""AI-powered Dockerfile generation.

Generates a working Dockerfile for a project, using the repo scan
to infer base image, system dependencies, and install commands.

Without an AI provider: builds a sensible minimal Dockerfile from
the scan heuristics (covers ~80% of Python ML projects).

With an AI provider: produces a fully tailored Dockerfile.
"""

from __future__ import annotations

import re

from researchforge.ai.providers import AiProvider
from researchforge.domain.project import Project
from researchforge.domain.repo_scan import RepoScan

# ---------------------------------------------------------------------------
# Heuristic (no AI) Dockerfile builder
# ---------------------------------------------------------------------------

_GPU_PACKAGES = {"torch", "torchvision", "torchaudio", "tensorflow", "jax", "cudf", "cuml"}
_CV_PACKAGES = {"opencv-python", "opencv-python-headless", "cv2", "Pillow", "pillow"}
_SYS_DEPS_FOR_CV = "libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev"
_SYS_DEPS_FOR_GIT = "git"

# arXiv-style detection: if any dep name contains these, add syslibs
_NEEDS_CV_LIBS = {"cv2", "opencv", "PIL", "pillow", "Pillow", "imageio", "skimage"}


def _python_version_from_scan(scan: RepoScan) -> str:
    if scan.python.python_requires:
        m = re.search(r">=\s*(\d+\.\d+)", scan.python.python_requires)
        if m:
            major_minor = m.group(1)
            # Round up to nearest supported slim version
            parts = major_minor.split(".")
            return f"{parts[0]}.{parts[1]}"
    return "3.11"


def _needs_cv_libs(deps: list[str]) -> bool:
    lowered = {d.lower() for d in deps}
    return any(k.lower() in lowered for k in _NEEDS_CV_LIBS)


def _needs_gpu(deps: list[str]) -> bool:
    lowered = {d.lower() for d in deps}
    return any(k.lower() in lowered for k in _GPU_PACKAGES)


def build_minimal_dockerfile(scan: RepoScan) -> str:
    """Generate a minimal working Dockerfile without AI."""
    python_ver = _python_version_from_scan(scan)
    deps = scan.python.dependencies

    sys_pkgs = [_SYS_DEPS_FOR_GIT]
    if _needs_cv_libs(deps):
        sys_pkgs.append(_SYS_DEPS_FOR_CV)

    sys_install = " ".join(sys_pkgs)

    lines = [
        f"FROM python:{python_ver}-slim",
        "",
        "WORKDIR /workspace",
        "",
        "# System dependencies",
        "RUN apt-get update && apt-get install -y \\",
        f"    {sys_install} \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
    ]

    # Install dependencies
    if scan.python.requirements_files:
        req = scan.python.requirements_files[0]
        lines += [
            f"# Python dependencies from {req}",
            f"COPY {req} .",
            "RUN pip install --no-cache-dir --upgrade pip && \\",
            f"    pip install --no-cache-dir -r {req}",
            "",
        ]
    elif scan.python.has_pyproject or scan.python.has_setup_py:
        lines += [
            "# Python dependencies via pip install",
            "COPY . .",
            "RUN pip install --no-cache-dir --upgrade pip && \\",
            "    pip install --no-cache-dir -e .",
            "",
        ]
    else:
        lines += [
            "# No requirements detected — add your install step here",
            "# COPY requirements.txt .",
            "# RUN pip install --no-cache-dir -r requirements.txt",
            "",
        ]

    lines += [
        "# Copy project",
        "COPY . .",
        "",
        "# ResearchForge runs: python benchmarks/evaluate.py",
        'CMD ["python", "benchmarks/evaluate.py"]',
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI-powered Dockerfile builder
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are generating a production-quality Dockerfile for a machine learning project.

Given the repository context, generate a Dockerfile that:
1. Uses an appropriate base image (python:X.Y-slim for CPU, or nvidia/cuda for GPU)
2. Installs all required system libraries (libgl1 for OpenCV, etc.)
3. Installs Python dependencies efficiently (COPY requirements first, then COPY .)
4. Works with ResearchForge experiments (benchmark runs `python benchmarks/evaluate.py`)
5. Does NOT bake in the model weights or large datasets — those are downloaded at runtime

RULES:
- Use multi-stage only if clearly beneficial
- Always run `pip install --upgrade pip` before installing packages
- Always use `--no-cache-dir` with pip
- Clean apt lists after apt-get install
- WORKDIR must be /workspace
- Output ONLY the Dockerfile content, nothing else

OUTPUT FORMAT:
<dockerfile>
(Dockerfile content here)
</dockerfile>
"""


def generate_dockerfile_with_ai(
    project: Project,
    scan: RepoScan,
    provider: AiProvider,
    *,
    cuda: bool = False,
) -> str:
    """Use an AI provider to generate a tailored Dockerfile."""
    deps = scan.python.dependencies[:30]
    req_files = scan.python.requirements_files

    user_prompt = (
        f"## Project objective\n{project.objective or 'ML experiment'}\n\n"
        f"## Repository info\n"
        f"- Python: {scan.python.python_requires or 'any'}\n"
        f"- Package name: {scan.python.package_name or 'unknown'}\n"
        f"- Requirements files: {', '.join(req_files) or 'none'}\n"
        f"- Has pyproject.toml: {scan.python.has_pyproject}\n"
        f"- Has setup.py: {scan.python.has_setup_py}\n"
        f"- Dependencies: {', '.join(deps)}\n"
        f"- Keywords: {', '.join(scan.keywords[:10])}\n"
        f"- GPU required: {cuda or _needs_gpu(scan.python.dependencies)}\n\n"
        "Generate a working Dockerfile for this project."
    )

    raw = provider.generate(_SYSTEM, user_prompt, max_tokens=2048)

    # Extract from tags
    m = re.search(r"<dockerfile>(.*?)</dockerfile>", raw, re.DOTALL)
    if m:
        content = m.group(1).strip()
        # Strip markdown fences
        content = re.sub(r"^```(?:dockerfile|docker)?\s*\n?", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n?```\s*$", "", content, flags=re.MULTILINE)
        return content.strip()

    # Fallback: return the raw response if no tags
    raw = re.sub(r"^```(?:dockerfile|docker)?\s*\n?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\n?```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()
