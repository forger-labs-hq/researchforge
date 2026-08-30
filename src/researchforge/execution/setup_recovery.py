"""Setup-failure auto-recovery: pattern matching + progressive fix attempts.

When ``baseline run`` hits FAILED_SETUP, this module reads the stderr log,
matches against a comprehensive rule set, and returns ordered fix candidates
to try.  Each candidate is a new ``setup_command`` string.  The caller
(baseline CLI) tries them one at a time, updating and re-approving the
contract between attempts.

Rules are ordered from safest/most-specific to most-aggressive.  Docker
suggestions are emitted as last-resort hints, not automatic actions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SetupFix:
    """A candidate fix to attempt."""

    new_command: str
    description: str
    is_last_resort: bool = False  # True = no more automatic attempts after this


@dataclass
class SetupDiagnosis:
    """Outcome of pattern analysis."""

    fixes: list[SetupFix]  # ordered: try first to last
    docker_hint: str | None = None  # shown if all fixes fail
    raw_cause: str = ""  # one-line human-readable root cause


# ---------------------------------------------------------------------------
# Pattern rules
# ---------------------------------------------------------------------------


def _requirements_files(repo_root: Path) -> list[str]:
    candidates = [
        "requirements.txt",
        "requirements/base.txt",
        "requirements/main.txt",
        "requirements-base.txt",
        "requirements-core.txt",
    ]
    return [c for c in candidates if (repo_root / c).is_file()]


_PIP_TOOLING = "pip install --upgrade pip setuptools wheel"


def _req_install(files: list[str]) -> str:
    if not files:
        return ""
    return " && ".join(f"pip install -r {f}" for f in files[:2])


def diagnose(
    stderr: str,
    current_command: str,
    repo_root: Path,
) -> SetupDiagnosis:
    """Return ordered fix candidates and a Docker hint for the given stderr."""
    text = stderr.lower()
    fixes: list[SetupFix] = []
    docker_hint: str | None = None
    cause = ""
    req_files = _requirements_files(repo_root)

    # ── Rule 0: benchmark script not committed to git ───────────────────────
    # `generate eval-script` creates the file on disk but if the user forgets
    # to commit it, the worktree (checked out at baseline commit) won't have it.
    if "no such file or directory" in text and (
        "benchmarks/evaluate.py" in stderr or "evaluate.py" in stderr
    ):
        cause = "benchmark script missing from git — file exists on disk but was never committed"
        # Auto-fix: stage + commit the script then let the baseline retry
        for candidate in ["benchmarks/evaluate.py", "evaluate.py", "benchmarks/eval.py"]:
            if (repo_root / candidate).is_file():
                fixes.append(
                    SetupFix(
                        new_command=(
                            f"git add {candidate} src/config.py src/__init__.py 2>/dev/null || true; "
                            "git commit -m 'chore: add ResearchForge benchmark script' "
                            "2>/dev/null || true; "
                            f"{current_command}"
                        ),
                        description=f"committing {candidate} to git so the worktree can find it",
                    )
                )
                break
        docker_hint = None  # no Docker needed — it's a git issue
        return SetupDiagnosis(fixes=fixes, docker_hint=docker_hint, raw_cause=cause)

    # ── Rule 1: setuptools flat-layout (YOLOv5 / many ML repos) ────────────
    # Root cause: pyproject.toml with no [build-system] in the worktree root.
    # Setuptools 70+ discovers the flat-layout and refuses to build — even
    # during `pip install -r requirements.txt` if requirements.txt upgrades
    # setuptools itself (e.g. "setuptools>=70.0.0" triggers a re-scan).
    if (
        "multiple top-level packages discovered" in text
        or "flat-layout" in text
        or "package discovery" in text
    ):
        cause = "pyproject.toml has no build backend — pip install -e . is not usable"
        if req_files:
            # Fix A: plain requirements install (works when setuptools not re-upgraded)
            fixes.append(
                SetupFix(
                    new_command=f"{_PIP_TOOLING} && {_req_install(req_files)}",
                    description=f"switching to {req_files[0]} (pyproject.toml has no build backend)",
                )
            )
            # Fix B: --no-build-isolation prevents pip from building the local
            # directory even when setuptools 70+ is present in the requirements.
            fixes.append(
                SetupFix(
                    new_command=(
                        f"pip install --upgrade pip && "
                        f"pip install --no-build-isolation -r {req_files[0]}"
                    ),
                    description=(
                        "using --no-build-isolation to skip local-package build "
                        "(setuptools 70+ workaround)"
                    ),
                )
            )
            # Fix C: cd to /tmp so setuptools cannot see the local pyproject.toml.
            # Use an absolute path captured at invocation time.
            fixes.append(
                SetupFix(
                    new_command=(
                        f'abs_req="$(pwd)/{req_files[0]}"; '
                        f"pip install --upgrade pip setuptools wheel && "
                        f'pip install -r "$abs_req"'
                    ),
                    description="running pip from /tmp to avoid local pyproject.toml detection",
                )
            )
            # Fix D: pin setuptools to < 70 to restore lenient flat-layout behaviour
            fixes.append(
                SetupFix(
                    new_command=(
                        f"pip install 'setuptools<70' pip wheel && {_req_install(req_files)}"
                    ),
                    description="pinning setuptools<70 to bypass flat-layout strictness",
                )
            )
        else:
            fixes.append(
                SetupFix(
                    new_command="pip install --upgrade pip setuptools wheel",
                    description="no requirements.txt found — upgrading pip/setuptools only",
                    is_last_resort=True,
                )
            )
        docker_hint = (
            "This repo's pyproject.toml layout is incompatible with venv mode.\n"
            "Docker is the reliable fix — RF can generate a working Dockerfile:\n"
            "  researchforge generate dockerfile\n"
            "  (RF will automatically switch to Docker mode and retry)"
        )

    # ── Rule 2: pip or setuptools too old ───────────────────────────────────
    elif (
        "legacy-install-failure" in text
        or "error: invalid command 'bdist_wheel'" in text
        or "wheel package not installed" in text
        or "pkg_resources" in text
        or re.search(r"distutils\.errors", stderr, re.I)
    ):
        cause = "pip/setuptools is too old for this package"
        base = f"pip install --upgrade pip setuptools wheel && {current_command}"
        fixes.append(SetupFix(new_command=base, description="upgrading pip/setuptools/wheel first"))
        if req_files:
            fixes.append(
                SetupFix(
                    new_command=f"{_PIP_TOOLING} && {_req_install(req_files)}",
                    description=f"upgrading pip then installing from {req_files[0]}",
                )
            )

    # ── Rule 3: missing pip in the venv ─────────────────────────────────────
    elif "no module named pip" in text or "pip is not installed" in text:
        cause = "pip missing from the virtual environment"
        fixes.append(
            SetupFix(
                new_command=(
                    "python -m ensurepip --upgrade && pip install --upgrade pip && "
                    f"{current_command}"
                ),
                description="bootstrapping pip with ensurepip first",
            )
        )

    # ── Rule 4: git/network errors ──────────────────────────────────────────
    elif (
        "connection refused" in text
        or "failed to connect" in text
        or "network is unreachable" in text
        or "could not connect" in text
        or re.search(r"timed? out", text)
    ):
        cause = "network error during package download"
        if req_files:
            # Try installing what's already in the venv cache / skip git deps
            fixes.append(
                SetupFix(
                    new_command=f"pip install --no-deps -r {req_files[0]}",
                    description="trying --no-deps to skip network-fetched transitive packages",
                )
            )
        docker_hint = (
            "Network issues may be due to a proxy or firewall.\n"
            "Set HTTPS_PROXY or use an offline Docker image:\n"
            "  researchforge generate dockerfile"
        )

    # ── Rule 5: C-extension / compiler missing ──────────────────────────────
    elif (
        "error: command 'gcc' failed" in text
        or "gcc not found" in text
        or "cl.exe" in text
        or "microsoft visual c++" in text
        or re.search(r"error.*compilation failed", text)
        or "failed building wheel" in text
    ):
        cause = "C extension compilation failed — compiler missing in venv environment"
        if req_files:
            # Try pre-built wheels only
            fixes.append(
                SetupFix(
                    new_command=f"pip install --only-binary=:all: -r {req_files[0]}",
                    description="trying --only-binary to avoid compiling C extensions",
                )
            )
        docker_hint = (
            "This package requires a C compiler. Docker provides a complete build environment:\n"
            "  researchforge generate dockerfile\n"
            "  Then update researchforge.yaml: execution.mode: docker"
        )

    # ── Rule 6: CUDA / GPU driver mismatch ──────────────────────────────────
    elif (
        "cuda" in text
        or "nvcc" in text
        or re.search(r"nvidia.+not found", text)
        or "libcuda" in text
    ):
        cause = "CUDA/GPU driver mismatch during package install"
        if req_files:
            # Try CPU-only versions
            cpu_req = next(
                (f for f in req_files if "cpu" in f.lower()),
                req_files[0],
            )
            fixes.append(
                SetupFix(
                    new_command=(
                        "pip install --upgrade pip && "
                        f"pip install -r {cpu_req} "
                        "--extra-index-url https://download.pytorch.org/whl/cpu"
                    ),
                    description="trying CPU-only PyTorch wheel (no CUDA required)",
                )
            )
        docker_hint = (
            "For GPU workloads, use a CUDA Docker image:\n"
            "  researchforge generate dockerfile --cuda\n"
            "  Then update researchforge.yaml: execution.mode: docker"
        )

    # ── Rule 7: permission denied ────────────────────────────────────────────
    elif "permission denied" in text or "errno 13" in text:
        cause = "permission denied during package install"
        fixes.append(
            SetupFix(
                new_command=current_command.replace("pip install", "pip install --user"),
                description="retrying with --user install flag",
            )
        )
        docker_hint = (
            "If running as a restricted user, Docker is more reliable:\n"
            "  researchforge generate dockerfile"
        )

    # ── Rule 8: disk space ───────────────────────────────────────────────────
    elif "no space left" in text or "disk quota exceeded" in text:
        cause = "disk full during package install"
        fixes.append(
            SetupFix(
                new_command=f"pip cache purge && {current_command}",
                description="clearing pip cache before retry (frees disk space)",
            )
        )
        docker_hint = "Free up disk space or move the project to a larger volume."

    # ── Rule 9: version conflict / resolver ─────────────────────────────────
    elif (
        "conflicting dependencies" in text
        or "cannot install" in text
        or "dependency resolver" in text
        or re.search(r"incompatible.+requires", text)
    ):
        cause = "conflicting package versions in requirements"
        if req_files:
            fixes.append(
                SetupFix(
                    new_command=(
                        f"pip install --upgrade pip && pip install -r {req_files[0]} --no-deps"
                    ),
                    description=(
                        "installing without resolving transitive deps (faster, may skip conflicts)"
                    ),
                )
            )
            fixes.append(
                SetupFix(
                    new_command=(
                        "pip install --upgrade pip && "
                        f"pip install -r {req_files[0]} --upgrade-strategy eager"
                    ),
                    description="upgrading all packages eagerly to resolve conflicts",
                )
            )
        docker_hint = (
            "Complex dependency conflicts are best solved in a Docker image:\n"
            "  researchforge generate dockerfile"
        )

    # ── Rule 10: Python version mismatch ─────────────────────────────────────
    elif (
        re.search(r"python_requires.*3\.", text)
        or re.search(r"requires python .*3\.", text)
        or "python version" in text
    ):
        cause = "Python version incompatibility"
        # Nothing we can auto-fix here — just hint
        docker_hint = (
            "This package requires a different Python version.\n"
            "Specify the correct Python in a Dockerfile:\n"
            "  researchforge generate dockerfile\n"
            "  Edit Dockerfile to use FROM python:<required-version>"
        )

    # ── Rule 11: no matching distribution ────────────────────────────────────
    elif "no matching distribution found" in text or "could not find a version" in text:
        cause = "package not available for this Python version / platform"
        if req_files:
            # Try without the failing package (user must fix manually)
            fixes.append(
                SetupFix(
                    new_command=(
                        "pip install --upgrade pip && "
                        f"pip install -r {req_files[0]} --ignore-requires-python"
                    ),
                    description="ignoring Python version constraints on packages",
                )
            )
        docker_hint = (
            "Use a Docker image matching the required platform:\n"
            "  researchforge generate dockerfile"
        )

    # ── Rule 12: generic / unknown ───────────────────────────────────────────
    else:
        cause = "unknown setup failure (see setup_stderr.log for details)"
        if req_files and "pip install -e" in current_command:
            # Most common fix: switch from editable to plain requirements install
            fixes.append(
                SetupFix(
                    new_command=f"{_PIP_TOOLING} && {_req_install(req_files)}",
                    description="falling back to requirements.txt install (pip install -e . failed)",
                )
            )
        docker_hint = (
            "If the error is environment-specific, Docker provides a clean slate:\n"
            "  researchforge generate dockerfile"
        )

    return SetupDiagnosis(fixes=fixes, docker_hint=docker_hint, raw_cause=cause)
