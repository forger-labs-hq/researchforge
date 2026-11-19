"""Kaggle API wrapper — competition data, kernel management, GPU execution.

Covers:
  - List public kernels
  - Download competition data (CPU mode only)
  - Pull kernel source code
  - Push kernel to Kaggle servers (Kaggle GPU mode)
  - Poll kernel status until complete
  - Fetch kernel output metrics
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class KaggleAPIError(Exception):
    """Kaggle API call failed; message is user-facing."""


@dataclass
class PublicKernel:
    """Metadata for a public kernel on Kaggle."""

    ref: str  # username/notebook-slug
    title: str
    author: str
    last_run_time: str
    total_votes: int


# ---------------------------------------------------------------------------
# Kernel listing
# ---------------------------------------------------------------------------


def list_competition_kernels(
    competition_slug: str,
    sort_by: str = "scoreDescending",
    page_size: int = 20,
) -> list[PublicKernel]:
    """List public kernels for a competition, sorted by preference."""
    try:
        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "list",
                "--competition",
                competition_slug,
                "--sort-by",
                sort_by,
                "--page-size",
                str(page_size),
                "--csv",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise KaggleAPIError(f"Failed to list kernels: {result.stderr.strip()}")

        return _parse_kernel_csv(result.stdout)
    except subprocess.TimeoutExpired as exc:
        raise KaggleAPIError("Kernel listing timed out") from exc


def _parse_kernel_csv(stdout: str) -> list[PublicKernel]:
    """Parse `kaggle kernels list --csv`. Titles may hold quoted commas; columns go by name."""
    lines = [
        ln
        for ln in stdout.splitlines()
        if ln.strip() and not ln.strip().lower().startswith("warning")
    ]
    kernels: list[PublicKernel] = []
    for row in csv.DictReader(lines):
        ref = (row.get("ref") or "").strip()
        if not ref:
            continue
        try:
            votes = int(row.get("totalVotes") or 0)
        except ValueError:
            votes = 0
        kernels.append(
            PublicKernel(
                ref=ref,
                title=(row.get("title") or "").strip(),
                author=(row.get("author") or "").strip(),
                last_run_time=(row.get("lastRunTime") or "").strip(),
                total_votes=votes,
            )
        )
    return kernels


# ---------------------------------------------------------------------------
# Competition data download (CPU mode only)
# ---------------------------------------------------------------------------


def leaderboard_direction(slug: str) -> str | None:
    """Which way the metric improves, read from the public leaderboard's order.

    Rank 1 is the best score, so scores that grow down the board mean lower is
    better. Returns "maximize", "minimize", or None when the board is missing or
    every score ties (a solved competition, say) and the order says nothing.
    """
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "leaderboard", slug, "--show", "--csv"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    return _direction_from_leaderboard_csv(result.stdout)


def _direction_from_leaderboard_csv(stdout: str) -> str | None:
    start = next((i for i, ln in enumerate(stdout.splitlines()) if ln.startswith("teamId,")), None)
    if start is None:
        return None
    scores: list[float] = []
    for row in csv.DictReader(stdout.splitlines()[start:]):
        try:
            scores.append(float(row.get("score") or ""))
        except ValueError:
            continue
    rising = sum(b > a for a, b in zip(scores, scores[1:], strict=False))
    falling = sum(b < a for a, b in zip(scores, scores[1:], strict=False))
    if rising == falling:
        return None
    return "minimize" if rising > falling else "maximize"


def download_competition_data(
    competition_slug: str,
    output_dir: Path,
    quiet: bool = False,
) -> bool:
    """Download competition data. Only call this for CPU mode.

    For Kaggle GPU / Competition GPU the data lives on Kaggle's servers —
    never download it locally.
    """
    import zipfile

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition_slug,
            "-p",
            str(output_dir),
        ]
        if quiet:
            cmd.append("-q")
        result = subprocess.run(cmd, capture_output=quiet, text=True, timeout=300, check=False)
        if result.returncode != 0:
            return False
        # Extract any zip files that were downloaded
        for zp in output_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(output_dir)
                zp.unlink()  # remove zip after extraction
            except (zipfile.BadZipFile, OSError):
                pass  # leave zip in place if extraction fails
        return True
    except subprocess.TimeoutExpired:
        return False


# ---------------------------------------------------------------------------
# Kernel source pull
# ---------------------------------------------------------------------------


def pull_kernel_source(kernel_ref: str, output_dir: Path) -> Path:
    """Download a kernel's source code from Kaggle.

    Args:
        kernel_ref: e.g. "username/kernel-slug"
        output_dir: Directory to write the source files into.

    Returns:
        Path to the directory containing the pulled files.

    Raises:
        KaggleAPIError: If the pull fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "pull", kernel_ref, "-p", str(output_dir), "--metadata"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise KaggleAPIError(f"Failed to pull kernel {kernel_ref}: {result.stderr.strip()}")
    return output_dir


# ---------------------------------------------------------------------------
# Kernel push → poll → output  (Kaggle GPU execution path)
# ---------------------------------------------------------------------------


def get_kaggle_username() -> str:
    """Read the configured Kaggle username from kaggle.json or env."""
    import os

    # Env var takes priority
    username = os.environ.get("KAGGLE_USERNAME", "")
    if username:
        return username
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    if creds_path.exists():
        try:
            return str(json.loads(creds_path.read_text()).get("username", ""))
        except (json.JSONDecodeError, OSError):
            pass
    raise KaggleAPIError(
        "Cannot determine Kaggle username. Set KAGGLE_USERNAME env var or "
        "ensure ~/.kaggle/kaggle.json contains a 'username' field."
    )


# What a round can attach to its Kaggle notebook, named as `kaggle kernels push` names them:
# datasets, other notebooks' outputs, Kaggle Models, the Docker image and the GPU type.
KERNEL_INPUT_LISTS = ("dataset_sources", "kernel_sources", "model_sources")
KERNEL_INPUT_VALUES = ("docker_image", "machine_shape")


def kernel_inputs(source: dict[str, Any]) -> dict[str, Any]:
    """The notebook inputs in a proposal or a pulled notebook's metadata, deduplicated.

    Only the fields that are set are returned, so ``{}`` means competition data only.
    """
    out: dict[str, Any] = {}
    for key in KERNEL_INPUT_LISTS:
        refs: list[str] = []
        for ref in source.get(key) or []:
            ref = str(ref).strip()
            if ref and ref not in refs:
                refs.append(ref)
        if refs:
            out[key] = refs
    for key in KERNEL_INPUT_VALUES:
        value = str(source.get(key) or "").strip()
        if value:
            out[key] = value
    return out


def build_kernel_metadata(
    username: str,
    kernel_slug: str,
    script_filename: str,
    competition_slug: str,
    enable_gpu: bool = True,
    enable_internet: bool = False,
    title: str = "",
    dataset_sources: list[str] | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the kernel-metadata.json dict required by `kaggle kernels push`.

    Args:
        dataset_sources: Optional list of Kaggle dataset refs ("username/dataset-slug")
            to attach to the kernel. Leave None (or []) for competition data only.
        inputs: Further notebook inputs (see ``kernel_inputs``): other notebooks'
            outputs, Kaggle Models, a Docker image and a GPU type, as a public
            notebook ported into a round needs them.
    """
    extra = kernel_inputs(
        {
            **(inputs or {}),
            "dataset_sources": [
                *(dataset_sources or []),
                *((inputs or {}).get("dataset_sources") or []),
            ],
        }
    )
    meta: dict[str, Any] = {
        "id": f"{username}/{kernel_slug}",
        "title": title or kernel_slug.replace("-", " ").title(),
        "code_file": script_filename,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "dataset_sources": extra.get("dataset_sources", []),
        "competition_sources": [competition_slug],
        "kernel_sources": extra.get("kernel_sources", []),
        "model_sources": extra.get("model_sources", []),
    }
    for key in KERNEL_INPUT_VALUES:
        if key in extra:
            meta[key] = extra[key]
    return meta


def push_kernel(kernel_dir: Path, timeout_seconds: int = 120) -> str:
    return push_kernel_versioned(kernel_dir, timeout_seconds)[0]


def push_kernel_versioned(kernel_dir: Path, timeout_seconds: int = 120) -> tuple[str, int | None]:
    """Push a kernel directory to Kaggle.

    The directory must contain:
      - A Python script (the code_file referenced in kernel-metadata.json)
      - kernel-metadata.json

    Returns:
        The kernel ref ("username/slug") from the metadata, and the version Kaggle
        reports for this push (None when its reply names none).

    Raises:
        KaggleAPIError: If the push fails.
    """
    meta_path = kernel_dir / "kernel-metadata.json"
    if not meta_path.exists():
        raise KaggleAPIError(f"kernel-metadata.json not found in {kernel_dir}")
    kernel_ref = json.loads(meta_path.read_text()).get("id", "")

    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise KaggleAPIError(f"Kernel push failed for {kernel_ref}:\n{result.stderr.strip()}")
    m = re.search(r"version (\d+)", result.stdout, re.I)
    return str(kernel_ref), int(m.group(1)) if m else None


def get_kernel_run_status(kernel_ref: str) -> tuple[str, str]:
    """Return (status, raw_output) for a kernel.

    Status values: "complete", "error", "running", "queued", "cancelacknowledged",
    "missing" (Kaggle answers 404: no such notebook, which is also what it says for a
    moment right after a push), "unknown".
    raw_output is the full CLI stdout — kept so callers can surface it in timeout errors.
    """
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_ref],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        raw = result.stdout.strip() or result.stderr.strip()
        if result.returncode != 0:
            return (
                "missing" if "404" in result.stderr and "Not Found" in result.stderr else "unknown",
                f"exit={result.returncode} stdout={result.stdout[:200]} "
                f"stderr={result.stderr[:200]}",
            )
        output = result.stdout.lower()
        for status in ("complete", "error", "running", "queued", "cancelacknowledged"):
            if status in output:
                return status, raw
        return "unknown", raw
    except subprocess.TimeoutExpired:
        return "unknown", "kaggle status command timed out after 15s"


def poll_kernel_until_complete(
    kernel_ref: str,
    timeout_seconds: float = 7200,
    poll_interval_seconds: int = 30,
) -> str:
    """Poll kernel status every poll_interval_seconds until done or timed out.

    Returns:
        Final status string: "complete", "error", or "timeout:<last_raw_output>"
        The last raw output is embedded in the timeout return value so callers
        can surface it in error messages instead of discarding it.
    """
    deadline = time.monotonic() + timeout_seconds
    last_raw = "(no polls completed)"
    poll_count = 0
    while time.monotonic() < deadline:
        status, raw = get_kernel_run_status(kernel_ref)
        last_raw = raw
        poll_count += 1
        if status in ("complete", "error", "cancelacknowledged"):
            return status
        remaining = deadline - time.monotonic()
        sleep_time = min(poll_interval_seconds, max(0, remaining))
        if sleep_time <= 0:
            break
        time.sleep(sleep_time)
    return f"timeout (after {poll_count} polls, last raw: {last_raw[:200]})"


def fetch_kernel_output(
    kernel_ref: str,
    output_dir: Path,
    force: bool = True,
    file_pattern: str | None = None,
    timeout: float = 600,
) -> list[Path]:
    """Download output files from a completed kernel run.

    Args:
        kernel_ref: "username/slug"
        output_dir: Directory to write output files.
        force: Overwrite existing files.
        file_pattern: Regex on file names; only matching files are downloaded. The
            run's log always comes too. A notebook can leave hundreds of files
            (caches, models) that a round never needs.
        timeout: Seconds before giving up; whatever arrived by then is returned.

    Returns:
        List of downloaded file paths (may be empty if kernel produced no output).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir), "-q"]
    if force:
        cmd.append("-o")
    if file_pattern:
        cmd += ["--file-pattern", file_pattern]
    started = time.time() - 2

    def written() -> list[Path]:
        # Only this download's files: an earlier, interrupted one may have left others.
        return [p for p in output_dir.iterdir() if p.is_file() and p.stat().st_mtime >= started]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return written()  # a slow download must not lose the run: keep what came through
    if result.returncode != 0:
        # Non-zero may mean "no output files" — not necessarily an error
        return []
    return written()
