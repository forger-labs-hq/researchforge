"""Pull and measure the best public notebook as the experiment baseline.

For Kaggle GPU / Competition GPU mode:
  - Pull kernel source code via `kaggle kernels pull`
  - Record the public LB score from the kernel title as baseline (honest)
  - The pulled source becomes the starting point for experiment patches

For CPU mode:
  - Pull kernel source
  - Run it locally and measure the actual CV score
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from researchforge.kaggle.api import (
    KaggleAPIError,
    PublicKernel,
    list_competition_kernels,
    pull_kernel_source,
)


@dataclass
class BaselineResult:
    """Result of baseline measurement."""

    kernel_ref: str
    kernel_title: str
    cv_score: float | None  # None = could not measure
    baseline_metrics: dict[str, Any]
    status: str  # "success", "success_lb_score", "failed"


def find_best_public_kernel(
    competition_slug: str,
    sort_by: str = "scoreDescending",
) -> PublicKernel | None:
    """Return the top public kernel for the competition, or None."""
    try:
        kernels = list_competition_kernels(competition_slug, sort_by=sort_by, page_size=20)
        return kernels[0] if kernels else None
    except KaggleAPIError:
        return None


def run_baseline_on_kernel(
    kernel_ref: str,
    competition_slug: str,
    gpu_backend: str,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
) -> BaselineResult:
    """Establish the baseline from the best public kernel.

    For Kaggle/Competition GPU:
      Pulls the kernel source code to use as the starting point for patches.
      Records the public LB score extracted from the kernel title as the
      baseline metric. This is honest: we label it "lb_score" not "cv_score".

    For CPU:
      Pulls the kernel source and runs it locally to measure a real CV score.
      Falls back to LB score extraction if local run fails.
    """
    if artifacts_dir is None:
        artifacts_dir = Path.cwd() / f".overnight_{competition_slug}"
    if data_dir is None:
        data_dir = artifacts_dir / "data"

    baseline_dir = artifacts_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Pull kernel source so we have real code to patch ------
    source_dir = baseline_dir / "source"
    try:
        pull_kernel_source(kernel_ref, source_dir)
    except KaggleAPIError as exc:
        # Pull failed — continue without source code (patching will create from scratch)
        (baseline_dir / "pull_error.txt").write_text(str(exc))

    # --- Step 2: Get kernel title for LB score extraction ---------------
    kernel_title = ""
    try:
        kernels = list_competition_kernels(competition_slug, page_size=50)
        for k in kernels:
            if k.ref == kernel_ref or kernel_ref.split("/")[-1] in k.ref:
                kernel_title = k.title
                break
    except KaggleAPIError:
        pass

    lb_score = _extract_lb_score(kernel_title)

    # --- Step 3: For CPU mode, try running locally ----------------------
    if gpu_backend == "cpu" and data_dir.exists():
        local_score = _run_locally(source_dir, data_dir, artifacts_dir)
        if local_score is not None:
            _save_baseline_meta(baseline_dir, kernel_ref, kernel_title, local_score, "local_cv")
            return BaselineResult(
                kernel_ref=kernel_ref,
                kernel_title=kernel_title,
                cv_score=local_score,
                baseline_metrics={"source": "local_run", "method": "cv_score"},
                status="success",
            )

    # --- Step 4: Fall back to LB score (Kaggle GPU mode or local run failed)
    if lb_score is not None:
        _save_baseline_meta(baseline_dir, kernel_ref, kernel_title, lb_score, "lb_score")
        return BaselineResult(
            kernel_ref=kernel_ref,
            kernel_title=kernel_title,
            cv_score=lb_score,
            baseline_metrics={"source": "kernel_title", "method": "lb_score_extraction"},
            status="success_lb_score",
        )

    # --- Step 5: No score available — return None so caller shows warning
    _save_baseline_meta(baseline_dir, kernel_ref, kernel_title, None, "not_measured")
    return BaselineResult(
        kernel_ref=kernel_ref,
        kernel_title=kernel_title,
        cv_score=None,
        baseline_metrics={"source": "none", "reason": "no score in title, local run skipped"},
        status="no_baseline",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CLAIM_PATTERNS = [
    # Explicit labels: "LB 0.480", "Public LB: 0.47", "score=0.95", "AUC 0.923", "MRR@25 0.41"
    r"(?i)\b(?:public\s+)?(?:LB|leaderboard|score|AUC|RMSE|MAE|F1|MRR(?:@\d+)?|mAP|acc(?:uracy)?)"
    r"\s*[:=\-]?\s*([0-9]+\.[0-9]+)",  # a decimal point is required: "lb472" is ambiguous
    # Bracketed: "[0.468] CASMI…", "(0.9487)"
    r"[\[(]\s*([0-9]+\.[0-9]{2,})\s*[\])]",
    # A bare decimal score in a title or slug-like title: "casmi26-0.482", "0.468 baseline"
    r"(?<![\w.])(?<!v)(?<!V)([0-9]{1,3}\.[0-9]{2,})(?![\w.])",
]


def claimed_lb_score(title: str) -> float | None:
    """The leaderboard score a public notebook claims in its title, as written.

    Kaggle's API exposes no score for kernels, so authors put it in the title
    ("CASMI26 LB 0.480", "[0.468] …", "casmi26-0.482"). This is a *claim*, not a
    measurement, and it is returned unscaled — compare it only with leaderboard
    scores on the same metric.
    """
    for pat in _CLAIM_PATTERNS:
        for m in re.finditer(pat, title or ""):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if 0.0 < val <= 1000.0:
                return val
    return None


def _extract_lb_score(title: str) -> float | None:
    """The score a kernel title claims, read as written.

    It used to divide anything between 1 and 100 by 100 as a "percentage", which
    turned a packing score of 22.25 into 0.2225. Scores are compared with other
    scores on the same metric, so they are never rescaled.
    """
    return claimed_lb_score(title)


def _run_locally(
    source_dir: Path,
    data_dir: Path,
    artifacts_dir: Path,
) -> float | None:
    """Try to run the kernel script locally and extract a CV score.

    Returns the metric value or None if execution fails.
    """
    import os
    import sys
    import tempfile

    from researchforge.execution.runner import SubprocessRunner

    # Find the main Python script in the pulled source
    scripts = list(source_dir.glob("*.py")) if source_dir.exists() else []
    if not scripts:
        return None

    script = scripts[0]
    runner = SubprocessRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        stdout_path = tmp / "stdout.txt"
        stderr_path = tmp / "stderr.txt"
        try:
            outcome = runner.run(
                [sys.executable, str(script)],
                cwd=data_dir,
                env=os.environ.copy(),
                timeout_seconds=1800,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        except Exception:
            return None

        if not outcome.ok:
            return None

        combined = (stdout_path.read_text() if stdout_path.exists() else "") + (
            stderr_path.read_text() if stderr_path.exists() else ""
        )
        return _parse_metric_from_output(combined)


def _parse_metric_from_output(text: str) -> float | None:
    """Extract the first recognisable metric float from script stdout."""
    patterns = [
        r"(?i)CV Score[:\s=]+([0-9]+\.[0-9]+)",
        r"(?i)Metric[:\s=]+([0-9]+\.[0-9]+)",
        r"(?i)AUC[:\s=]+([0-9]+\.[0-9]+)",
        r"(?i)RMSE[:\s=]+([0-9]+\.[0-9]+)",
        r"(?i)Score[:\s=]+([0-9]+\.[0-9]+)",
        r"([0-9]+\.[0-9]{4,})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _save_baseline_meta(
    baseline_dir: Path,
    kernel_ref: str,
    kernel_title: str,
    score: float | None,
    method: str,
) -> None:
    with (baseline_dir / "baseline_meta.json").open("w") as fh:
        json.dump(
            {
                "kernel_ref": kernel_ref,
                "kernel_title": kernel_title,
                "cv_score": score,
                "method": method,
            },
            fh,
            indent=2,
        )
