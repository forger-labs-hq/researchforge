"""Rounds whose Kaggle kernel outlived the round's wait, collected when it finishes.

execute-round waits a fixed time for a Kaggle kernel. A heavier notebook keeps
running after that, so the round is recorded with ``pending: true`` in
round_outputs.json instead of being lost as a failure. The monitor calls
:func:`collect` between its cycles: a finished kernel's score and submission file
replace the pending record, and an improved round is offered for submission.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from researchforge.kaggle.workspace import read_json

_STILL_RUNNING = ("running", "queued", "unknown")
# Kaggle says 404 for a notebook it no longer has, and briefly for one just pushed; a
# pending round this old whose notebook is gone is closed instead of waited on forever.
MISSING_AFTER_HOURS = 12


def pending_rounds(artifacts_dir: Path) -> dict[int, dict[str, object]]:
    from researchforge.kaggle.submission import OUTPUTS_FILE

    data = read_json(artifacts_dir / OUTPUTS_FILE, {}) or {}
    data = data if isinstance(data, dict) else {}
    return {
        int(k): v
        for k, v in data.items()
        if str(k).isdigit() and isinstance(v, dict) and v.get("pending") and v.get("kernel_ref")
    }


def collect(artifacts_dir: Path) -> list[str]:
    """Collect every pending round whose kernel has finished. Returns a line per round."""
    from researchforge.kaggle.api import get_kernel_run_status

    lines = []
    for round_num, out in sorted(pending_rounds(artifacts_dir).items()):
        ref = str(out["kernel_ref"])
        status, _ = get_kernel_run_status(ref)
        if (
            status == "missing"
            and _hours_since_start(artifacts_dir, round_num) < MISSING_AFTER_HOURS
        ):
            continue
        if status in _STILL_RUNNING:
            continue
        lines.append(_finish(artifacts_dir, round_num, ref, out.get("kernel_version"), status))
    return lines


def _hours_since_start(artifacts_dir: Path, round_num: int) -> float:
    from datetime import datetime

    with sqlite3.connect(artifacts_dir / "experiment_log.db") as conn:
        row = conn.execute(
            "SELECT started_at FROM experiments WHERE round_num=?", (round_num,)
        ).fetchone()
    try:
        started = datetime.fromisoformat(str(row[0])) if row and row[0] else None
    except ValueError:
        started = None
    if started is None:
        return 0.0  # unknown age: keep waiting rather than close a round too early
    return (datetime.now() - started.replace(tzinfo=None)).total_seconds() / 3600


def _finish(artifacts_dir: Path, round_num: int, ref: str, version: object, status: str) -> str:
    from researchforge.kaggle import control
    from researchforge.kaggle.benchmark_runner import BenchmarkRunner
    from researchforge.kaggle.experiment_loop import BenchmarkResult
    from researchforge.kaggle.hypothesis_pool import set_round_outcome
    from researchforge.kaggle.orchestrator import OvernightSetup
    from researchforge.kaggle.submission import record_round_outputs

    setup = OvernightSetup.load(artifacts_dir / "overnight_setup.json")
    kernel_version = (
        int(version) if isinstance(version, int | str) and str(version).isdigit() else None
    )
    if status == "complete":
        result = BenchmarkRunner(setup).collect_kernel(ref, kernel_version, 0.0)
    else:
        why = (
            f"Kaggle no longer has the notebook {ref} (404): it was deleted or never "
            "created, so its result cannot be collected"
            if status == "missing"
            else f"the kernel {ref} ended on Kaggle with status: {status}"
        )
        result = BenchmarkResult(
            passed=False,
            metric_value=0.0,
            metric_improved=False,
            duration_seconds=0.0,
            error_message=why,
        )
    outputs = dict(result.outputs or {"kernel_ref": ref, "kernel_version": kernel_version})
    record_round_outputs(artifacts_dir, round_num, outputs)
    with sqlite3.connect(artifacts_dir / "experiment_log.db") as conn:
        conn.execute(
            "UPDATE experiments SET full_passed=?, full_metric=?, full_improved=?, "
            "rejected_reason=? WHERE round_num=?",
            (
                int(result.passed),
                result.metric_value if result.passed else None,
                int(result.metric_improved),
                None if result.passed else (result.error_message or "")[:500],
                round_num,
            ),
        )
    outcome = (
        "improved" if result.metric_improved else ("no_improvement" if result.passed else "failed")
    )
    set_round_outcome(artifacts_dir, round_num, outcome)
    title = _round_title(artifacts_dir, round_num)
    if not result.passed:
        line = f"R{round_num} finished on Kaggle without a score: {result.error_message}"
        control.notify(artifacts_dir, "alert", f"⚠️ {line}")
        return line
    line = f"R{round_num} '{title}' finished on Kaggle: {result.metric_value:.4f}" + (
        " — improved" if result.metric_improved else " — no improvement"
    )
    control.notify(artifacts_dir, "improvement" if result.metric_improved else "kernel", line)
    if result.metric_improved:
        target = setup.selected_gpu.value if setup.selected_gpu else "kaggle_gpu"
        control.offer_submission(artifacts_dir, round_num, title, result.metric_value, target)
    return line


def _round_title(artifacts_dir: Path, round_num: int) -> str:
    with sqlite3.connect(artifacts_dir / "experiment_log.db") as conn:
        row = conn.execute(
            "SELECT patch_description FROM experiments WHERE round_num=?", (round_num,)
        ).fetchone()
    return str(row[0]) if row and row[0] else f"Round {round_num}"
