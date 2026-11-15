"""Where a round's code runs, told to whoever writes the proposal, and what it may not do there.

``prepare-round`` hands the proposer :func:`brief` so a round is written for the compute
the person chose. With Colab chosen, a round is the whole experiment on the Colab VM:
a round that only pushes a Kaggle notebook and waits on it spends Colab time waiting
and its Kaggle run is never collected, so :func:`refusal` stops it before it starts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import read_json

# `kaggle kernels push`, in a shell line, an argv list or the Python API.
_PUSHES_KERNEL = re.compile(r"kernels\W{0,6}push|kernels_push", re.I)


def _target(artifacts_dir: Path) -> str:
    setup = read_json(artifacts_dir / "overnight_setup.json", {}) or {}
    return str(setup.get("selected_gpu") or "cpu") if isinstance(setup, dict) else "cpu"


def brief(artifacts_dir: Path) -> dict[str, Any]:
    """What the proposer must know about where this round runs."""
    from researchforge.kaggle.run_config import round_time_limit_seconds

    target = _target(artifacts_dir)
    minutes = int(round_time_limit_seconds(artifacts_dir) // 60)
    rules = [
        "Print the metric on its own line (e.g. 'CV Score: 0.8123').",
        "Write the submission as submission.csv or submission.zip in the working folder, "
        "or print 'SUBMISSION_FILE: <path>', so the person can approve submitting it.",
    ]
    if target == "colab_gpu":
        from researchforge.kaggle import colab

        cfg = colab.ColabConfig.load(artifacts_dir)
        plan = colab.PLAN_LABELS.get(cfg.plan, cfg.plan)
        runs_on = (
            f"a fresh Google Colab VM ({cfg.gpu or 'CPU'}"
            f"{', high-RAM' if cfg.high_mem else ''}, {plan}), "
            f"as one Python script in {colab.REMOTE_DATA} with the competition data already "
            "downloaded there. The VM is released after the round."
        )
        rules.insert(
            0,
            "Colab only: do the whole experiment on this VM — install packages, load or serve "
            "the model on its GPU, run and score there. Never push, run or wait on a Kaggle "
            "notebook (`kaggle kernels push`) from a round: execute-round refuses it. Reading "
            "public notebooks or datasets is fine.",
        )
    elif target in ("kaggle_gpu", "competition_gpu"):
        runs_on = "a Kaggle notebook on Kaggle's GPU, with the competition data attached."
    elif target == "ssh_server":
        runs_on = "the person's SSH server, in the folder ssh-setup configured."
    elif target == "cloud_gpu":
        runs_on = "a rented cloud GPU instance, created for the round and terminated after."
    else:
        runs_on = "this machine, in the competition folder."
    return {
        "target": target,
        "runs_on": runs_on,
        "time_limit_minutes": minutes,
        "rules": rules,
        "note": (
            f"The round's code runs on {runs_on} It is stopped after {minutes} minutes "
            "(Run configs → Round time limit)."
        ),
    }


def refusal(artifacts_dir: Path, code: str) -> str:
    """Why this round's code may not run on the chosen compute, or ""."""
    if _target(artifacts_dir) == "colab_gpu" and _PUSHES_KERNEL.search(code or ""):
        return (
            "this project runs on Colab only, and the round pushes a Kaggle notebook "
            "(`kaggle kernels push`). Rewrite it to run the experiment on the Colab VM itself."
        )
    return ""
