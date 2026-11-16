"""A ported public notebook takes everything it attaches into its Kaggle round.

Top public notebooks read other notebooks' outputs and Kaggle Models, and pin a Docker
image and GPU type. A round used to attach only the datasets, so such a notebook
stopped at its first missing input and spent Kaggle GPU quota for nothing.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchforge.kaggle import api
from researchforge.kaggle.api import build_kernel_metadata, kernel_inputs

NOTEBOOK_META = {
    "id": "alexchilton/name-that-llm-0-701",
    "dataset_sources": ["a/prompts", "a/prompts"],
    "kernel_sources": ["lucasboesen/name-that-llm-glm4-9b-trunc25"],
    "model_sources": ["qwen-lm/qwen3/transformers/8b/1"],
    "docker_image": "gcr.io/kaggle-gpu-images/python@sha256:abc",
    "machine_shape": "NvidiaTeslaT4",
}


def test_kernel_inputs_keep_every_kind_once() -> None:
    assert kernel_inputs(NOTEBOOK_META) == {
        "dataset_sources": ["a/prompts"],
        "kernel_sources": ["lucasboesen/name-that-llm-glm4-9b-trunc25"],
        "model_sources": ["qwen-lm/qwen3/transformers/8b/1"],
        "docker_image": "gcr.io/kaggle-gpu-images/python@sha256:abc",
        "machine_shape": "NvidiaTeslaT4",
    }
    assert kernel_inputs({"dataset_sources": [], "docker_image": " "}) == {}


def test_the_pushed_metadata_carries_them() -> None:
    meta = build_kernel_metadata(
        "me", "rf-round", "experiment.py", "name-that-llm",
        dataset_sources=["x/extra"], inputs=NOTEBOOK_META,
    )  # fmt: skip
    assert meta["dataset_sources"] == ["x/extra", "a/prompts"]
    assert meta["kernel_sources"] == ["lucasboesen/name-that-llm-glm4-9b-trunc25"]
    assert meta["model_sources"] == ["qwen-lm/qwen3/transformers/8b/1"]
    assert meta["docker_image"] == NOTEBOOK_META["docker_image"]
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["competition_sources"] == ["name-that-llm"]


def test_a_plain_round_stays_competition_data_only() -> None:
    meta = build_kernel_metadata("me", "rf-round", "experiment.py", "name-that-llm")
    assert (meta["dataset_sources"], meta["kernel_sources"], meta["model_sources"]) == ([], [], [])
    assert "docker_image" not in meta and "machine_shape" not in meta


def test_propose_from_kernel_copies_every_input(tmp_path: Path) -> None:
    from researchforge.kaggle.kernel_port import build_proposal

    k = tmp_path / "nb"
    k.mkdir()
    cells = [{"cell_type": "code", "source": ["print('hi')\n"]}]
    (k / "nb.ipynb").write_text(json.dumps({"cells": cells}))
    (k / "kernel-metadata.json").write_text(json.dumps(NOTEBOOK_META))
    proposal, notes = build_proposal(k, "alexchilton/nb", title="Adopt 0.701", append="x = 1")
    assert proposal["dataset_sources"] == ["a/prompts"]
    assert proposal["kernel_sources"] == ["lucasboesen/name-that-llm-glm4-9b-trunc25"]
    assert proposal["model_sources"] == ["qwen-lm/qwen3/transformers/8b/1"]
    assert proposal["machine_shape"] == "NvidiaTeslaT4"
    assert not any("cannot attach" in n for n in notes)


def test_a_round_pushes_its_notebook_with_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle.benchmark_runner import BenchmarkRunner
    from researchforge.kaggle.gpu_detector import GPUBackend

    pushed: list[dict[str, Any]] = []

    def push(kernel_dir: Path, timeout_seconds: int = 120) -> tuple[str, int | None]:
        pushed.append(json.loads((kernel_dir / "kernel-metadata.json").read_text()))
        raise api.KaggleAPIError("stop here")  # what was pushed is all this test needs

    monkeypatch.setattr(api, "get_kaggle_username", lambda: "me")
    monkeypatch.setattr(api, "push_kernel_versioned", push)
    setup = SimpleNamespace(
        selected_gpu=GPUBackend.KAGGLE_GPU,
        artifacts_dir=tmp_path,
        competition_slug="name-that-llm",
        baseline_result=None,
    )
    runner = BenchmarkRunner(setup)  # type: ignore[arg-type]
    result = runner._run_kaggle_gpu("print('round')", 60, inputs=kernel_inputs(NOTEBOOK_META))
    assert "stop here" in (result.error_message or "")
    (meta,) = pushed
    assert meta["kernel_sources"] == ["lucasboesen/name-that-llm-glm4-9b-trunc25"]
    assert meta["model_sources"] == ["qwen-lm/qwen3/transformers/8b/1"]
    assert (meta["docker_image"], meta["machine_shape"]) == (
        NOTEBOOK_META["docker_image"],
        "NvidiaTeslaT4",
    )
