"""Setup for a code competition runs the full loop on Kaggle GPU, not an ideas list.

Only Kaggle notebooks can submit to a code competition, and rounds already run as
Kaggle notebooks there. Setup used to send every code competition to the ideation
path, which writes a plan and never creates the loop, so no round could run.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchforge.kaggle import orchestrator as orch_mod
from researchforge.kaggle.orchestrator import (
    OvernightError,
    OvernightOrchestrator,
    code_competition_compute,
)


def _route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    code: bool,
    data_gb: float = 1.0,
    gpu: str | None = None,
) -> list[tuple[str, Any]]:
    import researchforge.ai.providers as providers
    import researchforge.kaggle.competition_reader as reader
    import researchforge.kaggle.credentials as credentials

    monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(tmp_path / ".overnight_demo"))
    monkeypatch.setattr(credentials, "validate_credentials", lambda: True)
    monkeypatch.setattr(providers, "get_provider", lambda *a, **k: None)
    meta = SimpleNamespace(is_code_competition=code, estimated_data_gb=data_gb)
    monkeypatch.setattr(reader, "read_competition_metadata", lambda *a, **k: meta)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        OvernightOrchestrator,
        "_run_execution_flow",
        lambda self, meta, ai, gpu_preference, *a: calls.append(("loop", gpu_preference)) or {},
    )
    monkeypatch.setattr(
        OvernightOrchestrator,
        "_run_ideation_flow",
        lambda self, meta, ai, reason="": calls.append(("ideas", reason)) or {},
    )
    OvernightOrchestrator("demo").run_full_flow(gpu_preference=gpu)
    return calls


def test_a_code_competition_gets_the_loop_on_kaggle_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _route(tmp_path, monkeypatch, code=True) == [("loop", "kaggle")]
    assert _route(tmp_path, monkeypatch, code=True, gpu="kaggle") == [("loop", "kaggle")]


def test_a_code_competition_refuses_compute_off_kaggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(OvernightError, match="only Kaggle notebooks can submit"):
        _route(tmp_path, monkeypatch, code=True, gpu="colab")


def test_data_too_big_for_the_disk_still_gets_the_ideas_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _route(tmp_path, monkeypatch, code=False, data_gb=10**6)
    assert calls[0][0] == "ideas" and "exceeds free disk" in calls[0][1]


def test_an_ordinary_competition_keeps_its_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _route(tmp_path, monkeypatch, code=False, gpu="local") == [("loop", "local")]


@pytest.mark.parametrize(
    ("asked", "runs"),
    [
        (None, "kaggle"),
        ("auto", "kaggle"),
        ("kaggle", "kaggle"),
        ("kaggle_gpu", "kaggle_gpu"),
        ("competition", "competition"),
    ],
)
def test_code_competition_compute(asked: str | None, runs: str) -> None:
    assert code_competition_compute(asked) == runs


@pytest.mark.parametrize("asked", ["local", "colab", "ssh", "cloud"])
def test_code_competition_compute_refuses(asked: str) -> None:
    with pytest.raises(OvernightError, match="code competition"):
        code_competition_compute(asked)


def test_the_module_exports_the_helper() -> None:
    assert orch_mod.code_competition_compute is code_competition_compute
