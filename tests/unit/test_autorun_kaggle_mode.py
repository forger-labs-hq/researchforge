"""autorun --kaggle / --judgment: evidence tiers and Kaggle GPU decisions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchforge.autorun import kaggle_mode
from researchforge.domain.contract import MetricDirection
from researchforge.kaggle import control


def _exp(exp_id: str, title: str, status: str = "promising", files: list[str] | None = None) -> Any:
    return SimpleNamespace(
        experiment_id=exp_id,
        title=title,
        change_summary="",
        changed_files=files or ["src/model.py"],
        status=SimpleNamespace(value=status),
    )


@pytest.fixture
def measured(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    values: dict[str, float] = {}
    monkeypatch.setattr(kaggle_mode, "list_executions", lambda conn: [])
    monkeypatch.setattr(kaggle_mode, "measured_values", lambda executions: values)
    return values


def test_rule_marks_improvements_and_blind_changes_worth_kaggle_gpu(
    measured: dict[str, float],
) -> None:
    measured.update({"exp-001": 0.92, "exp-002": 0.88})
    records = kaggle_mode.assess(
        None,  # type: ignore[arg-type]
        [
            _exp("exp-001", "Tune learning rate"),
            _exp("exp-002", "Lower dropout"),
            _exp("exp-003", "Add COCONUT external data fallback"),
        ],
        baseline_value=0.90,
        direction=MetricDirection.MAXIMIZE,
    )
    by_id = {r.experiment_id: r for r in records}
    assert by_id["exp-001"].evidence_tier == "local_improvement"
    assert by_id["exp-001"].worth_kaggle_gpu is True
    assert by_id["exp-002"].evidence_tier == "local_no_change"
    assert by_id["exp-002"].worth_kaggle_gpu is False
    assert by_id["exp-003"].evidence_tier == "unmeasured"
    assert all(r.decided_by == "rule" for r in records)


def test_minimize_direction_counts_lower_as_better(measured: dict[str, float]) -> None:
    measured["exp-001"] = 9.7
    (rec,) = kaggle_mode.assess(
        None,  # type: ignore[arg-type]
        [_exp("exp-001", "Blend GPU ensemble")],
        baseline_value=11.2,
        direction=MetricDirection.MINIMIZE,
    )
    assert rec.gain is not None and rec.gain > 0
    assert rec.worth_kaggle_gpu is True


def test_judgment_without_a_terminal_queues_questions_and_judge_closes_them(
    tmp_path: Path, measured: dict[str, float]
) -> None:
    measured["exp-001"] = 0.92
    records = kaggle_mode.assess(
        None,  # type: ignore[arg-type]
        [_exp("exp-001", "Tune learning rate")],
        baseline_value=0.90,
        direction=MetricDirection.MAXIMIZE,
    )
    kaggle_mode.apply_judgment(tmp_path, records, judge=None)
    kaggle_mode.save_evidence(tmp_path, records)
    assert records[0].worth_kaggle_gpu is None and records[0].decided_by == "pending"
    (question,) = control.open_interactions(tmp_path)
    assert question.options == ["yes", "no", "skip"]

    rec = kaggle_mode.record_decision(tmp_path, "exp-001", "yes")
    assert rec["worth_kaggle_gpu"] is True and rec["decided_by"] == "judgment"
    assert control.open_interactions(tmp_path) == []
    # A judgment answer is not a setup answer: nothing leaks into user_answers.json
    assert not (tmp_path / "user_answers.json").exists()
    saved = json.loads((tmp_path / kaggle_mode.EVIDENCE_FILE).read_text())
    assert saved["exp-001"]["worth_kaggle_gpu"] is True


def test_interactive_judge_decides_directly(tmp_path: Path, measured: dict[str, float]) -> None:
    measured.update({"exp-001": 0.95, "exp-002": 0.91})
    records = kaggle_mode.assess(
        None,  # type: ignore[arg-type]
        [_exp("exp-001", "A"), _exp("exp-002", "B")],
        baseline_value=0.90,
        direction=MetricDirection.MAXIMIZE,
    )
    answers = iter(["no", "skip"])
    kaggle_mode.apply_judgment(tmp_path, records, judge=lambda rec: next(answers))  # type: ignore[arg-type,return-value]
    assert [(r.worth_kaggle_gpu, r.decided_by) for r in records] == [
        (False, "judgment"),
        (None, "skipped"),
    ]
    assert control.open_interactions(tmp_path) == []


def test_record_decision_for_unknown_experiment_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        kaggle_mode.record_decision(tmp_path, "exp-404", "yes")
