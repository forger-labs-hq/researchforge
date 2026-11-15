"""Kaggle-aware autorun: a local metric is evidence with a confidence, not a verdict.

``autorun --kaggle``
    Records every result (implies ``--no-metric-gate``), tags each experiment with how
    far its local metric can be trusted (high / medium / low / blind), and marks whether
    it is worth spending Kaggle GPU time on: by rule — it improved locally, or the local
    metric is blind to the change. Submission candidates are listed when the loop ends.

``autorun --judgment``
    Replaces the rule with a decision per experiment: an interactive y/n/skip prompt, or —
    when nobody is at the terminal — a non-blocking question in
    ``.researchforge/pending_interaction.json``, answered with
    ``researchforge experiment judge <exp-id> yes|no``.

Everything is written to ``.researchforge/kaggle_evidence.json``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from researchforge.domain.contract import MetricDirection
from researchforge.domain.experiment import Experiment
from researchforge.experiments.measurements import measured_values
from researchforge.experiments.selection import gain_over_baseline
from researchforge.storage.experiment_repository import list_executions

EVIDENCE_FILE = "kaggle_evidence.json"

Decision = Literal["yes", "no", "skip"]


@dataclass
class Evidence:
    experiment_id: str
    title: str
    status: str
    value: float | None
    gain: float | None
    metric_confidence: str  # high | medium | low | blind
    evidence_tier: str  # validated | local_improvement | local_no_change | unmeasured
    worth_kaggle_gpu: bool | None  # None = waiting for a judgment
    decided_by: str  # rule | judgment | pending | skipped
    recorded_at: str


JudgeFn = Callable[[Evidence], Decision]


def assess(
    conn: sqlite3.Connection,
    experiments: list[Experiment],
    *,
    baseline_value: float | None,
    direction: MetricDirection,
) -> list[Evidence]:
    """Evidence for each finished experiment, with the rule-based Kaggle GPU decision."""
    from researchforge.intelligence.scanner import tag_metric_confidence

    values = measured_values(list_executions(conn))
    now = datetime.now(UTC).isoformat(timespec="seconds")
    out: list[Evidence] = []
    for exp in experiments:
        value = values.get(exp.experiment_id)
        gain = (
            gain_over_baseline(value, baseline_value, direction)
            if value is not None and baseline_value is not None
            else None
        )
        confidence = str(
            tag_metric_confidence(
                f"{exp.title} {exp.change_summary or ''}", list(exp.changed_files)
            )
        )
        status = exp.status.value
        if status == "validated":
            tier = "validated"
        elif gain is None:
            tier = "unmeasured"
        elif gain > 0:
            tier = "local_improvement"
        else:
            tier = "local_no_change"
        worth = (gain is not None and gain > 0) or confidence == "blind"
        out.append(
            Evidence(
                experiment_id=exp.experiment_id,
                title=exp.title,
                status=status,
                value=value,
                gain=gain,
                metric_confidence=confidence,
                evidence_tier=tier,
                worth_kaggle_gpu=worth,
                decided_by="rule",
                recorded_at=now,
            )
        )
    return out


def load_evidence(state_dir: Path) -> dict[str, dict[str, Any]]:
    from researchforge.kaggle.workspace import read_json

    raw = read_json(state_dir / EVIDENCE_FILE, {}) or {}
    return raw if isinstance(raw, dict) else {}


def save_evidence(state_dir: Path, records: list[Evidence]) -> None:
    from researchforge.kaggle.workspace import atomic_write_json

    current = load_evidence(state_dir)
    for rec in records:
        current[rec.experiment_id] = asdict(rec)
    atomic_write_json(state_dir / EVIDENCE_FILE, current)


def apply_judgment(state_dir: Path, records: list[Evidence], judge: JudgeFn | None) -> None:
    """Decide per experiment: ask ``judge`` if given, else queue a non-blocking question."""
    from researchforge.kaggle.control import request_interaction

    for rec in records:
        if judge is not None:
            decision = judge(rec)
            if decision == "skip":
                rec.worth_kaggle_gpu, rec.decided_by = None, "skipped"
            else:
                rec.worth_kaggle_gpu, rec.decided_by = decision == "yes", "judgment"
            continue
        rec.worth_kaggle_gpu, rec.decided_by = None, "pending"
        request_interaction(
            state_dir,
            f"Worth Kaggle GPU validation for {rec.experiment_id} '{rec.title}' "
            f"(local {rec.metric_confidence} confidence, tier {rec.evidence_tier})?",
            kind="other",
            options=["yes", "no", "skip"],
            default="skip",
            answer_key=f"judge:{rec.experiment_id}",
        )


def record_decision(state_dir: Path, experiment_id: str, decision: Decision) -> dict[str, Any]:
    """Record a human decision for one experiment and close its open question, if any."""
    from researchforge.kaggle.control import load_interactions, respond
    from researchforge.kaggle.workspace import atomic_write_json

    current = load_evidence(state_dir)
    if experiment_id not in current:
        raise KeyError(f"No Kaggle evidence recorded for {experiment_id}")
    rec = current[experiment_id]
    rec["worth_kaggle_gpu"] = None if decision == "skip" else decision == "yes"
    rec["decided_by"] = "skipped" if decision == "skip" else "judgment"
    atomic_write_json(state_dir / EVIDENCE_FILE, current)
    for item in load_interactions(state_dir):
        if item.status == "open" and item.answer_key == f"judge:{experiment_id}":
            respond(state_dir, item.id, decision)
    return rec
