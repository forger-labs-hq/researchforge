"""Autorun loop state persisted to `.researchforge/autorun.json`.

The loop can run for hours; Ctrl-C, a crashed benchmark, or a closed laptop
must not lose the round history.  Every completed round is written here, so
`researchforge autorun --resume` picks up with the same stall counter, the
same elapsed-time budget, and the same accumulated round record.

The file is external input on the way back in: it is validated against these
models before the engine trusts a single field of it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from researchforge.config.paths import autorun_state_path

STATE_VERSION = 1


class AutorunStateError(RuntimeError):
    """The state file exists but cannot be trusted."""


class RoundRecord(BaseModel):
    """One completed synthesis round."""

    model_config = ConfigDict(extra="forbid")

    round_num: int = Field(ge=1)
    hypotheses_planned: list[str] = Field(default_factory=list)
    plan_ids: list[str] = Field(default_factory=list)
    experiments_run: int = 0
    promising: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    best_experiment_id: str | None = None
    best_metric_value: float | None = None
    improved: bool = False
    duration_seconds: float = 0.0
    completed_at: datetime


class AutorunState(BaseModel):
    """Everything needed to resume a loop mid-flight."""

    model_config = ConfigDict(extra="forbid")

    version: int = STATE_VERSION
    status: Literal["running", "stopped"] = "running"
    started_at: datetime
    updated_at: datetime
    elapsed_seconds: float = 0.0
    rounds_completed: int = 0
    global_stall_count: int = 0
    best_experiment_id: str | None = None
    best_metric_value: float | None = None
    stop_reason: str = ""
    objective_achieved: bool = False
    total_experiments: int = 0
    settings: dict[str, str] = Field(default_factory=dict)
    rounds: list[RoundRecord] = Field(default_factory=list)


def rounds_by_experiment(state: AutorunState | None) -> dict[str, int]:
    """Which loop round each experiment came from, for grouping it in a report.

    Rounds are not stored on the experiment itself — the round is a fact about
    the loop, not about the change — so it is read back from the loop's own
    record of what it ran. A project driven by hand has no rounds at all.
    """
    if state is None:
        return {}
    return {
        experiment_id: record.round_num
        for record in state.rounds
        for experiment_id in (*record.promising, *record.rejected, *record.failed)
    }


def new_state(settings: dict[str, str]) -> AutorunState:
    now = datetime.now(UTC)
    return AutorunState(started_at=now, updated_at=now, settings=settings)


def load_state(base: Path | None = None) -> AutorunState | None:
    """Read the persisted loop state, or None when no loop has ever run."""
    path = autorun_state_path(base)
    if not path.is_file():
        return None
    try:
        return AutorunState.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise AutorunStateError(
            f"{path} is not a valid autorun state file ({exc}). Delete it to start a fresh loop."
        ) from exc


def save_state(state: AutorunState, base: Path | None = None) -> Path:
    """Write the loop state atomically so a Ctrl-C can never truncate it."""
    path = autorun_state_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.model_copy(update={"updated_at": datetime.now(UTC)})
    temp = path.with_suffix(".json.tmp")
    temp.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    temp.replace(path)
    return path


def resumable(state: AutorunState | None) -> bool:
    """Whether `--resume` has anything to continue."""
    return state is not None and state.status == "running"


def summarize_settings(values: dict[str, object]) -> dict[str, str]:
    """Flatten a config snapshot to strings for display in the state file."""
    return {key: json.dumps(value) for key, value in values.items()}
