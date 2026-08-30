"""Baseline run models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from researchforge.domain.environment import ExecutionEngine
from researchforge.execution.metrics import MetricResult


class BaselineStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED_SETUP = "failed_setup"
    FAILED_EXECUTION = "failed_execution"
    FAILED_TIMEOUT = "failed_timeout"
    FAILED_INVALID_RESULT = "failed_invalid_result"


class EnvironmentFingerprint(BaseModel):
    platform: str
    execution_mode: ExecutionEngine
    python_version: str | None = None
    docker_image_id: str | None = None
    venv_packages_hash: str | None = None  # sha256 of `pip freeze`
    contract_id: str
    contract_version: int
    commit_sha: str


class BaselineRepeats(BaseModel):
    """What the individual measurements showed when a baseline was run more than once.

    A noisy benchmark measured once gives a reference point that is partly luck,
    and every improvement is then compared against that luck. Repeating it and
    averaging removes some of that, but only if the spread stays visible: a
    baseline with a wide spread cannot support a small claimed improvement, and
    this is where a reader can see that.
    """

    requested: int = Field(ge=1)
    values: list[float] = Field(min_length=1)
    """The primary metric from each repeat that succeeded, in the order they ran."""

    failed: int = Field(default=0, ge=0)
    mean: float
    stdev: float | None = None
    """None when only one repeat succeeded — a spread needs two measurements."""

    coefficient_of_variation: float | None = None


class BaselineRun(BaseModel):
    baseline_id: str
    contract_id: str
    contract_version: int
    commit_sha: str
    execution_mode: ExecutionEngine
    command: str
    command_kind: str = "full"  # which contract command ran (1C adds "screening")
    status: BaselineStatus
    failure_reason: str | None = None
    metrics: MetricResult | None = None
    repeats: BaselineRepeats | None = None
    """Set only when the baseline was measured more than once; `metrics` is then the mean."""

    warnings: list[str] = Field(default_factory=list)
    fingerprint: EnvironmentFingerprint
    stdout_path: str
    stderr_path: str
    results_path: str | None = None
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
