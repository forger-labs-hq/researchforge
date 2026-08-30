"""Experiment engine domain models (spec: Phase 1C states, manifest, funnel)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from researchforge.domain.baseline import EnvironmentFingerprint
from researchforge.domain.contract import ConstraintOperator
from researchforge.domain.environment import ExecutionEngine
from researchforge.domain.hypothesis import ExpectedImpact
from researchforge.execution.evaluation import EvaluationStatus
from researchforge.execution.metrics import MetricResult
from researchforge.execution.path_guard import PathViolation


class BenchmarkStage(StrEnum):
    SCREENING = "screening"
    FULL = "full"
    VALIDATION = "validation"


class ExperimentStatus(StrEnum):
    """Exactly the spec's experiment states."""

    PLANNED = "planned"
    APPROVED = "approved"
    PREPARING = "preparing"
    RUNNING = "running"
    FAILED_SETUP = "failed_setup"
    FAILED_EXECUTION = "failed_execution"
    REJECTED = "rejected"
    PROMISING = "promising"
    VALIDATING = "validating"
    VALIDATED = "validated"
    IMPLEMENTATION_READY = "implementation_ready"  # set in Phase 1D
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[ExperimentStatus] = frozenset(
    {
        ExperimentStatus.FAILED_SETUP,
        ExperimentStatus.FAILED_EXECUTION,
        ExperimentStatus.REJECTED,
        ExperimentStatus.VALIDATED,
        ExperimentStatus.IMPLEMENTATION_READY,
        ExperimentStatus.CANCELLED,
    }
)

ALLOWED_TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.PLANNED: frozenset({ExperimentStatus.APPROVED, ExperimentStatus.CANCELLED}),
    ExperimentStatus.APPROVED: frozenset({ExperimentStatus.PREPARING, ExperimentStatus.CANCELLED}),
    ExperimentStatus.PREPARING: frozenset(
        {
            ExperimentStatus.RUNNING,
            ExperimentStatus.FAILED_SETUP,
            ExperimentStatus.REJECTED,
            ExperimentStatus.CANCELLED,
        }
    ),
    ExperimentStatus.RUNNING: frozenset(
        {
            ExperimentStatus.FAILED_SETUP,
            ExperimentStatus.FAILED_EXECUTION,
            ExperimentStatus.REJECTED,
            ExperimentStatus.PROMISING,
            ExperimentStatus.CANCELLED,
        }
    ),
    ExperimentStatus.PROMISING: frozenset(
        {ExperimentStatus.VALIDATING, ExperimentStatus.PREPARING, ExperimentStatus.CANCELLED}
    ),
    ExperimentStatus.VALIDATING: frozenset(
        {
            ExperimentStatus.VALIDATED,
            ExperimentStatus.PROMISING,
            ExperimentStatus.REJECTED,
            ExperimentStatus.FAILED_EXECUTION,
            ExperimentStatus.CANCELLED,
        }
    ),
    ExperimentStatus.VALIDATED: frozenset({ExperimentStatus.IMPLEMENTATION_READY}),
    ExperimentStatus.FAILED_SETUP: frozenset(),
    ExperimentStatus.FAILED_EXECUTION: frozenset(),
    ExperimentStatus.REJECTED: frozenset(),
    ExperimentStatus.IMPLEMENTATION_READY: frozenset(),
    ExperimentStatus.CANCELLED: frozenset(),
}


class InvalidTransitionError(Exception):
    pass


def advance(current: ExperimentStatus, new: ExperimentStatus) -> ExperimentStatus:
    """Validate a state transition; returns `new` or raises."""
    if new not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(f"Illegal transition {current.value} -> {new.value}.")
    return new


class DecisionOutcome(StrEnum):
    KEEP = "keep"
    REJECT = "reject"
    INVESTIGATE = "investigate"


class Decision(BaseModel):
    outcome: DecisionOutcome
    reason: str


class ConstraintResult(BaseModel):
    name: str
    operator: ConstraintOperator
    threshold: float
    observed: float | None = None  # None: metric not reported
    passed: bool | None = None  # None: not evaluable (screening subsets only)
    detail: str | None = None


class PlanStatus(StrEnum):
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PlanApproval(BaseModel):
    approved_at: datetime
    method: Literal["typed", "flag"]
    experiment_ids: list[str]
    estimated_max_minutes: int


class ExperimentPlan(BaseModel):
    plan_id: str = Field(pattern=r"^plan-\d{3}$")
    hypothesis_id: str
    contract_id: str
    contract_version: int
    baseline_id: str  # the successful BaselineRun this plan is anchored to
    baseline_commit: str  # patches must apply at this commit
    approach_summary: str
    status: PlanStatus = PlanStatus.PLANNED
    approval: PlanApproval | None = None
    source_file: str
    created_at: datetime
    updated_at: datetime


def _lift_legacy_parent(values: object) -> object:
    """Read pre-DAG records, which stored a single `parent_experiment_id`.

    Experiments are persisted as JSON documents, so the migration lives here
    rather than in SQL: a record written before multi-parent support arrives
    with the singular key and no list, and is lifted into a one-item list.
    """
    if not isinstance(values, dict):
        return values
    if values.get("parent_experiment_ids") is not None:
        return values
    legacy = values.get("parent_experiment_id")
    return {**values, "parent_experiment_ids": [legacy] if legacy else []}


class Experiment(BaseModel):
    experiment_id: str = Field(pattern=r"^exp-\d{3}$")
    plan_id: str
    hypothesis_id: str
    parent_experiment_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Measured ancestors this experiment builds on, in the order the "
            "author declared them. More than one means it merges branches."
        ),
    )
    patch_includes_parents: bool = Field(
        default=False,
        description=(
            "Whether patch_text already contains every parent's change. Set when "
            "the branches could not be composed mechanically and the combination "
            "was re-authored as one self-contained diff; the parents then remain "
            "as lineage only and their patches are not applied again."
        ),
    )
    title: str
    change_summary: str
    patch_text: str  # the exact change, immutable (spec §4.5); "" for env-only experiments
    patch_sha256: str
    changed_files: list[str] = Field(default_factory=list)  # CLI-extracted, never authored
    env_overrides: dict[str, str] = Field(default_factory=dict)  # injected at run time
    path_violations: list[PathViolation] = Field(default_factory=list)
    expected_effect: ExpectedImpact | None = None
    observation: str | None = Field(
        default=None,
        description=(
            "What the run's own output showed, read back from the benchmark logs "
            "after it finished. Never a claim about the metric — the measurement "
            "is recorded separately and is what decides anything."
        ),
    )
    status: ExperimentStatus = ExperimentStatus.PLANNED
    decision: Decision | None = None
    created_at: datetime
    updated_at: datetime

    _lift_parent = model_validator(mode="before")(staticmethod(_lift_legacy_parent))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def parent_experiment_id(self) -> str | None:
        """The first declared parent — the single-lineage view of this node."""
        return self.parent_experiment_ids[0] if self.parent_experiment_ids else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_merge(self) -> bool:
        """Whether this experiment combines more than one branch."""
        return len(self.parent_experiment_ids) > 1


class RunStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExperimentRunGroup(BaseModel):
    run_id: str = Field(pattern=r"^run-\d{3}$")
    plan_id: str
    status: RunStatus = RunStatus.IN_PROGRESS
    execution_mode: ExecutionEngine
    screening_baseline_id: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)


class ExecutionArtifacts(BaseModel):
    diff_path: str
    stdout_path: str
    stderr_path: str
    results_path: str | None = None


class ExecutionRecordStatus(StrEnum):
    """Status of one stage attempt (superset of EvaluationStatus values)."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_SETUP = "failed_setup"
    FAILED_TESTS = "failed_tests"
    FAILED_EXECUTION = "failed_execution"
    FAILED_TIMEOUT = "failed_timeout"
    FAILED_INVALID_RESULT = "failed_invalid_result"
    REJECTED_PATHS = "rejected_paths"


def execution_status_from_evaluation(status: EvaluationStatus) -> ExecutionRecordStatus:
    return ExecutionRecordStatus(status.value)


class ExperimentExecution(BaseModel):
    """Manifest-shaped record for one stage attempt (spec manifest superset)."""

    execution_id: str
    experiment_id: str
    run_id: str
    hypothesis_id: str
    baseline_commit: str
    parent_experiment_ids: list[str] = Field(default_factory=list)
    execution_mode: ExecutionEngine
    benchmark_stage: BenchmarkStage
    attempt: int = Field(ge=1)
    change_summary: str
    changed_files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime | None = None
    status: ExecutionRecordStatus
    failure_reason: str | None = None
    metrics: MetricResult | None = None
    constraints: list[ConstraintResult] = Field(default_factory=list)
    artifacts: ExecutionArtifacts
    fingerprint: EnvironmentFingerprint
    decision: Decision | None = None
    warnings: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0

    _lift_parent = model_validator(mode="before")(staticmethod(_lift_legacy_parent))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def parent_experiment_id(self) -> str | None:
        return self.parent_experiment_ids[0] if self.parent_experiment_ids else None


class ValidationSummary(BaseModel):
    experiment_id: str
    attempts: int
    succeeded_attempts: int
    values: list[float] = Field(default_factory=list)
    mean: float | None = None
    stdev: float | None = None  # sample stdev (n-1); None when n < 2
    coefficient_of_variation: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    all_constraints_passed: bool = False
    improvement_confirmed_in_all: bool = False
    stdev_max: float | None = None
    """The spread limit this validation was held to, when one was set."""

    outcome: ExperimentStatus

    @property
    def stdev_exceeded(self) -> bool:
        """Whether the measured spread broke the limit the run was held to."""
        return (
            self.stdev_max is not None
            and self.stdev is not None
            and self.stdev > self.stdev_max
        )
