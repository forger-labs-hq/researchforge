"""Experiment contract models (spec: example researchforge.yaml + §12 entity).

`ContractSpec` mirrors the user-authored `researchforge.yaml` exactly — all
sections forbid unknown keys so typos fail loudly. `ExperimentContract` is
the frozen, stored snapshot created at approval time; changes to the yaml
after approval require re-approval and create a new version.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from researchforge.domain.project import ProjectMode


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ConstraintOperator(StrEnum):
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"
    EQ = "=="


class ContractExecutionMode(StrEnum):
    AUTO = "auto"
    DOCKER = "docker"
    VENV = "venv"


class NetworkMode(StrEnum):
    NONE = "none"
    ENABLED = "enabled"


class PrimaryMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    direction: MetricDirection
    target_value: float | None = Field(
        default=None,
        description=(
            "The value that would satisfy the objective. `autorun` stops as soon as "
            "the metric reaches it. None means 'improve as far as possible'."
        ),
    )

    @field_validator("target_value")
    @classmethod
    def _finite_target(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("target_value must be finite")
        return v


class HardConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    operator: ConstraintOperator
    value: float

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("constraint value must be finite")
        return v


class ObjectiveSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    primary_metric: PrimaryMetric
    hard_constraints: list[HardConstraint] = Field(default_factory=list)
    secondary_metrics: list[str] = Field(default_factory=list)


class ProjectSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    mode: ProjectMode


class RepositorySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_ref: str = Field(default="main", min_length=1)


class ExecutionSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ContractExecutionMode = ContractExecutionMode.AUTO
    trusted_repository: bool = False
    setup_command: str | None = None
    screening_command: str | None = None  # Phase 1C screening stage
    test_command: str | None = None  # optional required tests, run before evaluation
    full_command: str = Field(min_length=1)
    result_file: str = "artifacts/results.json"
    timeout_minutes: int = Field(default=20, ge=1, le=1440)
    cpu_limit: float = Field(default=2, gt=0)
    memory_mb: int = Field(default=4096, ge=256)
    max_experiments: int = Field(default=8, ge=1)  # consumed by Phase 1C
    stall: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Stop the run after this many consecutive non-improvements. "
            "None means run all approved experiments regardless."
        ),
    )


class PermissionsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    editable_paths: list[str] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=list)


class NetworkSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: NetworkMode = NetworkMode.NONE


class SecretsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forward_environment_variables: list[str] = Field(default_factory=list)


class ValidationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeat_finalists: int = Field(default=3, ge=1)  # consumed by Phase 1C
    require_existing_tests: bool = True


class ShippingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_branch_creation: bool = True  # consumed by Phase 1D
    allow_draft_pr: bool = False


class ContractSpec(BaseModel):
    """The `researchforge.yaml` document."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    project: ProjectSection
    objective: ObjectiveSection
    repository: RepositorySection = Field(default_factory=RepositorySection)
    execution: ExecutionSection
    permissions: PermissionsSection
    network: NetworkSection = Field(default_factory=NetworkSection)
    secrets: SecretsSection = Field(default_factory=SecretsSection)
    validation: ValidationSection = Field(default_factory=ValidationSection)
    shipping: ShippingSection = Field(default_factory=ShippingSection)


class ExperimentContract(BaseModel):
    """The immutable approved evaluation definition (spec §12)."""

    model_config = ConfigDict(frozen=True)

    contract_id: str
    contract_version: int = Field(ge=1)
    spec: ContractSpec
    source_sha256: str  # hash of the yaml file bytes at approval time
    baseline_commit: str  # resolved sha of repository.baseline_ref at approval
    approved_at: datetime
    created_at: datetime
