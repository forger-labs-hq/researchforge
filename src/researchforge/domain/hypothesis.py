"""The Hypothesis domain entity (spec: required hypothesis schema)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

HYPOTHESIS_ID_PATTERN = r"^hyp-\d{3}$"


class ImpactDirection(StrEnum):
    INCREASE = "increase"
    DECREASE = "decrease"
    UNKNOWN = "unknown"


class Level(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoveltyConfidence(StrEnum):
    """Deliberately has no HIGH member: a novelty guarantee is unrepresentable."""

    LOW = "low"
    MEDIUM = "medium"
    UNKNOWN = "unknown"


class HypothesisStatus(StrEnum):
    """Where a hypothesis stands in the human review that precedes planning.

    A newly imported hypothesis is `speculative`: the AI proposed it and nobody
    has judged it. Planning treats speculative and approved alike, so review is
    optional — but a `rejected` hypothesis is skipped, which is the whole point
    of being able to reject one.
    """

    SPECULATIVE = "speculative"
    APPROVED = "approved"
    REJECTED = "rejected"


ReviewOutcome = Literal[HypothesisStatus.APPROVED, HypothesisStatus.REJECTED]
"""The two statuses a human can set. `speculative` means nobody has decided."""


class HypothesisReview(BaseModel):
    """The human judgement on one hypothesis, and when it was made."""

    decision: ReviewOutcome
    reason: str = ""
    decided_at: datetime


class ExpectedImpact(BaseModel):
    metric: str | None = None
    direction: ImpactDirection = ImpactDirection.UNKNOWN


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(pattern=HYPOTHESIS_ID_PATTERN)
    title: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    supporting_paper_ids: list[str] = Field(default_factory=list)
    contradicting_paper_ids: list[str] = Field(default_factory=list)
    repository_observations: list[str] = Field(default_factory=list)
    expected_impact: ExpectedImpact = Field(default_factory=ExpectedImpact)
    feasibility: Level
    estimated_effort: Level
    estimated_experiment_count: int | None = Field(default=None, ge=1)
    novelty_confidence: NoveltyConfidence
    status: HypothesisStatus = HypothesisStatus.SPECULATIVE
    review: HypothesisReview | None = None
    proposed_experiment: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_status(self) -> Literal["supported", "unsupported"]:
        """Derived, never author-supplied: cites evidence or is labeled unsupported."""
        return "supported" if self.supporting_paper_ids else "unsupported"

    @property
    def is_plannable(self) -> bool:
        """Whether planning should consider this hypothesis at all."""
        return self.status is not HypothesisStatus.REJECTED
