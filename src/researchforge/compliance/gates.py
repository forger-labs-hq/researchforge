"""Gate abstractions and result models for the compliance engine.

A ``ComplianceGate`` is a deterministic check.  Gates must never call an LLM
as their primary evaluator.  Each gate returns a ``ComplianceResult``; the
engine collects them into a ``ComplianceRun``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from researchforge.compliance.policy import LoadedPolicy

# ── result models ────────────────────────────────────────────────────────────

GateStatus = Literal["pass", "fail", "warn", "skip", "error", "waived"]
GateSeverity = Literal["block", "warn", "info"]


class ComplianceResult(BaseModel):
    """One gate's verdict for one experiment execution."""

    gate_id: str
    name: str
    category: str
    status: GateStatus
    severity: GateSeverity
    message: str
    evidence: dict[str, Any] = {}
    remediation: str | None = None


class ComplianceRun(BaseModel):
    """All gate results for one experiment at one checkpoint."""

    compliance_run_id: str
    experiment_id: str
    project_id: str
    checkpoint: Literal["post_experiment", "pre_validation", "pre_ship"]
    policy_hash: str  # SHA-256 of the policy file; "" when no policy file
    policy_path: str
    results: list[ComplianceResult]
    created_at: datetime

    @property
    def blocking_failures(self) -> list[ComplianceResult]:
        return [r for r in self.results if r.status in ("fail", "error") and r.severity == "block"]

    @property
    def warnings(self) -> list[ComplianceResult]:
        return [r for r in self.results if r.status == "warn"]

    @property
    def passed(self) -> bool:
        """True iff there are no blocking failures."""
        return len(self.blocking_failures) == 0


# ── gate interface ───────────────────────────────────────────────────────────


@dataclass
class GateContext:
    """All inputs a gate may need during evaluation."""

    project_id: str
    experiment_id: str
    worktree: str  # absolute path to the experiment worktree
    repo_root: str  # absolute path to the project repository root
    changed_files: list[str]  # repo-relative paths changed by this experiment
    policy: LoadedPolicy
    extra: dict[str, Any] = field(default_factory=dict)


class ComplianceGate(ABC):
    """Base class for all compliance gates."""

    #: Unique machine-readable identifier, e.g. "ASSET-001"
    gate_id: str
    #: Human-readable label shown in UI and reports
    name: str
    #: Logical grouping, e.g. "protected_assets", "security", "governance"
    category: str
    #: Default severity when no policy is configured
    default_severity: GateSeverity = "block"

    @abstractmethod
    def evaluate(self, ctx: GateContext) -> ComplianceResult:
        """Run the gate deterministically and return a result."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pass(self, message: str, evidence: dict[str, Any] | None = None) -> ComplianceResult:
        return ComplianceResult(
            gate_id=self.gate_id,
            name=self.name,
            category=self.category,
            status="pass",
            severity=self.default_severity,
            message=message,
            evidence=evidence or {},
        )

    def _fail(
        self,
        message: str,
        evidence: dict[str, Any] | None = None,
        remediation: str | None = None,
        severity: GateSeverity | None = None,
    ) -> ComplianceResult:
        return ComplianceResult(
            gate_id=self.gate_id,
            name=self.name,
            category=self.category,
            status="fail",
            severity=severity or self.default_severity,
            message=message,
            evidence=evidence or {},
            remediation=remediation,
        )

    def _warn(self, message: str, evidence: dict[str, Any] | None = None) -> ComplianceResult:
        return ComplianceResult(
            gate_id=self.gate_id,
            name=self.name,
            category=self.category,
            status="warn",
            severity="warn",
            message=message,
            evidence=evidence or {},
        )

    def _skip(self, reason: str) -> ComplianceResult:
        return ComplianceResult(
            gate_id=self.gate_id,
            name=self.name,
            category=self.category,
            status="skip",
            severity="info",
            message=reason,
        )

    def _error(self, message: str, mandatory: bool = False) -> ComplianceResult:
        return ComplianceResult(
            gate_id=self.gate_id,
            name=self.name,
            category=self.category,
            status="error",
            severity="block" if mandatory else "warn",
            message=message,
        )


# ── factory ──────────────────────────────────────────────────────────────────


def new_compliance_run(
    project_id: str,
    experiment_id: str,
    checkpoint: Literal["post_experiment", "pre_validation", "pre_ship"],
    policy: LoadedPolicy,
    results: list[ComplianceResult],
) -> ComplianceRun:
    return ComplianceRun(
        compliance_run_id=f"cr-{uuid4().hex[:12]}",
        experiment_id=experiment_id,
        project_id=project_id,
        checkpoint=checkpoint,
        policy_hash=policy.raw_hash,
        policy_path=policy.path,
        results=results,
        created_at=datetime.now(UTC),
    )
