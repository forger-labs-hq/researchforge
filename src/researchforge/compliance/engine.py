"""Compliance gate engine.

Discovers the applicable gates for a given checkpoint, evaluates them, and
returns a ``ComplianceRun``.  Errors in individual gates are caught and
recorded — a gate crash never silently passes.
"""

from __future__ import annotations

import sqlite3
import traceback
from typing import Literal

from researchforge.compliance.gate_impls.human_approval import HumanApprovalGate
from researchforge.compliance.gate_impls.protected_assets import ProtectedAssetGate
from researchforge.compliance.gate_impls.provider_policy import ProviderPolicyGate
from researchforge.compliance.gates import (
    ComplianceGate,
    ComplianceResult,
    ComplianceRun,
    GateContext,
    new_compliance_run,
)
from researchforge.compliance.policy import LoadedPolicy

Checkpoint = Literal["post_experiment", "pre_validation", "pre_ship"]

# Gates that run at each checkpoint.  Order matters for reporting.
_POST_EXPERIMENT_GATES: list[type[ComplianceGate]] = [
    ProtectedAssetGate,
    ProviderPolicyGate,
]

_PRE_VALIDATION_GATES: list[type[ComplianceGate]] = [
    ProtectedAssetGate,
    ProviderPolicyGate,
]

_PRE_SHIP_GATES: list[type[ComplianceGate]] = [
    ProtectedAssetGate,
    ProviderPolicyGate,
    HumanApprovalGate,
]


def _build_gates(
    checkpoint: Checkpoint,
    conn: sqlite3.Connection | None,
) -> list[ComplianceGate]:
    """Instantiate gate objects for the given checkpoint."""
    if checkpoint == "post_experiment":
        classes = _POST_EXPERIMENT_GATES
    elif checkpoint == "pre_validation":
        classes = _PRE_VALIDATION_GATES
    else:
        classes = _PRE_SHIP_GATES

    gates: list[ComplianceGate] = []
    for cls in classes:
        import inspect

        sig = inspect.signature(cls.__init__)
        if "conn" in sig.parameters:
            gates.append(cls(conn=conn))  # type: ignore[call-arg]
        else:
            gates.append(cls())
    return gates


def _safe_evaluate(gate: ComplianceGate, ctx: GateContext) -> ComplianceResult:
    """Evaluate a gate; catch all exceptions and return an error result."""
    try:
        return gate.evaluate(ctx)
    except Exception:
        tb = traceback.format_exc()
        return ComplianceResult(
            gate_id=gate.gate_id,
            name=gate.name,
            category=gate.category,
            status="error",
            severity="block",
            message=f"Gate raised an unexpected exception: {tb[:500]}",
        )


def run_compliance(
    project_id: str,
    experiment_id: str,
    checkpoint: Checkpoint,
    changed_files: list[str],
    worktree: str,
    repo_root: str,
    policy: LoadedPolicy,
    conn: sqlite3.Connection | None = None,
    extra: dict | None = None,
) -> ComplianceRun:
    """Evaluate all gates for the given checkpoint and return a ComplianceRun."""
    ctx = GateContext(
        project_id=project_id,
        experiment_id=experiment_id,
        worktree=worktree,
        repo_root=repo_root,
        changed_files=changed_files,
        policy=policy,
        extra=extra or {},
    )

    gates = _build_gates(checkpoint, conn)
    results = [_safe_evaluate(gate, ctx) for gate in gates]

    return new_compliance_run(
        project_id=project_id,
        experiment_id=experiment_id,
        checkpoint=checkpoint,
        policy=policy,
        results=results,
    )
