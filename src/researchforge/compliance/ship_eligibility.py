"""Canonical ship-eligibility decision.

A single function encapsulates all reasons an experiment can or cannot ship.
Nothing else should scatter this logic — import ``can_ship`` here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from researchforge.compliance.gates import ComplianceRun
from researchforge.compliance.policy import LoadedPolicy


@dataclass
class ShipDecision:
    eligible: bool
    status: str  # "ship_eligible" | "ready_for_approval" | "not_eligible"
    blocking_reasons: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [f"Status: {self.status.upper().replace('_', ' ')}"]
        for r in self.blocking_reasons:
            lines.append(f"  BLOCK  {r}")
        for a in self.required_actions:
            lines.append(f"  ACTION {a}")
        for w in self.warnings:
            lines.append(f"  WARN   {w}")
        return lines


def can_ship(
    experiment_id: str,
    is_technically_validated: bool,
    compliance_runs: list[ComplianceRun],
    policy: LoadedPolicy,
    conn: sqlite3.Connection | None = None,
) -> ShipDecision:
    """Decide whether an experiment is eligible to ship.

    Parameters
    ----------
    experiment_id:
        The experiment being evaluated.
    is_technically_validated:
        True iff the experiment reached VALIDATED status (repeated benchmark
        runs all passed constraints).
    compliance_runs:
        All ComplianceRun records for this experiment (any checkpoint).
    policy:
        The loaded project policy.
    conn:
        Optional DB connection for live approval lookups.
    """
    blocking: list[str] = []
    actions: list[str] = []
    warnings: list[str] = []

    # 1. Technical gate
    if not is_technically_validated:
        blocking.append("Experiment has not been technically validated.")

    # 2. Blocking compliance failures across all runs
    for run in compliance_runs:
        for failure in run.blocking_failures:
            blocking.append(f"[{failure.gate_id}] {failure.name}: {failure.message}")

    # 3. Collect warnings
    for run in compliance_runs:
        for w in run.warnings:
            warnings.append(f"[{w.gate_id}] {w.name}: {w.message}")

    # 4. Human approval
    needs_approval = policy.spec.shipping.human_approval_required
    if needs_approval:
        has_approval = False
        if conn is not None:
            row = conn.execute(
                "SELECT 1 FROM compliance_approvals WHERE experiment_id = ? AND revoked = 0",
                (experiment_id,),
            ).fetchone()
            has_approval = row is not None
        if not has_approval:
            actions.append(
                f"Human approval required — run `researchforge compliance approve {experiment_id}`"
            )

    # ── derive status ─────────────────────────────────────────────────────────
    if blocking:
        status = "not_eligible"
        eligible = False
    elif actions:
        status = "ready_for_approval"
        eligible = False
    else:
        status = "ship_eligible"
        eligible = True

    return ShipDecision(
        eligible=eligible,
        status=status,
        blocking_reasons=blocking,
        required_actions=actions,
        warnings=warnings,
    )
