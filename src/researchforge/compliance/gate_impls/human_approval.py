"""Human approval gate (APPR-001).

Blocks shipping when ``policy.shipping.human_approval_required`` is True and
no approval record exists for the experiment.
"""

from __future__ import annotations

import sqlite3

from researchforge.compliance.gates import ComplianceGate, ComplianceResult, GateContext


class HumanApprovalGate(ComplianceGate):
    gate_id = "APPR-001"
    name = "Human approval"
    category = "governance"
    default_severity = "block"

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self._conn = conn

    def evaluate(self, ctx: GateContext) -> ComplianceResult:
        if not ctx.policy.spec.shipping.human_approval_required:
            return self._skip("Human approval not required by policy.")

        if self._conn is None:
            return self._skip("No database connection — cannot check approval records.")

        row = self._conn.execute(
            "SELECT approval_id, actor, created_at FROM compliance_approvals "
            "WHERE experiment_id = ? AND revoked = 0",
            (ctx.experiment_id,),
        ).fetchone()

        if row is None:
            return self._fail(
                f"Human approval required but not yet recorded for {ctx.experiment_id}.",
                evidence={"experiment_id": ctx.experiment_id},
                remediation=(
                    "Run `researchforge compliance approve <experiment_id>` to "
                    "record approval before shipping."
                ),
            )

        return self._pass(
            f"Approved by {row['actor']} at {row['created_at']}.",
            evidence={
                "approval_id": row["approval_id"],
                "actor": row["actor"],
                "approved_at": row["created_at"],
            },
        )
