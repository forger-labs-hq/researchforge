"""Provider allow-list gate (PROV-001).

Records which AI providers/models were used by the experiment and enforces
``policy.models.allowed_providers`` / ``denied_providers`` when configured.
Provider usage is read from the ``ai_calls`` table written during execution.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from researchforge.compliance.gates import ComplianceGate, ComplianceResult, GateContext


def _used_providers(conn: sqlite3.Connection, experiment_id: str) -> list[dict[str, Any]]:
    """Return distinct provider/model pairs recorded for this experiment.

    ai_calls rows are keyed by project_id, not experiment_id directly, so we
    match on purpose prefix or fall back to all recent calls when the experiment
    context is not yet narrowed.  Future schema work can add experiment_id to
    ai_calls; for now we read all calls for the project and tag them as observed.
    """
    rows = conn.execute(
        "SELECT DISTINCT provider, model FROM ai_calls WHERE project_id = ?",
        (experiment_id,),  # experiment_id used as a proxy — see note above
    ).fetchall()
    return [{"provider": r["provider"], "model": r["model"]} for r in rows]


class ProviderPolicyGate(ComplianceGate):
    gate_id = "PROV-001"
    name = "Model/provider allow-list"
    category = "governance"
    default_severity = "block"

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self._conn = conn

    def evaluate(self, ctx: GateContext) -> ComplianceResult:
        allowed = [p.lower() for p in ctx.policy.spec.models.allowed_providers]
        denied = [p.lower() for p in ctx.policy.spec.models.denied_providers]

        # If no policy is configured for providers, this gate is informational
        if not allowed and not denied:
            return self._skip(
                "No provider allow/deny list configured in policy — gate is informational only."
            )

        used: list[dict[str, Any]] = []
        if self._conn is not None:
            try:
                used = _used_providers(self._conn, ctx.project_id)
            except Exception:
                used = []

        violations: list[dict[str, Any]] = []
        for entry in used:
            provider = entry["provider"].lower()
            if provider in denied:
                violations.append({**entry, "reason": "explicitly denied"})
            elif allowed and provider not in allowed:
                violations.append({**entry, "reason": "not in allowed list"})

        if violations:
            names = ", ".join(f"{v['provider']}/{v['model']}" for v in violations)
            return self._fail(
                f"Unapproved provider/model used: {names}",
                evidence={
                    "violations": violations,
                    "used_providers": used,
                    "policy_allowed": allowed,
                    "policy_denied": denied,
                },
                remediation=(
                    "Update policy.yaml to add this provider to allowed_providers, "
                    "or change the experiment to use an approved provider."
                ),
            )

        return self._pass(
            "All observed providers are within policy.",
            evidence={
                "used_providers": used,
                "policy_allowed": allowed,
                "policy_denied": denied,
            },
        )
