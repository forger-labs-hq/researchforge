"""Protected-asset integrity gate (ASSET-001).

Verifies that files listed in ``policy.protected_files`` plus the implicit
always-protected set were not modified by the experiment.  This is a cheap,
deterministic check run at every post-experiment checkpoint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from researchforge.compliance.gates import ComplianceGate, ComplianceResult, GateContext

# Paths that are always protected regardless of policy config
IMPLICIT_PROTECTED: tuple[str, ...] = (
    ".researchforge/policy.yaml",
    "researchforge.yaml",
    ".researchforge/",
    ".git/",
)


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file, or empty string if it does not exist."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matches_protected(changed_path: str, protected_entry: str) -> bool:
    """Return True if ``changed_path`` falls under ``protected_entry``."""
    if protected_entry.endswith("/"):
        prefix = protected_entry.rstrip("/")
        return changed_path == prefix or changed_path.startswith(prefix + "/")
    return changed_path == protected_entry


class ProtectedAssetGate(ComplianceGate):
    gate_id = "ASSET-001"
    name = "Protected asset integrity"
    category = "protected_assets"
    default_severity = "block"

    def evaluate(self, ctx: GateContext) -> ComplianceResult:
        protected = list(IMPLICIT_PROTECTED) + list(ctx.policy.spec.protected_files)

        violations: list[dict[str, Any]] = []
        repo = Path(ctx.repo_root)

        for changed in ctx.changed_files:
            for entry in protected:
                if _matches_protected(changed, entry):
                    current_hash = _sha256_file(repo / changed)
                    violations.append(
                        {
                            "path": changed,
                            "matched_rule": entry,
                            "current_sha256": current_hash,
                        }
                    )
                    break  # one rule per changed path is enough

        if violations:
            paths = ", ".join(v["path"] for v in violations)
            return self._fail(
                f"Protected asset(s) modified: {paths}",
                evidence={"violations": violations},
                remediation=(
                    "Remove changes to protected files from the experiment patch. "
                    "Protected assets must never be altered by an experiment."
                ),
            )

        return self._pass(
            f"No protected assets modified ({len(ctx.changed_files)} file(s) checked).",
            evidence={
                "checked_files": ctx.changed_files,
                "protected_rules": protected,
            },
        )
