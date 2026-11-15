"""Which pairs of repo experiments are worth merging.

One rule set shared by ``autorun --dry-run`` and the live dashboard, so the
merge candidates the UI draws are exactly the ones the engine would suggest.
Works on JSON-shaped experiment dicts (``Experiment.model_dump(mode="json")``).
"""

from __future__ import annotations

from typing import Any


def is_improved(e: dict[str, Any]) -> bool:
    """Promising, or kept by a decision — the engine's notion of an improvement."""
    decision = e.get("decision") or {}
    return e.get("status") == "promising" or decision.get("outcome") == "keep"


def find_merge_opportunities(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pairs of independently improved experiments that can be combined."""
    improved = [e for e in experiments if is_improved(e)]
    out: list[dict[str, Any]] = []
    for i, a in enumerate(improved):
        for b in improved[i + 1 :]:
            # Same-hypothesis pairs are variations of one setting
            # (e.g. rrf_weight_mid=0.6 vs 1.5), not independent improvements.
            if a.get("hypothesis_id") == b.get("hypothesis_id"):
                continue
            # Overlapping patched files cannot be mechanically merged.
            if set(a.get("changed_files") or []) & set(b.get("changed_files") or []):
                continue
            # Both must have improved from different lineages.
            parents_a = set(a.get("parent_experiment_ids") or [])
            if parents_a & set(b.get("parent_experiment_ids") or []):
                continue
            if a["experiment_id"] == b["experiment_id"]:
                continue
            out.append(
                {
                    "parent_a": a["experiment_id"],
                    "parent_b": b["experiment_id"],
                    "hypothesis_a": a.get("hypothesis_id"),
                    "hypothesis_b": b.get("hypothesis_id"),
                    "command": (
                        "researchforge experiment plan <hyp> "
                        f"--merge {a['experiment_id']},{b['experiment_id']} --json"
                    ),
                }
            )
    return out
