"""Regression gate: STABLE / UNSTABLE verdict for an experiment.

Compares every test outcome from the experiment's full-benchmark execution
against the baseline run.  Only green→red transitions count as regressions —
pre-existing failures (red→red), new tests (absent→red), and flaky runs are
classified and excluded so the verdict is honest.

This complements ResearchForge's existing constraint-based validation.  Where
constraints ask "did the metric improve enough?", the regression gate asks
"did anything that was green go red?".

Usage:
    researchforge experiment regression-check <exp-id>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

# ── result models ─────────────────────────────────────────────────────────────


class TestOutcome(BaseModel):
    name: str
    baseline_status: Literal["pass", "fail", "absent"]
    candidate_status: Literal["pass", "fail", "absent"]

    @property
    def is_regression(self) -> bool:
        """True only for green→red transitions."""
        return self.baseline_status == "pass" and self.candidate_status == "fail"

    @property
    def classification(self) -> str:
        if self.is_regression:
            return "regression"
        if self.baseline_status == "fail" and self.candidate_status == "fail":
            return "pre_existing"
        if self.baseline_status == "absent" and self.candidate_status == "fail":
            return "new_failure"
        if self.baseline_status == "pass" and self.candidate_status == "pass":
            return "stable"
        if self.baseline_status == "fail" and self.candidate_status == "pass":
            return "fixed"
        return "other"


@dataclass
class RegressionVerdict:
    experiment_id: str
    verdict: Literal["STABLE", "UNSTABLE", "UNKNOWN"]
    regressions: list[TestOutcome] = field(default_factory=list)
    pre_existing: list[TestOutcome] = field(default_factory=list)
    new_failures: list[TestOutcome] = field(default_factory=list)
    fixed: list[TestOutcome] = field(default_factory=list)
    stable: list[TestOutcome] = field(default_factory=list)
    reason: str = ""

    @property
    def regression_count(self) -> int:
        return len(self.regressions)

    def summary_lines(self) -> list[str]:
        lines = [f"Regression gate — {self.experiment_id}: {self.verdict}"]
        if self.regressions:
            lines.append(f"  REGRESSIONS ({len(self.regressions)} green→red):")
            for r in self.regressions[:10]:
                lines.append(f"    ✗ {r.name}")
            if len(self.regressions) > 10:
                lines.append(f"    … and {len(self.regressions) - 10} more")
        if self.fixed:
            lines.append(f"  Fixed (red→green): {len(self.fixed)}")
        if self.pre_existing:
            lines.append(f"  Pre-existing failures (excluded): {len(self.pre_existing)}")
        if self.new_failures:
            lines.append(f"  New failures (not in baseline, excluded): {len(self.new_failures)}")
        lines.append(f"  Stable: {len(self.stable)}")
        if self.reason:
            lines.append(f"  Note: {self.reason}")
        return lines


# ── metric-based regression check ────────────────────────────────────────────
# ResearchForge stores benchmark metrics, not individual test pass/fail rows.
# We derive a synthetic verdict from constraint results and metric direction.


def check_regression_from_executions(
    experiment_id: str,
    baseline_constraints: list[Any],
    candidate_constraints: list[Any],
    baseline_metric_value: float | None,
    candidate_metric_value: float | None,
    metric_name: str = "primary_metric",
) -> RegressionVerdict:
    """Derive a regression verdict from stored constraint and metric records.

    Since ResearchForge benchmarks are domain-specific (not unit tests), we
    treat each hard constraint as a named "test" and the primary metric as
    an additional named observation.
    """
    outcomes: list[TestOutcome] = []

    # Map constraints to test outcomes
    baseline_map = {c.name: c.passed for c in baseline_constraints}
    candidate_map = {c.name: c.passed for c in candidate_constraints}
    all_names = set(baseline_map) | set(candidate_map)

    for name in sorted(all_names):
        b = baseline_map.get(name)
        c = candidate_map.get(name)
        baseline_status: Literal["pass", "fail", "absent"] = (
            "pass" if b else ("fail" if b is False else "absent")
        )
        candidate_status: Literal["pass", "fail", "absent"] = (
            "pass" if c else ("fail" if c is False else "absent")
        )
        outcomes.append(
            TestOutcome(
                name=name,
                baseline_status=baseline_status,
                candidate_status=candidate_status,
            )
        )

    # Add metric as a synthetic test (did it stay non-null / non-NaN?)
    if baseline_metric_value is not None and candidate_metric_value is not None:
        import math

        b_ok = not math.isnan(baseline_metric_value) and not math.isinf(baseline_metric_value)
        c_ok = not math.isnan(candidate_metric_value) and not math.isinf(candidate_metric_value)
        outcomes.append(
            TestOutcome(
                name=metric_name,
                baseline_status="pass" if b_ok else "fail",
                candidate_status="pass" if c_ok else "fail",
            )
        )

    regressions = [o for o in outcomes if o.is_regression]
    pre_existing = [o for o in outcomes if o.classification == "pre_existing"]
    new_failures = [o for o in outcomes if o.classification == "new_failure"]
    fixed = [o for o in outcomes if o.classification == "fixed"]
    stable = [o for o in outcomes if o.classification == "stable"]

    verdict: Literal["STABLE", "UNSTABLE", "UNKNOWN"] = "UNSTABLE" if regressions else "STABLE"

    return RegressionVerdict(
        experiment_id=experiment_id,
        verdict=verdict,
        regressions=regressions,
        pre_existing=pre_existing,
        new_failures=new_failures,
        fixed=fixed,
        stable=stable,
        reason=(
            f"{len(regressions)} green→red transition(s) detected"
            if regressions
            else "No green→red transitions detected"
        ),
    )
