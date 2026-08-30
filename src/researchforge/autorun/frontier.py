"""Which hypothesis to try next, and where in the graph to try it.

A hypothesis used to be spent the moment it was planned once, anywhere. That
made the loop a queue that drains rather than a search that expands: a graph
full of promising nodes would sit untouched because every idea had already been
"used" against the baseline, and the run would stop with nothing left to try.

What is actually spent is a hypothesis *at a node*. The same idea applied to a
different ancestor is a different experiment with a different measurement, and
is exactly the move a person makes after a result comes in. Two rules keep that
from degenerating:

1. A pair is tried once. Re-planning the same hypothesis against the same node
   reproduces an experiment already in the graph, patch for patch.
2. A hypothesis already applied anywhere in a node's lineage is not applied
   again on top of itself, where its patch would either conflict or land as a
   change that measures exactly what its parent did.

Everything here is pure and works on ids, so the policy can be argued with in a
test rather than inferred from a run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from researchforge.experiments.selection import BASELINE_NODE


@dataclass(frozen=True)
class Attempt:
    """One hypothesis, tried at one node, and what came of it."""

    hypothesis_id: str
    node_id: str
    """The node its patch was written against; the baseline for a fresh start."""

    gain: float | None = None
    """Improvement over the baseline, or None when it never measured."""

    no_op: bool = False
    """It measured exactly what its parent did — the change did nothing."""


def lineage_hypotheses(
    node_id: str,
    ancestors_of: Mapping[str, set[str]],
    hypothesis_of: Mapping[str, str],
) -> set[str]:
    """Every hypothesis already applied on the way to `node_id`, including its own."""
    if node_id == BASELINE_NODE:
        return set()
    lineage = {node_id, *ancestors_of.get(node_id, set())}
    return {hypothesis_of[node] for node in lineage if node in hypothesis_of}


def available_hypotheses(
    candidates: Sequence[str],
    node_id: str,
    attempts: Iterable[Attempt],
    lineage: set[str],
) -> list[str]:
    """The candidates that can still say something new at this node.

    Order is the caller's, so a deterministic candidate list gives a
    deterministic frontier.
    """
    tried = {attempt.hypothesis_id for attempt in attempts if attempt.node_id == node_id}
    return [
        hypothesis
        for hypothesis in candidates
        if hypothesis not in tried and hypothesis not in lineage
    ]


PROVEN, UNTRIED, SPENT = 0, 1, 2
"""Ranking tiers: what has worked, what is unknown, what has not worked."""


def _record(hypothesis_id: str, attempts: Iterable[Attempt]) -> tuple[int, float, int]:
    """This hypothesis's tier, best gain anywhere, and how often it did nothing."""
    history = [attempt for attempt in attempts if attempt.hypothesis_id == hypothesis_id]
    if not history:
        return UNTRIED, 0.0, 0

    gains = [attempt.gain for attempt in history if attempt.gain is not None]
    best = max(gains, default=0.0)
    no_ops = sum(1 for attempt in history if attempt.no_op)
    return (PROVEN if best > 0 else SPENT), best, no_ops


def rank_hypotheses(available: Sequence[str], attempts: Iterable[Attempt]) -> list[str]:
    """Best move first, judged on what each hypothesis has already produced.

    An idea that improved the metric somewhere is the strongest candidate to
    carry onto the new best node. An idea never tried at all comes next, since
    an unknown is worth more than a known failure. What is left is ordered by
    how much it managed and how often it changed nothing measurable — a
    hypothesis whose every attempt landed on its parent's exact value is
    telling you its knob is not wired to the benchmark.
    """
    history = list(attempts)
    return sorted(
        available,
        key=lambda h: (
            (record := _record(h, history))[0],
            -record[1],
            record[2],
            h,
        ),
    )


def round_breadth(
    available: int,
    remaining_minutes: float | None,
    minutes_per_hypothesis: float,
) -> int:
    """How many hypotheses one round should commit to.

    Each hypothesis costs a plan's worth of benchmark runs, so with a time
    budget the round takes as many as the remaining hours can actually pay for.
    With no budget stated there is nothing to divide, and the round takes one:
    a narrow round measures, learns, and re-selects, which is the whole point of
    running a search instead of a batch.
    """
    if available <= 0:
        return 0
    if remaining_minutes is None or minutes_per_hypothesis <= 0:
        return 1
    affordable = int(remaining_minutes // minutes_per_hypothesis)
    return max(1, min(available, affordable))
