"""Which node should the next round build on?

Always branching from the current best is greedy: a path that looks strongest
after two rounds can be a dead end, and the run never returns to the promising
branch it abandoned.  So the loop scores every node in the experiment graph the
way tree search does — value plus an uncertainty bonus — and expands the winner.

    score(node) = reward(node) + C * sqrt(ln(total_visits) / visits(node))

`reward` is how much of the best-known improvement this node captured, so it is
comparable across metrics of any scale.  `visits` is how many children have been
spawned from the node, so a heavily-explored branch loses its bonus and the
search drifts to an untried one on its own — backtracking with no rule for it.

`C` is the caller's exploration constant: at 0 this is plain "expand the best".
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

from researchforge.domain.contract import MetricDirection

BASELINE_NODE = "baseline"
"""The frozen reference every experiment is measured against."""


@dataclass(frozen=True)
class NodeStats:
    """One node of the experiment graph, as the selector sees it."""

    node_id: str
    gain: float = 0.0
    """Direction-aware improvement over the baseline, never below zero."""

    visits: int = 0
    """Children already spawned from this node."""


def gain_over_baseline(
    value: float, baseline_value: float, direction: MetricDirection
) -> float:
    """How much this measurement improved on the baseline, floored at zero.

    A node that did worse than the baseline is worth zero rather than a negative
    score: it is still a place the search may branch from, just not a promising
    one.
    """
    raw = (
        value - baseline_value
        if direction is MetricDirection.MAXIMIZE
        else baseline_value - value
    )
    return max(0.0, raw)


def reward(node: NodeStats, best_gain: float) -> float:
    """The node's gain as a fraction of the best gain anywhere (0–1).

    Normalizing against the field is what makes one exploration constant work
    for an F1 in [0, 1] and a latency in milliseconds alike.
    """
    if best_gain <= 0:
        return 0.0
    return min(1.0, node.gain / best_gain)


def ucb1_score(node: NodeStats, best_gain: float, total_visits: int, explore: float) -> float:
    """The node's exploitation value plus its exploration bonus.

    An unexplored node is scored as though it had one visit rather than as
    infinitely attractive: every round adds new nodes, so an infinite score
    would mean the loop could never return to a node it had already expanded.
    """
    exploitation = reward(node, best_gain)
    if explore <= 0:
        return exploitation
    bonus = sqrt(log(max(total_visits, 2)) / max(node.visits, 1))
    return exploitation + explore * bonus


def selection_order(nodes: list[NodeStats], explore: float) -> list[NodeStats]:
    """Every node, best candidate first.

    Ties break toward the larger gain, then the baseline, then the newer node.
    Preferring the baseline matters while nothing has beaten it yet: every node
    scores zero then, and starting fresh beats compounding on a change that did
    not help. The ordering is total, so the same graph always chooses the same
    node and a round can be explained from the log afterwards.

    The whole order matters, not just its head: the best node may have nothing
    left to try on it, and the round should move down the list rather than
    conclude the search is over.
    """
    if not nodes:
        return []
    best_gain = max(node.gain for node in nodes)
    total_visits = sum(node.visits for node in nodes)
    return sorted(
        nodes,
        key=lambda node: (
            ucb1_score(node, best_gain, total_visits, explore),
            node.gain,
            node.node_id == BASELINE_NODE,
            node.node_id,
        ),
        reverse=True,
    )


def select_node(nodes: list[NodeStats], explore: float) -> NodeStats | None:
    """The node the next experiments should build on, or None when there are none."""
    order = selection_order(nodes, explore)
    return order[0] if order else None


@dataclass(frozen=True)
class MergeProposal:
    """Two independent winners worth combining into one multi-parent experiment."""

    parents: tuple[str, str]
    combined_gain: float


def propose_merge(
    nodes: list[NodeStats],
    ancestors_of: dict[str, set[str]],
    already_merged: set[frozenset[str]],
) -> MergeProposal | None:
    """The most promising pair of unrelated winners, or None if there is none.

    Two nodes are worth merging only when each improved on the baseline on its
    own and neither is in the other's lineage — otherwise the "merge" is just
    the deeper of the two, which has already been measured.
    """
    winners = sorted(
        (node for node in nodes if node.gain > 0 and node.node_id != BASELINE_NODE),
        key=lambda node: (-node.gain, node.node_id),
    )
    for index, left in enumerate(winners):
        for right in winners[index + 1 :]:
            if right.node_id in ancestors_of.get(left.node_id, set()):
                continue
            if left.node_id in ancestors_of.get(right.node_id, set()):
                continue
            if frozenset({left.node_id, right.node_id}) in already_merged:
                continue
            return MergeProposal(
                parents=(left.node_id, right.node_id),
                combined_gain=left.gain + right.gain,
            )
    return None
