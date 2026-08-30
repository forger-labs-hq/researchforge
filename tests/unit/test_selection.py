"""UCB1 node selection over the experiment graph."""

import pytest

from researchforge.domain.contract import MetricDirection
from researchforge.experiments.selection import (
    BASELINE_NODE,
    NodeStats,
    gain_over_baseline,
    propose_merge,
    reward,
    select_node,
    ucb1_score,
)


class TestGainOverBaseline:
    def test_maximize_improvement_is_positive(self) -> None:
        assert gain_over_baseline(0.85, 0.80, MetricDirection.MAXIMIZE) == pytest.approx(0.05)

    def test_maximize_regression_floors_at_zero(self) -> None:
        assert gain_over_baseline(0.70, 0.80, MetricDirection.MAXIMIZE) == 0.0

    def test_minimize_improvement_is_positive(self) -> None:
        assert gain_over_baseline(120.0, 150.0, MetricDirection.MINIMIZE) == 30.0

    def test_minimize_regression_floors_at_zero(self) -> None:
        assert gain_over_baseline(180.0, 150.0, MetricDirection.MINIMIZE) == 0.0

    def test_equal_to_baseline_is_zero(self) -> None:
        assert gain_over_baseline(0.80, 0.80, MetricDirection.MAXIMIZE) == 0.0


class TestReward:
    def test_best_node_scores_one(self) -> None:
        assert reward(NodeStats("exp-001", gain=0.05), best_gain=0.05) == 1.0

    def test_half_the_best_gain_scores_half(self) -> None:
        assert reward(NodeStats("exp-002", gain=0.025), best_gain=0.05) == 0.5

    def test_no_gain_scores_zero(self) -> None:
        assert reward(NodeStats("exp-003", gain=0.0), best_gain=0.05) == 0.0

    def test_nothing_beat_the_baseline_scores_zero(self) -> None:
        assert reward(NodeStats("exp-001", gain=0.0), best_gain=0.0) == 0.0

    def test_large_metric_scale_still_normalizes(self) -> None:
        """A latency gain of 30ms and an F1 gain of 0.05 both normalize to 1.0."""
        assert reward(NodeStats("exp-001", gain=30.0), best_gain=30.0) == 1.0


class TestUcb1Score:
    def test_zero_explore_is_pure_exploitation(self) -> None:
        node = NodeStats("exp-001", gain=0.05, visits=99)
        assert ucb1_score(node, best_gain=0.05, total_visits=100, explore=0.0) == 1.0

    def test_bonus_shrinks_as_a_node_is_expanded(self) -> None:
        fresh = NodeStats("exp-001", gain=0.05, visits=1)
        explored = NodeStats("exp-002", gain=0.05, visits=16)
        fresh_score = ucb1_score(fresh, best_gain=0.05, total_visits=20, explore=0.5)
        explored_score = ucb1_score(explored, best_gain=0.05, total_visits=20, explore=0.5)
        assert fresh_score > explored_score

    def test_unvisited_scored_as_one_visit(self) -> None:
        unvisited = NodeStats("exp-001", gain=0.0, visits=0)
        once = NodeStats("exp-002", gain=0.0, visits=1)
        assert ucb1_score(unvisited, 0.05, 20, 0.5) == ucb1_score(once, 0.05, 20, 0.5)

    def test_score_is_finite_on_an_empty_graph(self) -> None:
        node = NodeStats(BASELINE_NODE, gain=0.0, visits=0)
        assert ucb1_score(node, best_gain=0.0, total_visits=0, explore=0.5) > 0.0

    def test_larger_constant_gives_a_larger_bonus(self) -> None:
        node = NodeStats("exp-001", gain=0.0, visits=1)
        assert ucb1_score(node, 0.05, 20, 1.0) > ucb1_score(node, 0.05, 20, 0.25)


class TestSelectNode:
    def test_empty_graph_selects_nothing(self) -> None:
        assert select_node([], explore=0.5) is None

    def test_pure_exploitation_picks_the_best(self) -> None:
        nodes = [
            NodeStats(BASELINE_NODE, gain=0.0, visits=3),
            NodeStats("exp-001", gain=0.02, visits=0),
            NodeStats("exp-002", gain=0.06, visits=1),
        ]
        chosen = select_node(nodes, explore=0.0)
        assert chosen is not None
        assert chosen.node_id == "exp-002"

    def test_exploration_pivots_off_an_exhausted_best(self) -> None:
        """The best node has been expanded many times; an untried branch wins."""
        nodes = [
            NodeStats(BASELINE_NODE, gain=0.0, visits=2),
            NodeStats("exp-001", gain=0.06, visits=40),
            NodeStats("exp-002", gain=0.05, visits=1),
        ]
        chosen = select_node(nodes, explore=0.5)
        assert chosen is not None
        assert chosen.node_id == "exp-002"

    def test_exploration_still_prefers_a_clear_winner(self) -> None:
        nodes = [
            NodeStats(BASELINE_NODE, gain=0.0, visits=2),
            NodeStats("exp-001", gain=0.06, visits=1),
            NodeStats("exp-002", gain=0.0, visits=1),
        ]
        chosen = select_node(nodes, explore=0.5)
        assert chosen is not None
        assert chosen.node_id == "exp-001"

    def test_first_round_picks_the_baseline(self) -> None:
        chosen = select_node([NodeStats(BASELINE_NODE)], explore=0.5)
        assert chosen is not None
        assert chosen.node_id == BASELINE_NODE

    def test_nothing_has_worked_yet_so_start_fresh(self) -> None:
        """No node beat the baseline — compounding on a failure is worse than not."""
        nodes = [
            NodeStats(BASELINE_NODE, gain=0.0, visits=2),
            NodeStats("exp-001", gain=0.0, visits=2),
            NodeStats("exp-002", gain=0.0, visits=2),
        ]
        chosen = select_node(nodes, explore=0.0)
        assert chosen is not None
        assert chosen.node_id == BASELINE_NODE

    def test_ties_break_deterministically(self) -> None:
        nodes = [
            NodeStats("exp-002", gain=0.05, visits=1),
            NodeStats("exp-001", gain=0.05, visits=1),
        ]
        first = select_node(nodes, explore=0.5)
        second = select_node(list(reversed(nodes)), explore=0.5)
        assert first is not None and second is not None
        assert first.node_id == second.node_id == "exp-002"


class TestProposeMerge:
    def test_two_independent_winners_are_proposed(self) -> None:
        nodes = [
            NodeStats(BASELINE_NODE, gain=0.0),
            NodeStats("exp-001", gain=0.05),
            NodeStats("exp-002", gain=0.03),
        ]
        ancestors = {"exp-001": {BASELINE_NODE}, "exp-002": {BASELINE_NODE}}
        proposal = propose_merge(nodes, ancestors, already_merged=set())
        assert proposal is not None
        assert proposal.parents == ("exp-001", "exp-002")
        assert proposal.combined_gain == pytest.approx(0.08)

    def test_a_node_and_its_own_ancestor_are_not_merged(self) -> None:
        nodes = [NodeStats("exp-001", gain=0.03), NodeStats("exp-002", gain=0.05)]
        ancestors = {"exp-002": {BASELINE_NODE, "exp-001"}}
        assert propose_merge(nodes, ancestors, already_merged=set()) is None

    def test_losers_are_never_merged(self) -> None:
        nodes = [NodeStats("exp-001", gain=0.05), NodeStats("exp-002", gain=0.0)]
        ancestors = {"exp-001": set(), "exp-002": set()}
        assert propose_merge(nodes, ancestors, already_merged=set()) is None

    def test_a_pair_already_combined_is_skipped(self) -> None:
        nodes = [
            NodeStats("exp-001", gain=0.05),
            NodeStats("exp-002", gain=0.03),
            NodeStats("exp-003", gain=0.01),
        ]
        ancestors = {"exp-001": set(), "exp-002": set(), "exp-003": set()}
        proposal = propose_merge(
            nodes, ancestors, already_merged={frozenset({"exp-001", "exp-002"})}
        )
        assert proposal is not None
        assert proposal.parents == ("exp-001", "exp-003")

    def test_the_baseline_is_not_a_merge_parent(self) -> None:
        nodes = [NodeStats(BASELINE_NODE, gain=0.9), NodeStats("exp-001", gain=0.05)]
        assert propose_merge(nodes, {}, already_merged=set()) is None

    def test_highest_combined_gain_wins(self) -> None:
        nodes = [
            NodeStats("exp-001", gain=0.05),
            NodeStats("exp-002", gain=0.04),
            NodeStats("exp-003", gain=0.01),
        ]
        ancestors = {"exp-001": set(), "exp-002": set(), "exp-003": set()}
        proposal = propose_merge(nodes, ancestors, already_merged=set())
        assert proposal is not None
        assert proposal.parents == ("exp-001", "exp-002")

    def test_nothing_to_merge_on_an_empty_graph(self) -> None:
        assert propose_merge([], {}, already_merged=set()) is None
