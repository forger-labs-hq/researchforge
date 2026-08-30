"""What the loop is still allowed to try, and in what order.

The properties here are the difference between a search that expands a graph
and a queue that drains: the same idea may be worth trying again somewhere
else, and never worth trying twice in the same place.
"""

from researchforge.autorun.frontier import (
    Attempt,
    available_hypotheses,
    lineage_hypotheses,
    rank_hypotheses,
    round_breadth,
)
from researchforge.experiments.selection import BASELINE_NODE

ALL = ["hyp-001", "hyp-002", "hyp-003"]

ANCESTORS = {
    "exp-008": {BASELINE_NODE},
    "exp-012": {"exp-008", BASELINE_NODE},
}
HYPOTHESIS_OF = {"exp-008": "hyp-003", "exp-012": "hyp-001"}


class TestLineage:
    def test_the_baseline_has_no_history(self) -> None:
        assert lineage_hypotheses(BASELINE_NODE, ANCESTORS, HYPOTHESIS_OF) == set()

    def test_a_node_counts_its_own_hypothesis(self) -> None:
        assert lineage_hypotheses("exp-008", ANCESTORS, HYPOTHESIS_OF) == {"hyp-003"}

    def test_a_deeper_node_counts_the_whole_chain(self) -> None:
        assert lineage_hypotheses("exp-012", ANCESTORS, HYPOTHESIS_OF) == {"hyp-001", "hyp-003"}


class TestAvailability:
    def test_an_untouched_node_offers_everything(self) -> None:
        assert available_hypotheses(ALL, "exp-008", [], set()) == ALL

    def test_a_pair_is_only_tried_once(self) -> None:
        attempts = [Attempt("hyp-002", "exp-008", gain=0.0)]

        assert available_hypotheses(ALL, "exp-008", attempts, set()) == ["hyp-001", "hyp-003"]

    def test_the_same_hypothesis_is_still_open_elsewhere(self) -> None:
        """This is the whole point: a different parent is a different experiment."""
        attempts = [Attempt("hyp-002", BASELINE_NODE, gain=0.0)]

        assert "hyp-002" in available_hypotheses(ALL, "exp-008", attempts, set())

    def test_a_hypothesis_already_in_the_lineage_is_refused(self) -> None:
        """Applying it on top of itself either conflicts or changes nothing."""
        lineage = lineage_hypotheses("exp-008", ANCESTORS, HYPOTHESIS_OF)

        assert available_hypotheses(ALL, "exp-008", [], lineage) == ["hyp-001", "hyp-002"]

    def test_the_baseline_cannot_repeat_itself(self) -> None:
        """It has no lineage, so only the pair rule stops an exact re-run."""
        attempts = [Attempt(h, BASELINE_NODE, gain=0.0) for h in ALL]

        assert available_hypotheses(ALL, BASELINE_NODE, attempts, set()) == []


class TestRanking:
    def test_what_has_worked_comes_first(self) -> None:
        attempts = [
            Attempt("hyp-001", BASELINE_NODE, gain=0.0),
            Attempt("hyp-003", BASELINE_NODE, gain=0.009),
        ]

        assert rank_hypotheses(ALL, attempts)[0] == "hyp-003"

    def test_an_untried_idea_outranks_one_that_failed(self) -> None:
        attempts = [Attempt("hyp-001", BASELINE_NODE, gain=0.0)]

        ranked = rank_hypotheses(["hyp-001", "hyp-002"], attempts)

        assert ranked == ["hyp-002", "hyp-001"]

    def test_a_hypothesis_that_never_changed_anything_sinks(self) -> None:
        """Measuring its parent's exact value means the knob is not wired up."""
        attempts = [
            Attempt("hyp-001", BASELINE_NODE, gain=0.0, no_op=True),
            Attempt("hyp-001", "exp-008", gain=0.0, no_op=True),
            Attempt("hyp-002", BASELINE_NODE, gain=0.0),
        ]

        assert rank_hypotheses(["hyp-001", "hyp-002"], attempts) == ["hyp-002", "hyp-001"]

    def test_the_strongest_result_wins_among_proven_ideas(self) -> None:
        attempts = [
            Attempt("hyp-001", BASELINE_NODE, gain=0.002),
            Attempt("hyp-002", BASELINE_NODE, gain=0.009),
        ]

        assert rank_hypotheses(["hyp-001", "hyp-002"], attempts) == ["hyp-002", "hyp-001"]

    def test_an_unmeasured_attempt_does_not_count_as_a_result(self) -> None:
        """A run that crashed says nothing about the idea."""
        attempts = [Attempt("hyp-001", BASELINE_NODE, gain=None)]

        assert rank_hypotheses(["hyp-001", "hyp-002"], attempts) == ["hyp-002", "hyp-001"]

    def test_the_order_is_stable_for_the_same_history(self) -> None:
        attempts = [Attempt("hyp-002", BASELINE_NODE, gain=0.009)]

        assert rank_hypotheses(ALL, attempts) == rank_hypotheses(ALL, attempts)


class TestBreadth:
    def test_nothing_available_costs_nothing(self) -> None:
        assert round_breadth(0, 480, 60) == 0

    def test_without_a_budget_a_round_takes_one_move(self) -> None:
        assert round_breadth(4, None, 60) == 1

    def test_a_budget_pays_for_as_many_as_it_can(self) -> None:
        assert round_breadth(4, 480, 120) == 4
        assert round_breadth(4, 200, 120) == 1

    def test_it_never_commits_to_more_than_exists(self) -> None:
        assert round_breadth(2, 4800, 60) == 2

    def test_a_round_always_tries_something(self) -> None:
        """Better one experiment that may overrun than a round that does nothing."""
        assert round_breadth(3, 10, 120) == 1
