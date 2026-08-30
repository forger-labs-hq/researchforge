"""Ancestor ordering over the experiment DAG."""

import pytest

from researchforge.experiments.graph import GraphCycleError, ancestor_order

# a → b ↘
#  ↘ c → d    (d merges the two branches that fork off a)
DIAMOND = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}


class TestAncestorOrder:
    def test_no_parents_is_empty(self) -> None:
        assert ancestor_order("a", lambda node: DIAMOND[node]) == []

    def test_single_parent(self) -> None:
        assert ancestor_order("b", lambda node: DIAMOND[node]) == ["a"]

    def test_chain_is_root_first(self) -> None:
        chain = {"one": [], "two": ["one"], "three": ["two"]}
        assert ancestor_order("three", lambda node: chain[node]) == ["one", "two"]

    def test_node_itself_is_excluded(self) -> None:
        assert "d" not in ancestor_order("d", lambda node: DIAMOND[node])

    def test_shared_root_appears_once(self) -> None:
        assert ancestor_order("d", lambda node: DIAMOND[node]).count("a") == 1

    def test_merge_puts_root_before_both_branches(self) -> None:
        order = ancestor_order("d", lambda node: DIAMOND[node])
        assert order[0] == "a"
        assert set(order[1:]) == {"b", "c"}

    def test_declaration_order_breaks_ties(self) -> None:
        graph = {"a": [], "left": ["a"], "right": ["a"], "merge": ["right", "left"]}
        assert ancestor_order("merge", lambda node: graph[node]) == ["a", "right", "left"]

    def test_unknown_parent_is_a_leaf_not_an_error(self) -> None:
        """Whether a node exists is the caller's business; traversal only orders."""
        graph = {"x": ["ghost"]}
        assert ancestor_order("x", lambda node: graph.get(node, [])) == ["ghost"]


class TestCycles:
    def test_two_node_cycle(self) -> None:
        graph = {"a": ["b"], "b": ["a"]}
        with pytest.raises(GraphCycleError):
            ancestor_order("a", lambda node: graph[node])

    def test_self_parent(self) -> None:
        graph = {"a": ["a"]}
        with pytest.raises(GraphCycleError):
            ancestor_order("a", lambda node: graph[node])

    def test_longer_cycle(self) -> None:
        graph = {"a": ["c"], "b": ["a"], "c": ["b"]}
        with pytest.raises(GraphCycleError):
            ancestor_order("a", lambda node: graph[node])

    def test_error_names_the_closing_path(self) -> None:
        graph = {"a": ["b"], "b": ["a"]}
        with pytest.raises(GraphCycleError) as caught:
            ancestor_order("a", lambda node: graph[node])
        assert caught.value.path == ("a", "b", "a")
        assert "a → b → a" in str(caught.value)
