"""Graph traversal for the experiment DAG.

An experiment may build on more than one measured ancestor: two independent
winners can be merged into a single variant.  That turns the old parent chain
into a directed acyclic graph, and every consumer needs the same two answers
from it — in what order do a node's ancestors have to be applied, and does the
graph contain a cycle.

Both answers come from one depth-first post-order walk: parents are emitted
before the nodes that depend on them, a diamond emits its shared root once,
and a cycle is reported with the path that closes it instead of looping.
"""

from __future__ import annotations

from collections.abc import Callable

ParentsFn = Callable[[str], list[str]]
"""Answers "which nodes does this one build on", by id."""


class GraphCycleError(ValueError):
    """The parent graph is not acyclic."""

    def __init__(self, path: tuple[str, ...]) -> None:
        self.path = path
        super().__init__(" → ".join(path))


def ancestor_order(node_id: str, parents_of: ParentsFn) -> list[str]:
    """A node's ancestors in dependency order, roots first, each one once.

    The node itself is not included — callers apply their own change last.
    """
    order: list[str] = []
    resolved: set[str] = set()

    def visit(current: str, path: tuple[str, ...]) -> None:
        if current in resolved:
            return
        if current in path:
            raise GraphCycleError((*path, current))
        for parent in parents_of(current):
            visit(parent, (*path, current))
        resolved.add(current)
        order.append(current)

    for parent in parents_of(node_id):
        visit(parent, (node_id,))
    return order
