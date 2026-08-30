"""Placing the experiment graph on a canvas, without a JavaScript layout library.

The dashboard is a single self-contained HTML file with no bundler and no CDN,
so the graph has to be positioned in Python before the SVG is written.  Column
= depth is not enough once experiments have several parents: a merge of an early
and a late winner spans layers, and an edge drawn straight across would cut
through whatever cards happen to sit between them.

This is the classic layered approach, kept to the three steps that earn their
place:

1. **Layering** by longest path from the baseline, so every edge points forward
   at least one layer and no edge ever runs backwards.
2. **Bend points** for edges that skip a layer.  Each one reserves a real slot
   in the layer it crosses, which is what keeps a long edge out of the cards it
   passes — the guarantee a midpoint-routed line cannot make.
3. **Ordering** within each layer by the average position of a node's
   neighbours, swept a few times.  This is a heuristic for fewer crossings, not
   a minimum; crossing minimisation is NP-hard and a readable graph does not
   need the optimum.

Everything here is pure and works on opaque keys: no experiment, no SVG, no
colours.  The input may come from a hand-edited database, so a graph that is
not acyclic is laid out awkwardly rather than hanging or raising.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

BARYCENTER_SWEEPS = 4
"""Ordering passes. Four is where the crossing count stops improving on graphs
of the size a research project produces (tens of nodes, not thousands)."""

ROOT_KEY = "__baseline__"


@dataclass(frozen=True)
class Metrics:
    """Card size and spacing, in user units of the SVG viewBox."""

    node_w: float = 208
    node_h: float = 76
    gap_x: float = 64
    gap_y: float = 18
    pad: float = 18
    bend_h: float = 22
    """Row height for a bend point. Smaller than a card because nothing is
    drawn there — it is reserved space for an edge to pass through."""


@dataclass(frozen=True)
class Box:
    """One laid-out slot: an experiment card, the baseline root, or a bend."""

    key: str
    layer: int
    row: int
    x: float
    y: float
    height: float
    is_bend: bool = False
    slot: int = 0
    """Which sub-column of its layer. Non-zero only where a wide layer was
    wrapped into a grid instead of running off the bottom of the page."""

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class Route:
    """One parent→child edge, as the polyline the SVG should draw."""

    parent: str
    child: str
    points: tuple[tuple[float, float], ...]
    layer_span: int
    """How many layers the edge crosses. More than one means it was routed
    through bend points rather than straight across."""


@dataclass(frozen=True)
class Layout:
    boxes: tuple[Box, ...]
    routes: tuple[Route, ...]
    width: float
    height: float
    layers: int

    @property
    def cards(self) -> tuple[Box, ...]:
        """The boxes that get drawn — bends are routing space, not content."""
        return tuple(box for box in self.boxes if not box.is_bend)

    def box(self, key: str) -> Box | None:
        return next((box for box in self.boxes if box.key == key), None)


def resolved_parents(
    keys: Sequence[str], declared: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """Each node's parents, restricted to nodes actually being drawn.

    A parent outside this graph — an experiment from another plan, or one
    deleted since — is dropped, and a node left with none attaches to the
    baseline instead. The alternative is an edge to nowhere.
    """
    present = set(keys)
    parents: dict[str, list[str]] = {}
    for key in keys:
        known = [p for p in dict.fromkeys(declared.get(key, ())) if p in present and p != key]
        parents[key] = known or [ROOT_KEY]
    return parents


def assign_layers(keys: Sequence[str], parents: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Layer per node: one past its deepest parent, with the baseline at 0.

    Nodes are settled in dependency order, so a node is placed only once every
    parent has been. Anything left unsettled is part of a cycle and cannot be
    layered honestly; those go in a final layer of their own, in key order, so
    a corrupt graph still renders.
    """
    layer = {ROOT_KEY: 0}
    pending = list(keys)
    while pending:
        settled = [key for key in pending if all(p in layer for p in parents[key])]
        if not settled:
            break
        for key in settled:
            layer[key] = 1 + max(layer[p] for p in parents[key])
        pending = [key for key in pending if key not in layer]

    if pending:
        stranded = 1 + max(layer.values())
        for key in sorted(pending):
            layer[key] = stranded
    return layer


def _bend_key(parent: str, child: str, layer: int) -> str:
    return f"__bend__{parent}->{child}@{layer}"


def _chains(
    parents: Mapping[str, Sequence[str]], layer: Mapping[str, int]
) -> dict[tuple[str, str], list[str]]:
    """The full path of each edge, with a bend key per layer it skips."""
    chains: dict[tuple[str, str], list[str]] = {}
    for child, sources in parents.items():
        for parent in sources:
            bends = [
                _bend_key(parent, child, crossed)
                for crossed in range(layer[parent] + 1, layer[child])
            ]
            chains[parent, child] = [parent, *bends, child]
    return chains


def _barycenter(
    key: str,
    links: Mapping[str, Sequence[str]],
    row: Mapping[str, int],
    layer: Mapping[str, int],
    depth: int,
) -> float:
    """Average row of a node's neighbours outside its own layer.

    A node with no neighbours elsewhere has nothing pulling on it, so it keeps
    the row it already has.
    """
    neighbours = [row[n] for n in links[key] if n in row and layer[n] != depth]
    return sum(neighbours) / len(neighbours) if neighbours else float(row[key])


def _order_rows(
    layer: Mapping[str, int],
    links: Mapping[str, list[str]],
    seed: Sequence[str],
) -> dict[str, int]:
    """Row per node, from repeated barycenter sweeps over the layered graph.

    `links` is symmetric — a node is pulled towards its parents and its
    children alike — because ordering a layer well means answering to both
    sides of it. Ties keep the previous row, so the result is deterministic and
    a stable graph does not reshuffle between dashboard builds.
    """
    layers: dict[int, list[str]] = {}
    for key in seed:
        layers.setdefault(layer[key], []).append(key)

    row = {key: index for keys in layers.values() for index, key in enumerate(keys)}
    for _ in range(BARYCENTER_SWEEPS):
        for depth in sorted(layers):
            members = layers[depth]
            members.sort(key=lambda key: (_barycenter(key, links, row, layer, depth), row[key]))
            for index, key in enumerate(members):
                row[key] = index
    return row


def layout_graph(
    keys: Sequence[str],
    declared_parents: Mapping[str, Sequence[str]],
    metrics: Metrics | None = None,
) -> Layout:
    """Position the baseline root, one card per key, and every edge between them.

    `keys` fixes the tie-break order, so callers control what "first" means
    (the dashboard passes experiments in id order). The baseline root is added
    automatically as the graph's single source.
    """
    metrics = metrics or Metrics()
    parents = resolved_parents(keys, declared_parents)
    layer = assign_layers(keys, parents)
    chains = _chains(parents, layer)

    bends = list(dict.fromkeys(key for chain in chains.values() for key in chain[1:-1]))
    for key in bends:
        layer[key] = _layer_from_bend(key)

    links: dict[str, list[str]] = {key: [] for key in (ROOT_KEY, *keys, *bends)}
    for chain in chains.values():
        for upper, lower in zip(chain, chain[1:], strict=False):
            links[upper].append(lower)
            links[lower].append(upper)

    seed = [ROOT_KEY, *keys, *bends]
    row = _order_rows(layer, links, seed)

    heights = dict.fromkeys(bends, metrics.bend_h)
    columns: dict[int, list[str]] = {}
    for key in seed:
        columns.setdefault(layer[key], []).append(key)

    slots = _slots_per_layer(columns, chains, heights)
    top = metrics.pad + rail_band(slots)

    boxes = _place(seed, layer, row, heights, metrics, slots, top)
    boxes = _centre_on_children(boxes, chains, metrics, top)
    by_key = {box.key: box for box in boxes}

    offsets = _trunk_offsets(chains, layer, row, metrics)
    routes = tuple(
        Route(
            parent=parent,
            child=child,
            points=_polyline([by_key[key] for key in chain], metrics, offsets[parent], top),
            layer_span=layer[child] - layer[parent],
        )
        for (parent, child), chain in sorted(chains.items())
    )

    depth = max(layer.values())
    return Layout(
        boxes=boxes,
        routes=routes,
        width=metrics.pad + max((box.x + metrics.node_w for box in boxes), default=0),
        height=metrics.pad + max((box.y + box.height for box in boxes), default=0),
        layers=depth + 1,
    )


def _layer_from_bend(key: str) -> int:
    return int(key.rsplit("@", 1)[1])


MAX_COLUMN_ROWS = 5
"""Cards in one sub-column before a layer wraps. Past roughly this many, a
round reads as a list running off the page rather than as a set of siblings."""


def _slots_per_layer(
    columns: Mapping[int, Sequence[str]],
    chains: Mapping[tuple[str, str], Sequence[str]],
    heights: Mapping[str, float],
) -> dict[int, int]:
    """How many sub-columns each layer is split into.

    Only a layer nothing leaves is ever wrapped. A node in a second sub-column
    has cards to its right that belong to its own layer, so an edge out of it
    would have to cross them; a layer of leaves — the usual shape of a freshly
    planned round — has no such edge to draw.
    """
    departures = {parent for parent, _ in chains}
    slots: dict[int, int] = {}
    for depth, members in columns.items():
        wrappable = (
            len(members) > MAX_COLUMN_ROWS
            and not any(key in heights for key in members)
            and not any(key in departures for key in members)
        )
        slots[depth] = -(-len(members) // MAX_COLUMN_ROWS) if wrappable else 1
    return slots


def _place(
    keys: Sequence[str],
    layer: Mapping[str, int],
    row: Mapping[str, int],
    heights: Mapping[str, float],
    metrics: Metrics,
    slots: Mapping[int, int],
    top: float,
) -> tuple[Box, ...]:
    """Turn (layer, row) into coordinates, stacking each column by its heights.

    Rows are stacked rather than multiplied by a fixed pitch so a bend costs
    only the space it needs, and the columns stay visually aligned because
    every column starts at the same top edge. A wrapped layer occupies several
    sub-columns, which pushes every later layer further right.
    """
    columns: dict[int, list[str]] = {}
    for key in keys:
        columns.setdefault(layer[key], []).append(key)

    step = metrics.node_w + metrics.gap_x
    left = dict.fromkeys(columns, metrics.pad)
    for depth in sorted(columns):
        if depth + 1 in left:
            left[depth + 1] = left[depth] + slots.get(depth, 1) * step

    boxes: list[Box] = []
    for depth, members in sorted(columns.items()):
        members.sort(key=lambda key: row[key])
        per_slot = -(-len(members) // slots.get(depth, 1))
        offsets = dict.fromkeys(range(slots.get(depth, 1)), top)
        for index, member in enumerate(members):
            slot = index // per_slot if per_slot else 0
            height = heights.get(member, metrics.node_h)
            boxes.append(
                Box(
                    key=member,
                    layer=depth,
                    row=row[member],
                    x=left[depth] + slot * step,
                    y=offsets[slot],
                    height=height,
                    is_bend=member in heights,
                    slot=slot,
                )
            )
            offsets[slot] += height + metrics.gap_y
    return tuple(boxes)


def _centre_on_children(
    boxes: Sequence[Box],
    chains: Mapping[tuple[str, str], Sequence[str]],
    metrics: Metrics,
    top: float,
) -> tuple[Box, ...]:
    """Move each node opposite the middle of what it leads to.

    Stacking every column from the top leaves a parent pinned to the first row
    while its children run far down the page, and the eye reads the picture as
    a list with wires attached. Sitting a parent across from the centre of its
    children is what makes a fan look like the bracket it is.

    Columns are swept right to left, since a node's position depends on
    children already placed. Within a column the previous order is kept and
    overlaps are pushed down, so nothing here can make two cards collide.
    """
    by_key = {box.key: box for box in boxes}
    children: dict[str, list[str]] = {}
    for parent, child in chains:
        children.setdefault(parent, []).append(child)

    columns: dict[tuple[int, int], list[Box]] = {}
    for box in boxes:
        columns.setdefault((box.layer, box.slot), []).append(box)

    for depth in sorted(columns, reverse=True):
        members = sorted(columns[depth], key=lambda box: box.row)
        desired = []
        for box in members:
            targets = [by_key[key].center_y for key in children.get(box.key, []) if key in by_key]
            centre = sum(targets) / len(targets) if targets else box.center_y
            desired.append(centre - box.height / 2)

        placed: list[float] = []
        cursor = top
        for box, want in zip(members, desired, strict=True):
            start = max(want, cursor)
            placed.append(start)
            cursor = start + box.height + metrics.gap_y

        # Greedy placement can only push down, which drifts a column away from
        # the children it was aimed at. Sliding it back as a whole keeps the
        # aim without reintroducing an overlap.
        drift = min(
            (start - want for start, want in zip(placed, desired, strict=True)),
            default=0.0,
        )
        slide = min(drift, min(placed, default=top) - top)
        for box, start in zip(members, placed, strict=True):
            by_key[box.key] = Box(
                key=box.key,
                layer=box.layer,
                row=box.row,
                x=box.x,
                y=start - slide,
                height=box.height,
                is_bend=box.is_bend,
                slot=box.slot,
            )
        columns[depth] = [by_key[box.key] for box in members]

    return tuple(by_key[box.key] for box in boxes)


MAX_TRUNK_OFFSET = 9.0
"""How far apart two parents' trunks are placed inside a gutter, at most."""


def _trunk_offsets(
    chains: Mapping[tuple[str, str], Sequence[str]],
    layer: Mapping[str, int],
    row: Mapping[str, int],
    metrics: Metrics,
) -> dict[str, float]:
    """The gutter position each node's outgoing edges turn at.

    Every edge leaving a node shares one vertical trunk, so a parent with nine
    children draws as a single bracket with nine stubs rather than nine near
    parallel lines nobody can follow to its end. What has to stay apart is
    different parents turning in the same gutter, so the offset is per node,
    spread by row within its layer.
    """
    senders: dict[int, list[str]] = {}
    for parent, _child in chains:
        column = senders.setdefault(layer[parent], [])
        if parent not in column:
            column.append(parent)

    offsets: dict[str, float] = {}
    for column in senders.values():
        column.sort(key=lambda key: (row.get(key, 0), key))
        step = min(MAX_TRUNK_OFFSET, metrics.gap_x / (len(column) + 1))
        for index, parent in enumerate(column):
            offsets[parent] = (index - (len(column) - 1) / 2) * step
    return offsets


RAIL_STEP = 7.0
"""Vertical spacing between the rails that feed wrapped sub-columns."""

RAIL_CLEARANCE = 11.0
"""Gap between the first rail and the top of the cards it passes over."""


def rail_band(slots: Mapping[int, int]) -> float:
    """Space above the cards for the rails that reach wrapped sub-columns."""
    deepest = max(slots.values(), default=1)
    return 0.0 if deepest < 2 else RAIL_CLEARANCE + (deepest - 2) * RAIL_STEP


def _polyline(
    chain: Sequence[Box], metrics: Metrics, nudge: float = 0.0, top: float = 0.0
) -> tuple[tuple[float, float], ...]:
    """An orthogonal path from the parent's right edge to the child's left edge.

    Every vertical run happens in the gutter between two columns, so a line
    never crosses the column a card lives in. `nudge` places the trunk each
    parent turns in, so one parent's edges leave as a single bracket.

    Reaching a wrapped sub-column is the one case a gutter cannot serve on its
    own: the cards of the same layer sit in between. Those edges climb to a
    rail above the layer, cross there, and come down the gutter beside their
    own sub-column — `top` is where the cards begin, so the rails stay clear
    of them.
    """
    points: list[tuple[float, float]] = []
    for index, (upper, lower) in enumerate(zip(chain, chain[1:], strict=False)):
        gutter = upper.x + metrics.node_w + metrics.gap_x / 2 + (nudge if index == 0 else 0.0)
        if index == 0:
            points.append((upper.x + metrics.node_w, upper.center_y))
        if lower.slot > upper.slot:
            rail = top - RAIL_CLEARANCE - (lower.slot - 1) * RAIL_STEP
            points.extend(
                [
                    (gutter, upper.center_y),
                    (gutter, rail),
                    (lower.x - metrics.gap_x / 2, rail),
                    (lower.x - metrics.gap_x / 2, lower.center_y),
                    (lower.x, lower.center_y),
                ]
            )
            continue
        points.extend(
            [
                (gutter, upper.center_y),
                (gutter, lower.center_y),
                (lower.x, lower.center_y),
            ]
        )
    return tuple(dict.fromkeys(points))
