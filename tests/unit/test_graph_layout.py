"""Laying out the experiment DAG: layers, rows, and edge routes.

These are the properties a reader depends on being true of the picture — an
edge never runs backwards, a long edge never crosses a card, siblings do not
collapse into one line — asserted on coordinates rather than on pixels.
"""

from researchforge.reporting.graph_layout import (
    ROOT_KEY,
    Box,
    Layout,
    Metrics,
    Route,
    assign_layers,
    layout_graph,
    resolved_parents,
)

METRICS = Metrics()


def _cards(layout: Layout) -> dict[str, Box]:
    return {box.key: box for box in layout.cards}


def _route(layout: Layout, parent: str, child: str) -> Route:
    return next(r for r in layout.routes if (r.parent, r.child) == (parent, child))


class TestResolvedParents:
    def test_a_node_without_parents_attaches_to_the_baseline(self) -> None:
        assert resolved_parents(["exp-001"], {}) == {"exp-001": [ROOT_KEY]}

    def test_a_declared_parent_is_kept(self) -> None:
        parents = resolved_parents(["exp-001", "exp-002"], {"exp-002": ["exp-001"]})
        assert parents["exp-002"] == ["exp-001"]

    def test_a_parent_outside_the_graph_is_dropped(self) -> None:
        parents = resolved_parents(["exp-002"], {"exp-002": ["exp-001"]})
        assert parents["exp-002"] == [ROOT_KEY]

    def test_a_partly_present_merge_keeps_only_what_is_drawn(self) -> None:
        parents = resolved_parents(["exp-001", "exp-003"], {"exp-003": ["exp-001", "exp-002"]})
        assert parents["exp-003"] == ["exp-001"]

    def test_a_repeated_parent_is_listed_once(self) -> None:
        parents = resolved_parents(["exp-001", "exp-002"], {"exp-002": ["exp-001", "exp-001"]})
        assert parents["exp-002"] == ["exp-001"]

    def test_a_node_cannot_be_its_own_parent(self) -> None:
        parents = resolved_parents(["exp-001"], {"exp-001": ["exp-001"]})
        assert parents["exp-001"] == [ROOT_KEY]


class TestAssignLayers:
    def test_the_baseline_is_the_source(self) -> None:
        layers = assign_layers(["exp-001"], {"exp-001": [ROOT_KEY]})
        assert layers[ROOT_KEY] == 0
        assert layers["exp-001"] == 1

    def test_a_chain_advances_one_layer_at_a_time(self) -> None:
        keys = ["exp-001", "exp-002", "exp-003"]
        parents = {
            "exp-001": [ROOT_KEY],
            "exp-002": ["exp-001"],
            "exp-003": ["exp-002"],
        }
        layers = assign_layers(keys, parents)
        assert [layers[key] for key in keys] == [1, 2, 3]

    def test_a_merge_lands_past_its_deepest_parent(self) -> None:
        keys = ["exp-001", "exp-002", "exp-003"]
        parents = {
            "exp-001": [ROOT_KEY],
            "exp-002": ["exp-001"],
            "exp-003": ["exp-001", "exp-002"],
        }
        layers = assign_layers(keys, parents)
        assert layers["exp-003"] == 3

    def test_declaration_order_does_not_matter(self) -> None:
        parents = {
            "exp-003": ["exp-002"],
            "exp-002": ["exp-001"],
            "exp-001": [ROOT_KEY],
        }
        layers = assign_layers(["exp-003", "exp-002", "exp-001"], parents)
        assert layers["exp-003"] == 3

    def test_a_cycle_is_parked_rather_than_looping_forever(self) -> None:
        parents = {"exp-001": ["exp-002"], "exp-002": ["exp-001"]}
        layers = assign_layers(["exp-001", "exp-002"], parents)
        assert layers["exp-001"] == layers["exp-002"]
        assert layers["exp-001"] > 0


class TestLayoutGeometry:
    def test_every_key_gets_a_card_plus_the_baseline(self) -> None:
        layout = layout_graph(["exp-001", "exp-002"], {})
        assert set(_cards(layout)) == {ROOT_KEY, "exp-001", "exp-002"}

    def test_an_empty_graph_still_places_the_baseline(self) -> None:
        layout = layout_graph([], {})
        assert [box.key for box in layout.cards] == [ROOT_KEY]
        assert layout.layers == 1

    def test_a_child_sits_to_the_right_of_its_parent(self) -> None:
        layout = layout_graph(["exp-001", "exp-002"], {"exp-002": ["exp-001"]})
        cards = _cards(layout)
        assert cards["exp-002"].x > cards["exp-001"].x

    def test_siblings_share_a_column(self) -> None:
        cards = _cards(layout_graph(["exp-001", "exp-002", "exp-003"], {}))
        assert cards["exp-001"].x == cards["exp-002"].x == cards["exp-003"].x

    def test_siblings_do_not_overlap(self) -> None:
        cards = _cards(layout_graph(["exp-001", "exp-002", "exp-003"], {}))
        first, second, third = sorted((cards["exp-001"].y, cards["exp-002"].y, cards["exp-003"].y))
        assert second - first >= METRICS.node_h
        assert third - second >= METRICS.node_h

    def test_every_edge_points_forward(self) -> None:
        keys = ["exp-001", "exp-002", "exp-003"]
        parents = {
            "exp-002": ["exp-001"],
            "exp-003": ["exp-001", "exp-002"],
        }
        layout = layout_graph(keys, parents)
        assert all(route.layer_span >= 1 for route in layout.routes)

    def test_a_route_starts_at_the_parent_and_ends_at_the_child(self) -> None:
        layout = layout_graph(["exp-001", "exp-002"], {"exp-002": ["exp-001"]})
        cards = _cards(layout)
        route = _route(layout, "exp-001", "exp-002")
        assert route.points[0] == (cards["exp-001"].x + METRICS.node_w, cards["exp-001"].center_y)
        assert route.points[-1] == (cards["exp-002"].x, cards["exp-002"].center_y)

    def test_the_canvas_contains_every_card(self) -> None:
        layout = layout_graph(["exp-001", "exp-002", "exp-003"], {"exp-003": ["exp-001"]})
        for card in layout.cards:
            assert card.x + METRICS.node_w <= layout.width
            assert card.y + card.height <= layout.height


class TestLongEdges:
    """An edge skipping a layer must route around the cards it passes."""

    def _spanning_layout(self) -> Layout:
        keys = ["exp-001", "exp-002", "exp-003", "exp-004"]
        parents = {
            "exp-002": ["exp-001"],
            "exp-003": ["exp-002"],
            "exp-004": ["exp-001", "exp-003"],
        }
        return layout_graph(keys, parents)

    def test_the_long_edge_is_recorded_as_spanning(self) -> None:
        layout = self._spanning_layout()
        assert _route(layout, "exp-001", "exp-004").layer_span == 3

    def test_the_long_edge_clears_every_card_it_passes(self) -> None:
        layout = self._spanning_layout()
        route = _route(layout, "exp-001", "exp-004")
        intermediate = [card for card in layout.cards if card.key in {"exp-002", "exp-003"}]
        for x, y in route.points:
            for card in intermediate:
                inside_x = card.x < x < card.x + METRICS.node_w
                inside_y = card.y < y < card.y + card.height
                assert not (inside_x and inside_y), f"({x},{y}) is inside {card.key}"

    def test_bends_are_routing_space_not_content(self) -> None:
        layout = self._spanning_layout()
        assert any(box.key.startswith("__bend__") for box in layout.boxes)
        assert not any(card.key.startswith("__bend__") for card in layout.cards)


class TestFanOut:
    def test_sibling_edges_share_one_trunk(self) -> None:
        """One parent's edges are a bracket, not lines to follow one by one."""
        layout = layout_graph(["exp-001", "exp-002", "exp-003"], {})
        turns = {route.points[1][0] for route in layout.routes if route.parent == ROOT_KEY}
        assert len(turns) == 1

    def test_two_parents_in_a_layer_turn_at_different_places(self) -> None:
        """Separate brackets in the same gutter must not read as one."""
        layout = layout_graph(
            ["exp-001", "exp-002", "exp-003", "exp-004"],
            {"exp-003": ["exp-001"], "exp-004": ["exp-002"]},
        )
        turns = {
            route.points[1][0] for route in layout.routes if route.parent in {"exp-001", "exp-002"}
        }
        assert len(turns) == 2

    def test_a_single_edge_is_not_nudged(self) -> None:
        layout = layout_graph(["exp-001"], {})
        route = layout.routes[0]
        expected = METRICS.pad + METRICS.node_w + METRICS.gap_x / 2
        assert route.points[1][0] == expected

    def test_nudges_stay_inside_the_gutter(self) -> None:
        layout = layout_graph([f"exp-{n:03d}" for n in range(1, 9)], {})
        root_right = METRICS.pad + METRICS.node_w
        for route in layout.routes:
            turn_x = route.points[1][0]
            assert root_right < turn_x < root_right + METRICS.gap_x


class TestWideLayers:
    """A round of many siblings should read as a set, not as a long list."""

    def _fan(self, count: int) -> Layout:
        return layout_graph([f"exp-{n:03d}" for n in range(1, count + 1)], {})

    def test_a_small_fan_stays_in_one_column(self) -> None:
        assert {box.slot for box in self._fan(4).cards} == {0}

    def test_a_large_fan_wraps_into_more_than_one_column(self) -> None:
        layout = self._fan(9)
        cards = [box for box in layout.cards if box.key != ROOT_KEY]

        assert {box.slot for box in cards} == {0, 1}
        assert len({box.x for box in cards}) == 2

    def test_wrapping_makes_the_picture_shorter_than_the_list(self) -> None:
        stacked = 9 * METRICS.node_h + 8 * METRICS.gap_y

        assert self._fan(9).height < stacked

    def test_an_edge_into_a_wrapped_column_clears_the_cards_beside_it(self) -> None:
        """It crosses above the layer instead of cutting through it."""
        layout = self._fan(9)
        first_column_top = min(
            box.y for box in layout.cards if box.slot == 0 and box.key != ROOT_KEY
        )
        crossing = [
            route
            for route in layout.routes
            if (box := layout.box(route.child)) is not None and box.slot == 1
        ]

        assert crossing
        for route in crossing:
            assert min(y for _, y in route.points) < first_column_top

    def test_a_layer_something_leaves_is_never_wrapped(self) -> None:
        """A card in a second sub-column has its own layer to its right."""
        keys = [f"exp-{n:03d}" for n in range(1, 9)]
        layout = layout_graph([*keys, "exp-009"], {"exp-009": ["exp-004"]})

        assert {box.slot for box in layout.cards if box.key in keys} == {0}


class TestCentring:
    def test_a_parent_sits_across_from_the_middle_of_its_children(self) -> None:
        layout = layout_graph(["exp-001", "exp-002", "exp-003"], {})
        children = [box for box in layout.cards if box.key != ROOT_KEY]
        root = layout.box(ROOT_KEY)

        assert root is not None
        middle = (min(b.center_y for b in children) + max(b.center_y for b in children)) / 2
        assert abs(root.center_y - middle) < 1.0

    def test_cards_in_a_column_never_overlap(self) -> None:
        layout = layout_graph(
            ["exp-001", "exp-002", "exp-003", "exp-004", "exp-005"],
            {"exp-004": ["exp-001"], "exp-005": ["exp-001"]},
        )
        columns: dict[tuple[int, int], list[Box]] = {}
        for box in layout.boxes:
            columns.setdefault((box.layer, box.slot), []).append(box)

        for members in columns.values():
            members.sort(key=lambda box: box.y)
            for upper, lower in zip(members, members[1:], strict=False):
                assert upper.y + upper.height <= lower.y


class TestDeterminism:
    def test_the_same_graph_lays_out_the_same_way_twice(self) -> None:
        keys = ["exp-001", "exp-002", "exp-003", "exp-004"]
        parents = {"exp-003": ["exp-001"], "exp-004": ["exp-001", "exp-002"]}
        first = layout_graph(keys, parents)
        second = layout_graph(keys, parents)
        assert first == second
