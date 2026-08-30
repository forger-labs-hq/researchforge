"""What the experiment-graph SVG promises a reader: badges, legend, tooltips.

The SVG is parsed rather than string-matched wherever the assertion is about
structure, so a change in attribute order or spacing does not break a test that
is really about "this edge is dotted".
"""

from xml.etree import ElementTree

from researchforge.reporting.svg_charts import GraphNode, graph_chart

SVG = "{http://www.w3.org/2000/svg}"

NODES = [
    GraphNode("exp-001", "Lower the threshold", "promising", 0.85, 6.2, [], 1),
    GraphNode("exp-002", "Swap the backbone", "rejected", 0.79, -1.3, [], 1),
    GraphNode("exp-003", "Threshold plus warmup", "promising", 0.87, 8.7, ["exp-001"], 2),
    GraphNode(
        "exp-004",
        "Both winners combined",
        "validated",
        0.89,
        11.2,
        ["exp-003", "exp-002"],
        3,
    ),
]


def _render(**kwargs: object) -> ElementTree.Element:
    svg = graph_chart(NODES, baseline_value=0.80, metric_name="f1", **kwargs)  # type: ignore[arg-type]
    return ElementTree.fromstring(svg)


def _edges(root: ElementTree.Element) -> dict[str, ElementTree.Element]:
    return {
        element.get("data-graph-edge", ""): element
        for element in root.iter(f"{SVG}polyline")
        if element.get("data-graph-edge")
    }


def _cards(root: ElementTree.Element) -> dict[str, ElementTree.Element]:
    return {
        element.get("data-graph-node", ""): element
        for element in root.iter(f"{SVG}rect")
        if element.get("data-graph-node")
    }


class TestStructure:
    def test_the_svg_is_well_formed(self) -> None:
        assert _render().tag == f"{SVG}svg"

    def test_every_experiment_and_the_baseline_get_a_card(self) -> None:
        assert set(_cards(_render())) == {
            "baseline",
            "exp-001",
            "exp-002",
            "exp-003",
            "exp-004",
        }

    def test_every_declared_parent_gets_an_edge(self) -> None:
        assert set(_edges(_render())) == {
            "baseline->exp-001",
            "baseline->exp-002",
            "exp-001->exp-003",
            "exp-003->exp-004",
            "exp-002->exp-004",
        }

    def test_a_card_carries_its_status_and_round(self) -> None:
        card = _cards(_render())["exp-004"]
        assert card.get("data-status") == "validated"
        assert card.get("data-round") == "3"

    def test_a_project_without_rounds_has_no_round_attribute(self) -> None:
        svg = graph_chart(
            [GraphNode("exp-001", "Solo", "promising", 0.85, 6.2)],
            baseline_value=0.80,
            metric_name="f1",
        )
        card = _cards(ElementTree.fromstring(svg))["exp-001"]
        assert card.get("data-round") is None


class TestRoundGrouping:
    def test_one_rounds_experiments_sit_together_in_a_layer(self) -> None:
        """Layer membership is the graph's; the row order is free to group rounds."""
        svg = graph_chart(
            [
                GraphNode("exp-001", "Round one", "promising", 0.85, 6.2, [], 1),
                GraphNode("exp-002", "Round two", "rejected", 0.79, -1.3, [], 2),
                GraphNode("exp-003", "Round one again", "promising", 0.86, 7.5, [], 1),
            ],
            baseline_value=0.80,
            metric_name="f1",
        )
        root = ElementTree.fromstring(svg)
        tops = {
            card.get("data-graph-node"): float(card.get("y", 0))
            for card in _cards(root).values()
        }
        assert tops["exp-001"] < tops["exp-003"] < tops["exp-002"]


class TestMergeEdges:
    def test_edges_into_a_merge_are_dotted(self) -> None:
        edges = _edges(_render())
        assert edges["exp-003->exp-004"].get("stroke-dasharray")
        assert edges["exp-002->exp-004"].get("stroke-dasharray")

    def test_edges_into_an_ordinary_child_are_solid(self) -> None:
        edges = _edges(_render())
        assert edges["exp-001->exp-003"].get("stroke-dasharray") is None
        assert edges["baseline->exp-001"].get("stroke-dasharray") is None

    def test_a_merge_card_says_so(self) -> None:
        svg = graph_chart(NODES, baseline_value=0.80, metric_name="f1")
        assert "MERGE" in svg


class TestPathToBest:
    def test_the_whole_ancestry_of_the_best_is_highlighted(self) -> None:
        edges = _edges(_render(best_experiment_id="exp-004"))
        highlighted = {
            name for name, edge in edges.items() if edge.get("stroke-width") == "2.5"
        }
        assert highlighted == {
            "baseline->exp-001",
            "baseline->exp-002",
            "exp-001->exp-003",
            "exp-003->exp-004",
            "exp-002->exp-004",
        }

    def test_a_branch_off_the_best_path_is_not_highlighted(self) -> None:
        edges = _edges(_render(best_experiment_id="exp-003"))
        assert edges["exp-003->exp-004"].get("stroke-width") != "2.5"
        assert edges["baseline->exp-002"].get("stroke-width") != "2.5"

    def test_nothing_is_highlighted_without_a_best(self) -> None:
        edges = _edges(_render())
        assert all(edge.get("stroke-width") != "2.5" for edge in edges.values())


class TestBadgesAndLegend:
    def test_a_measured_experiment_shows_its_delta(self) -> None:
        root = _render()
        deltas = {
            element.get("data-graph-delta"): element.text
            for element in root.iter(f"{SVG}text")
            if element.get("data-graph-delta")
        }
        assert deltas["exp-001"] == "+6.2%"
        assert deltas["exp-002"] == "-1.3%"

    def test_an_unmeasured_experiment_says_so_instead_of_showing_zero(self) -> None:
        svg = graph_chart(
            [GraphNode("exp-001", "Crashed", "failed_execution")],
            baseline_value=0.80,
            metric_name="f1",
        )
        assert "not measured" in svg
        assert "data-graph-delta" not in svg

    def test_the_legend_names_every_colour_and_the_merge_style(self) -> None:
        svg = graph_chart(NODES, baseline_value=0.80, metric_name="f1")
        assert "LEGEND" in svg
        for label in ("baseline", "shipped / validated", "kept", "rejected"):
            assert label in svg
        assert "merge (multi-parent)" in svg


class TestInheritedResults:
    """A card's percentage is against the baseline, so a child can wear a gain
    it did not earn. The graph has to say when that is what happened."""

    INHERITED = [
        GraphNode("exp-008", "Longer fine-tuning", "promising", 0.838, 1.1, [], 1),
        GraphNode("exp-014", "Score threshold", "promising", 0.838, 1.1, ["exp-008"], 2),
    ]

    def _svg(self) -> str:
        return graph_chart(self.INHERITED, baseline_value=0.829, metric_name="map50")

    def test_a_child_that_ties_its_parent_is_badged_no_change(self) -> None:
        assert "KEPT · NO CHANGE" in self._svg()

    def test_the_parent_that_earned_the_gain_is_not(self) -> None:
        svg = self._svg()

        assert svg.count("NO CHANGE") == 1
        assert "KEPT" in svg

    def test_the_tooltip_says_the_metric_was_inherited(self) -> None:
        assert "no change vs exp-008" in self._svg()

    def test_a_child_that_moved_the_metric_is_left_alone(self) -> None:
        svg = graph_chart(
            [
                GraphNode("exp-008", "Longer fine-tuning", "promising", 0.838, 1.1, [], 1),
                GraphNode("exp-014", "Score threshold", "promising", 0.841, 1.4, ["exp-008"], 2),
            ],
            baseline_value=0.829,
            metric_name="map50",
        )

        assert "NO CHANGE" not in svg

    def test_an_experiment_that_ties_the_baseline_says_so_too(self) -> None:
        svg = graph_chart(
            [GraphNode("exp-001", "Knob nothing reads", "rejected", 0.829, 0.0, [], 1)],
            baseline_value=0.829,
            metric_name="map50",
        )

        assert "REJECTED · NO CHANGE" in svg
        assert "no change vs the baseline" in svg

    def test_an_unmeasured_parent_makes_no_claim_either_way(self) -> None:
        svg = graph_chart(
            [
                GraphNode("exp-008", "Never finished", "failed_execution", None, None, [], 1),
                GraphNode("exp-014", "Score threshold", "promising", 0.838, 1.1, ["exp-008"], 2),
            ],
            baseline_value=0.829,
            metric_name="map50",
        )

        assert "NO CHANGE" not in svg


class TestTooltips:
    def _tooltip(self, experiment_id: str) -> str:
        root = _render()
        titles = [element.text or "" for element in root.iter(f"{SVG}title")]
        return next(text for text in titles if text.startswith(experiment_id))

    def test_a_tooltip_carries_the_untruncated_title_and_metric(self) -> None:
        text = self._tooltip("exp-004")
        assert "Both winners combined" in text
        assert "f1: 0.89" in text
        assert "+11.2% vs baseline" in text

    def test_a_tooltip_names_every_parent_of_a_merge(self) -> None:
        text = self._tooltip("exp-004")
        assert "exp-003" in text
        assert "exp-002" in text

    def test_a_root_experiment_reports_the_baseline_as_its_parent(self) -> None:
        assert "builds on: baseline" in self._tooltip("exp-001")

    def test_an_observation_reaches_the_tooltip(self) -> None:
        svg = graph_chart(
            [
                GraphNode(
                    "exp-001",
                    "Solo",
                    "promising",
                    0.85,
                    6.2,
                    observation="Loss plateaued at epoch 3.",
                )
            ],
            baseline_value=0.80,
            metric_name="f1",
        )
        assert "observed: Loss plateaued at epoch 3." in svg


class TestLinks:
    def test_each_card_links_to_its_experiment_when_asked(self) -> None:
        svg = graph_chart(NODES, baseline_value=0.80, metric_name="f1", link_base="/experiments")
        for node in NODES:
            assert f"href='/experiments/{node.experiment_id}'" in svg

    def test_the_static_dashboard_renders_without_links(self) -> None:
        svg = graph_chart(NODES, baseline_value=0.80, metric_name="f1")
        assert "<a " not in svg
