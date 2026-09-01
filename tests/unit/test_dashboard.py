"""Dashboard: SVG renderers, HTML assembly, gates, and self-containment."""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.reporting.svg_charts import (
    Bar,
    Point,
    ProgressPoint,
    SpreadRow,
    bar_chart,
    funnel_chart,
    progress_chart,
    scatter_chart,
    spread_chart,
)


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


class TestSvgCharts:
    def test_bar_chart_has_baseline_plus_experiment_bars(self) -> None:
        svg = bar_chart(
            [
                Bar(label="exp-001", value=0.85, status="validated", delta_pct=6.25),
                Bar(label="exp-002", value=0.7, status="rejected"),
                Bar(label="exp-003", value=0.81, status="promising", note="screening"),
            ],
            baseline_value=0.8,
            metric_name="f1",
        )
        root = _parse(svg)
        bars = [e for e in root.iter() if e.get("data-bar")]
        assert [b.get("data-bar") for b in bars] == ["baseline", "exp-001", "exp-002", "exp-003"]
        assert "screening" in svg  # screening values are labeled
        assert "+6.2%" in svg or "+6.3%" in svg

    def test_scatter_chart_constraint_line_position(self) -> None:
        svg = scatter_chart(
            [Point(label="exp-001", x=150.0, y=0.85, status="promising", pareto=True)],
            baseline=Point(label="baseline", x=100.0, y=0.8, status="baseline"),
            x_label="p95_latency_ms",
            y_label="f1",
            x_threshold=200.0,
            threshold_note="p95_latency_ms <= 200",
        )
        root = _parse(svg)
        line = next(e for e in root.iter() if e.get("data-role") == "constraint-line")
        region = next(e for e in root.iter() if e.get("data-role") == "violation-region")
        # The shaded violating region starts exactly at the constraint line.
        assert line.get("x1") == region.get("x")
        assert any(e.get("data-role") == "pareto-ring" for e in root.iter())
        assert "p95_latency_ms &lt;= 200" in svg or "p95_latency_ms <= 200" in svg

    def test_funnel_counts(self) -> None:
        svg = funnel_chart(
            [("imported", 3), ("screening", 3), ("full benchmark", 2), ("validated", 1)],
            ["", "exp-003 failed_execution", "exp-002 rejected", ""],
        )
        root = _parse(svg)
        counts = {
            e.get("data-funnel"): e.get("data-count") for e in root.iter() if e.get("data-funnel")
        }
        assert counts == {
            "imported": "3",
            "screening": "3",
            "full benchmark": "2",
            "validated": "1",
        }
        assert "exp-002 rejected" in svg

    def test_spread_chart_attempts_and_mean(self) -> None:
        svg = spread_chart(
            [
                SpreadRow(
                    label="exp-001",
                    values=[0.85, 0.84],
                    mean=0.845,
                    stdev=0.007,
                    outcome="validated",
                    extra_values=[0.85],
                )
            ],
            baseline_value=0.8,
            metric_name="f1",
        )
        root = _parse(svg)
        attempts = [e for e in root.iter() if e.get("data-attempt-value")]
        assert len(attempts) == 2
        assert any(e.get("data-role") == "mean-tick" for e in root.iter())
        assert any(e.get("data-role") == "full-run-value" for e in root.iter())


class TestProgressChart:
    def test_kept_discarded_and_running_best(self) -> None:
        svg = progress_chart(
            [
                ProgressPoint(index=1, value=0.85, kept=True, label="Variant normalize"),
                ProgressPoint(index=2, value=0.82, kept=False, label="Variant ngram"),
                ProgressPoint(index=3, value=0.88, kept=True, label="Variant stemming"),
            ],
            baseline_value=0.8,
            metric_name="f1",
            lower_is_better=False,
        )
        root = _parse(svg)
        kept = [e for e in root.iter() if e.get("data-progress") == "kept"]
        discarded = [e for e in root.iter() if e.get("data-progress") == "discarded"]
        assert len(kept) == 2 and len(discarded) == 1
        assert any(e.get("data-progress") == "baseline" for e in root.iter())
        assert any(e.get("data-role") == "running-best" for e in root.iter())
        assert "Variant normalize" in svg and "Variant stemming" in svg
        assert "Variant ngram" not in svg  # discarded points are not annotated
        assert "2 kept improvement(s)" in svg
        assert "higher is better" in svg

    def test_progress_points_direction_and_survival(self) -> None:
        """Running best only advances through surviving improvements."""
        from datetime import UTC, datetime, timedelta

        from researchforge.domain.contract import MetricDirection
        from researchforge.domain.experiment import (
            BenchmarkStage,
            ExecutionRecordStatus,
            Experiment,
            ExperimentExecution,
            ExperimentStatus,
        )
        from researchforge.execution.metrics import MetricResult, MetricValue
        from researchforge.reporting.dashboard import progress_points

        now = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)

        def experiment(eid: str, title: str, status: ExperimentStatus) -> Experiment:
            return Experiment(
                experiment_id=eid,
                plan_id="plan-001",
                hypothesis_id="hyp-001",
                title=title,
                change_summary="s",
                patch_text="p",
                patch_sha256="0" * 64,
                status=status,
                created_at=now,
                updated_at=now,
            )

        start = now

        from researchforge.domain.baseline import EnvironmentFingerprint
        from researchforge.domain.experiment import ExecutionArtifacts

        def execution(eid: str, value: float, minutes: int) -> ExperimentExecution:
            return ExperimentExecution(
                execution_id=f"x-{eid}",
                experiment_id=eid,
                run_id="run-001",
                hypothesis_id="hyp-001",
                baseline_commit="c" * 40,
                execution_mode="venv",
                benchmark_stage=BenchmarkStage.FULL,
                attempt=1,
                change_summary="s",
                started_at=start + timedelta(minutes=minutes),
                status=ExecutionRecordStatus.SUCCEEDED,
                metrics=MetricResult(
                    schema_version=1,
                    primary_metric=MetricValue(name="latency", value=value),
                ),
                artifacts=ExecutionArtifacts(diff_path="d", stdout_path="o", stderr_path="e"),
                fingerprint=EnvironmentFingerprint(
                    platform="test",
                    execution_mode="venv",
                    contract_id="contract-001",
                    contract_version=1,
                    commit_sha="c" * 40,
                ),
            )

        experiments = [
            experiment("exp-001", "worse", ExperimentStatus.PROMISING),
            experiment("exp-002", "better but rejected", ExperimentStatus.REJECTED),
            experiment("exp-003", "better and kept", ExperimentStatus.VALIDATED),
        ]
        executions = [
            execution("exp-001", 120.0, 0),  # worse (higher latency)
            execution("exp-002", 80.0, 1),  # improves, but rejected -> discarded
            execution("exp-003", 90.0, 2),  # improves the (unchanged) best -> kept
        ]

        points = progress_points(
            experiments, executions, baseline_value=100.0, direction=MetricDirection.MINIMIZE
        )

        assert [(p.experiment_id, p.kept) for p in points] == [
            ("exp-001", False),
            ("exp-002", False),
            ("exp-003", True),
        ]


class TestDashboardCommand:
    def test_dashboard_on_validated_project(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(app, ["dashboard", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["run_id"] == "run-001"
        html = Path(payload["path"]).read_text(encoding="utf-8")

        assert "0.8" in html  # baseline value
        assert "0.85" in html  # winner value
        assert "exp-002" in html  # the rejected loser is shown
        assert "p95_latency_ms" in html  # constraint scatter present
        assert "validated" in html
        assert "Funnel" in html and "Validation spread" in html
        assert "Progress —" in html and "kept improvement" in html

        # Self-contained: no external references, no scripts.
        assert "<script" not in html
        assert not re.search(r"https?://(?!www\.w3\.org)", html)

    def test_economics_reports_time_used_and_work_avoided(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        """The section accounts for compute without claiming a human alternative."""
        result = cli_runner.invoke(app, ["dashboard", "--json"])

        assert result.exit_code == 0, result.output
        html = Path(json.loads(result.output)["path"]).read_text(encoding="utf-8")

        assert "Time economics" in html
        assert "Compute used" in html
        assert "By outcome" in html
        assert "Work avoided" in html
        assert "The record" in html
        # exp-002 bought its metric with latency and was stopped.
        assert "Caught before it shipped" in html

    def test_outcome_percentages_sum_to_a_whole(self) -> None:
        """The baseline has no outcome, so these are of experiment compute.

        Sharing the "Compute used" denominator would leave the rows adding up to
        less than 100% in a table that looks like it should, which reads as an
        arithmetic bug even when the seconds are right.
        """
        from researchforge.reporting.dashboard import _economics_section
        from researchforge.reporting.economics import (
            Avoided,
            Caught,
            Economics,
            Record,
            StageTime,
        )

        html = _economics_section(
            Economics(
                stages=StageTime(baseline=300.0, full=900.0),
                by_outcome={"kept": 300.0, "rejected": 600.0},
                avoided=Avoided(),
                record=Record(experiments=2, kept=1, rejected=1),
                caught=Caught(),
            ),
            "f1",
        )

        percentages = [int(m) for m in re.findall(r">(\d+)%<", html)]
        # Four "Compute used" rows are of the total; the rest are of experiments.
        assert sum(percentages[-2:]) == 100
        assert "of experiments" in html

    def test_economics_makes_no_counterfactual_claim(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        """No "would have taken a human N hours" — that number cannot be checked."""
        result = cli_runner.invoke(app, ["dashboard", "--json"])

        html = Path(json.loads(result.output)["path"]).read_text(encoding="utf-8").lower()

        for claim in ("would have", "manually", "engineer-hour", "saved you", "by hand"):
            assert claim not in html, f"unfalsifiable claim in dashboard: {claim}"

    def test_gate_without_contract(self, cli_runner: CliRunner, initialized_project: Path) -> None:
        result = cli_runner.invoke(app, ["dashboard"])
        assert result.exit_code == 1
        assert "approved contract" in result.output

    def test_baseline_but_no_runs_renders_empty_state(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(app, ["dashboard", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["run_id"] is None
        html = Path(payload["path"]).read_text(encoding="utf-8")
        assert "No experiment runs recorded yet" in html
        assert "0.8" in html  # baseline card still rendered

    def test_unknown_run_rejected(
        self, cli_runner: CliRunner, funnel_project: Path, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(app, ["dashboard", "--run", "run-999"])
        assert result.exit_code == 1
        assert "Unknown run" in result.output

    def test_rebuild_records_single_deliverable_and_open_flag(
        self,
        cli_runner: CliRunner,
        validated_project: Path,
        isolated_project_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from contextlib import closing

        import researchforge.reporting.dashboard_cli as dashboard_cli
        from researchforge.domain.deliverable import DeliverableKind
        from researchforge.storage.db import open_project_db
        from researchforge.storage.deliverable_repository import list_deliverables

        opened: list[str] = []
        monkeypatch.setattr(dashboard_cli.webbrowser, "open", lambda url: opened.append(url))

        assert cli_runner.invoke(app, ["dashboard"]).exit_code == 0
        assert cli_runner.invoke(app, ["dashboard", "--open"]).exit_code == 0

        assert len(opened) == 1 and opened[0].startswith("file://")
        with closing(open_project_db()) as conn:
            dashboards = list_deliverables(conn, kind=DeliverableKind.DASHBOARD)
        assert len(dashboards) == 1


class TestDashboardV2:
    def test_graph_chart_nodes_and_edges(self) -> None:
        from researchforge.reporting.svg_charts import GraphNode, graph_chart

        svg = graph_chart(
            [
                GraphNode("exp-001", "Root", "promising", 0.85, 6.2),
                GraphNode("exp-002", "Loser", "rejected", 0.82, 2.9),
                GraphNode("exp-003", "Child", "validated", 0.87, 8.7, ["exp-001"]),
            ],
            baseline_value=0.8,
            metric_name="f1",
        )
        root = _parse(svg)
        nodes = {e.get("data-graph-node") for e in root.iter() if e.get("data-graph-node")}
        assert nodes == {"baseline", "exp-001", "exp-002", "exp-003"}
        edges = {e.get("data-graph-edge") for e in root.iter() if e.get("data-graph-edge")}
        assert edges == {
            "baseline->exp-001",
            "baseline->exp-002",
            "exp-001->exp-003",
        }
        assert "<a " not in svg  # static rendering: no links

        linked = graph_chart(
            [GraphNode("exp-001", "Root", "promising", 0.85)],
            baseline_value=0.8,
            metric_name="f1",
            link_base="/experiments",
        )
        assert "href='/experiments/exp-001'" in linked

    def test_stats_and_tree_in_dashboard(
        self, cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path
    ) -> None:
        result = cli_runner.invoke(app, ["dashboard", "--json"])
        assert result.exit_code == 0, result.output
        html = Path(json.loads(result.output)["path"]).read_text(encoding="utf-8")

        assert "best score" in html
        assert "0.85" in html and "+6.2%" in html  # winner + delta vs baseline 0.80
        assert "1 kept · 1 discarded · 0 err" in html
        assert "frontier" in html
        assert "Experiment graph" in html
        assert "data-graph-node='baseline'" in html
        assert "data-graph-edge='baseline-&gt;exp-001'" in html or "baseline->exp-001" in html
        assert "<script" not in html  # static file stays script-free
