"""Export dashboard charts from a project as standalone SVGs for the README.

The dashboard's SVGs reference CSS custom properties defined by its stylesheet,
so they only render inside that page. This substitutes the light-theme palette
and pins a size, producing files that render anywhere Markdown does.

    python scripts/export_readme_charts.py <project-dir> <output-dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from researchforge.config.settings import load_settings
from researchforge.execution.ranking import build_ranking_report
from researchforge.reporting.dashboard import graph_nodes, progress_points
from researchforge.reporting.svg_charts import graph_chart, progress_chart
from researchforge.storage.baseline_repository import get_latest_successful_baseline
from researchforge.storage.contract_repository import get_active_contract
from researchforge.storage.db import open_project_db
from researchforge.storage.experiment_repository import (
    list_executions,
    list_experiments,
    list_runs,
)

PALETTE = {
    "--bg": "#ffffff",
    "--card": "#f6f8fa",
    "--fg": "#1f2328",
    "--fg-muted": "#59636e",
    "--grid": "#e1e4e8",
    "--brand": "#7C3AED",
    "--accent": "#F59E0B",
    "--chart-good": "#10B981",
    "--chart-info": "#7C3AED",
    "--chart-bad": "#EF4444",
    "--chart-muted": "#9CA3AF",
    "--chart-baseline": "#F59E0B",
}


def standalone(svg: str) -> str:
    """Inline the palette and give the root a background and explicit size."""
    for name, value in PALETTE.items():
        svg = svg.replace(f"var({name})", value)
    svg = re.sub(r"var\(--[a-z-]+\)", "#59636e", svg)
    box = re.search(r"viewBox='0 0 ([\d.]+) ([\d.]+)'", svg)
    assert box is not None, "chart has no viewBox"
    width, height = box.group(1), box.group(2)
    svg = svg.replace(
        "<svg ", f"<svg width='{width}' height='{height}' ", 1
    )
    return svg.replace(
        ">", f"><rect width='{width}' height='{height}' fill='#ffffff'/>", 1
    )


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    conn = open_project_db(project)

    contract = get_active_contract(conn)
    baseline = get_latest_successful_baseline(conn)
    assert contract is not None and baseline is not None and baseline.metrics is not None
    spec = contract.spec
    metric = spec.objective.primary_metric
    baseline_value = baseline.metrics.primary_metric.value

    experiments = list_experiments(conn)
    executions = list_executions(conn)

    ranking = None
    groups = list_runs(conn)
    if groups:
        latest = groups[-1]
        ranking = build_ranking_report(
            latest.run_id,
            baseline,
            list_experiments(conn, latest.plan_id),
            list_executions(conn, run_id=latest.run_id),
            spec,
            tradeoff_material_pct=load_settings().tradeoff_material_pct,
        )

    charts = {
        "yolov5-progress.svg": progress_chart(
            progress_points(experiments, executions, baseline_value, metric.direction),
            baseline_value,
            metric.name,
            lower_is_better=metric.direction.value == "minimize",
        ),
        "yolov5-graph.svg": graph_chart(
            graph_nodes(experiments, executions, ranking),
            baseline_value,
            metric.name,
            best_experiment_id="exp-008",
        ),
    }
    for name, svg in charts.items():
        (out / name).write_text(standalone(svg))
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
