"""Self-contained HTML dashboard: experiments vs the frozen baseline.

One static file from recorded data only — inline CSS and inline SVG, no
scripts, no network. The honesty rules of the reports apply: screening
numbers are labeled screening, one-off results carry the caveat, and
rejected/failed experiments are shown, not hidden.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from researchforge import __version__
from researchforge.config.settings import load_settings
from researchforge.domain.baseline import BaselineRun
from researchforge.domain.contract import ContractSpec, ExperimentContract, MetricDirection
from researchforge.domain.experiment import (
    BenchmarkStage,
    Experiment,
    ExperimentExecution,
    ExperimentRunGroup,
    ExperimentStatus,
)
from researchforge.execution.ranking import (
    ONE_OFF_CAVEAT,
    RankingReport,
    build_ranking_report,
    signed_improvement,
)
from researchforge.execution.validation import summarize_validation
from researchforge.reporting.economics import Economics, build_economics, humanize_seconds
from researchforge.reporting.svg_charts import (
    Bar,
    GraphNode,
    Point,
    ProgressPoint,
    SpreadRow,
    bar_chart,
    funnel_chart,
    graph_chart,
    progress_chart,
    scatter_chart,
    spread_chart,
    status_color,
)

DASHBOARD_CSS = """
:root {
  --bg: #ffffff; --card: #f6f8fa; --fg: #1f2328; --fg-muted: #59636e;
  --grid: #e1e4e8; --brand: #7C3AED; --accent: #F59E0B;
  --chart-good: #10B981; --chart-info: #7C3AED;
  --chart-bad: #EF4444; --chart-muted: #9CA3AF; --chart-baseline: #F59E0B;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --card: #161b22; --fg: #e6edf3; --fg-muted: #8d96a0;
    --grid: #30363d; --brand: #A78BFA; --accent: #FCD34D;
    --chart-good: #34D399; --chart-info: #A78BFA;
    --chart-bad: #F87171; --chart-muted: #6B7280; --chart-baseline: #FCD34D;
  }
}
* { box-sizing: border-box; }
html { border-top: 3px solid var(--brand); }
body { background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 880px;
  padding: 28px 20px 64px; font-family: ui-sans-serif, system-ui, sans-serif; }
h1 { font-size: 1.45rem; margin: 0 0 4px; font-weight: 700; }
h2 { font-size: 1rem; font-weight: 600; margin: 36px 0 10px; padding-left: 10px;
  border-left: 3px solid var(--accent); color: var(--fg); }
.sub { color: var(--fg-muted); margin: 0 0 20px; font-size: 0.9rem; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; }
.card { background: var(--card); border-radius: 8px; padding: 12px 16px;
  flex: 1 1 160px; border: 1px solid var(--grid); }
.card .k { color: var(--fg-muted); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.05em; }
.card .v { font-size: 1.2rem; font-weight: 600; margin-top: 3px; overflow-wrap: anywhere; }
.card .d { color: var(--fg-muted); font-size: 0.78rem; margin-top: 3px; }
.card.stat { border-top: 2px solid var(--brand); }
.card.stat .v { font-size: 1.8rem; letter-spacing: -0.02em; color: var(--brand); }
.card.econ { flex: 1 1 240px; }
.card h3 { font-size: 0.78rem; font-weight: 600; margin: 0 0 8px; color: var(--fg-muted);
  text-transform: uppercase; letter-spacing: 0.04em; }
.card table { font-size: 0.8rem; }
.card td { padding: 4px 0; border-bottom: none; }
.card td:nth-child(2) { text-align: right; font-variant-numeric: tabular-nums; }
.card td:nth-child(3) { text-align: right; color: var(--fg-muted); width: 3.2em; }
.card .sub { margin: 0; font-size: 0.8rem; }
.big { font-size: 1.7rem; font-weight: 600; margin: 0 0 4px;
  letter-spacing: -0.02em; color: var(--brand); }
h3 { font-size: 0.92rem; font-weight: 600; margin: 22px 0 6px; }
svg { width: 100%; height: auto; background: var(--card); border-radius: 8px;
  padding: 8px; border: 1px solid var(--grid); }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid);
  vertical-align: top; }
th { color: var(--fg-muted); font-weight: 600; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.04em; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 10px; color: #fff;
  font-size: 0.72rem; font-weight: 600; white-space: nowrap; letter-spacing: 0.02em; }
.caveat { background: var(--card); border-left: 3px solid var(--accent);
  padding: 10px 14px; border-radius: 0 8px 8px 0; color: var(--fg-muted);
  font-size: 0.84rem; margin: 10px 0; }
.empty { color: var(--fg-muted); font-style: italic; }
.rf-masthead { display: flex; align-items: center; gap: 18px;
  margin: 0 0 28px; padding-bottom: 24px; border-bottom: 1px solid var(--grid); }
.rf-fox { font-family: ui-monospace, monospace; font-size: 0.78rem; line-height: 1.5;
  color: var(--accent); white-space: pre; margin: 0; user-select: none; }
.rf-brand-name { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--brand); margin-bottom: 6px; }
footer { margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--grid);
  color: var(--fg-muted); font-size: 0.78rem; }

/* ── Animations ─────────────────────────────────────────────────────────── */
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:.3} }
.live { animation: pulse-dot 2s ease-in-out infinite; }

/* ── Card hover lift ─────────────────────────────────────────────────────── */
.card { transition: transform .15s ease, box-shadow .15s ease; }
.card:hover { transform: translateY(-2px);
  box-shadow: 0 6px 24px rgba(124,58,237,.12); }
.card.stat .v {
  background: linear-gradient(135deg, var(--brand), var(--accent));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; }

/* ── Masthead gradient background ───────────────────────────────────────── */
.rf-masthead {
  background: linear-gradient(135deg,
    color-mix(in srgb, var(--brand) 7%, var(--bg)) 0%, var(--bg) 70%);
  border-radius: 12px; padding: 20px 24px; margin-bottom: 28px; }

/* ── Hub project-card status borders ────────────────────────────────────── */
.project-card { border-left: 3px solid var(--brand); }
.project-card.s-baselined,
.project-card.s-experiments_running { border-left-color: var(--accent); }
.project-card.s-validated,
.project-card.s-shipped { border-left-color: var(--chart-good); }
.project-card.s-missing { border-left-color: var(--chart-bad); opacity: .7; }
.project-card .card-name { font-weight: 700; font-size: 1rem;
  color: var(--fg); text-decoration: none; }
.project-card .card-name:hover { color: var(--brand); }
.project-card .card-stats { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.stat-pill { background: var(--bg); border: 1px solid var(--grid); border-radius: 12px;
  padding: 2px 10px; font-size: 0.72rem; color: var(--fg-muted); }
.project-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }

/* ── Empty state ─────────────────────────────────────────────────────────── */
.empty-state { text-align: center; padding: 60px 20px; }
.empty-fox { font-family: ui-monospace, monospace; color: var(--accent);
  font-size: 1rem; line-height: 1.6; display: inline-block; white-space: pre; }
.empty-state p { color: var(--fg-muted); margin: 8px 0; }

/* ── Nav improvements ───────────────────────────────────────────────────── */
nav { align-items: center; }
nav a { padding: 4px 0; border-bottom: 2px solid transparent;
  transition: color .1s, border-color .1s; }
nav a.active,nav a[class=active] { color: var(--fg); border-bottom-color: var(--brand); }
nav a:hover { color: var(--fg); }
.nav-brand { font-weight: 800; color: var(--brand); letter-spacing: -.01em;
  font-size: .95rem; flex-shrink: 0; margin-right: 4px; }

/* ── Inline code ─────────────────────────────────────────────────────────── */
code { background: var(--card); padding: 1px 6px; border-radius: 4px;
  font-size: .88em; border: 1px solid var(--grid); }
"""


def _badge(status: str) -> str:
    return f"<span class='badge' style='background:{status_color(status)}'>{escape(status)}</span>"


def _logo_html_inline(size: int = 56) -> str:
    """Embed the logo as a base64 data URI for self-contained static HTML files.
    Falls back to ASCII fox art when no logo is installed.
    """
    import base64

    search_paths = [
        Path.home() / ".researchforge" / "logo.png",
        Path.home() / ".researchforge" / "logo.jpeg",
        Path.home() / ".researchforge" / "logo.jpg",
    ]
    for p in search_paths:
        if p.is_file():
            mime = "image/png" if p.suffix == ".png" else "image/jpeg"
            b64 = base64.b64encode(p.read_bytes()).decode()
            return (
                f"<img src='data:{mime};base64,{b64}' alt='ResearchForge' "
                f"style='width:{size}px;height:{size}px;object-fit:contain;"
                f"border-radius:8px;background:transparent'>"
            )
    return "<pre class='rf-fox'>/\\   /\\\n( o   o )\n \\_____/</pre>"


def _latest_full_value(executions: list[ExperimentExecution], experiment_id: str) -> float | None:
    for execution in reversed(executions):
        if (
            execution.experiment_id == experiment_id
            and execution.benchmark_stage is BenchmarkStage.FULL
            and execution.metrics is not None
        ):
            return execution.metrics.primary_metric.value
    return None


def _stage_reached(executions: list[ExperimentExecution], experiment_id: str) -> str:
    order = [BenchmarkStage.SCREENING, BenchmarkStage.FULL, BenchmarkStage.VALIDATION]
    reached = [e.benchmark_stage for e in executions if e.experiment_id == experiment_id]
    if not reached:
        return "never ran"
    return max(reached, key=order.index).value


def _bars(
    experiments: list[Experiment],
    executions: list[ExperimentExecution],
    ranking: RankingReport | None,
) -> list[Bar]:
    deltas: dict[str, float | None] = {}
    if ranking is not None:
        for row in [*ranking.candidates, *ranking.rejected]:
            deltas[row.experiment_id] = row.primary_delta_pct
    bars = []
    for experiment in experiments:
        value = _latest_full_value(executions, experiment.experiment_id)
        note = None
        if value is None:  # fall back to a screening value, clearly labeled
            for execution in reversed(executions):
                if (
                    execution.experiment_id == experiment.experiment_id
                    and execution.metrics is not None
                ):
                    value, note = execution.metrics.primary_metric.value, "screening"
                    break
        if value is None:
            continue
        bars.append(
            Bar(
                label=experiment.experiment_id,
                value=value,
                status=experiment.status.value,
                delta_pct=deltas.get(experiment.experiment_id),
                note=note,
            )
        )
    return bars


_SURVIVOR_STATUSES = (
    ExperimentStatus.PROMISING,
    ExperimentStatus.VALIDATING,
    ExperimentStatus.VALIDATED,
    ExperimentStatus.IMPLEMENTATION_READY,
)


def progress_points(
    experiments: list[Experiment],
    executions: list[ExperimentExecution],
    baseline_value: float,
    direction: MetricDirection,
) -> list[ProgressPoint]:
    """Chronological full-benchmark measurements with running-best bookkeeping.

    A point is *kept* when it beat the running best AND the experiment
    survived (a constraint violator with a better primary metric stays
    discarded — the running best never advances through it).
    """
    titles = {e.experiment_id: e.title for e in experiments}
    statuses = {e.experiment_id: e.status for e in experiments}
    first_full: dict[str, ExperimentExecution] = {}
    for execution in sorted(executions, key=lambda e: e.started_at):
        if (
            execution.benchmark_stage is BenchmarkStage.FULL
            and execution.metrics is not None
            and execution.experiment_id not in first_full
        ):
            first_full[execution.experiment_id] = execution

    points: list[ProgressPoint] = []
    best = baseline_value
    for index, execution in enumerate(
        sorted(first_full.values(), key=lambda e: e.started_at), start=1
    ):
        assert execution.metrics is not None
        value = execution.metrics.primary_metric.value
        survived = statuses.get(execution.experiment_id) in _SURVIVOR_STATUSES
        kept = survived and signed_improvement(best, value, direction) > 0
        if kept:
            best = value
        points.append(
            ProgressPoint(
                index=index,
                value=value,
                kept=kept,
                label=titles.get(execution.experiment_id, execution.experiment_id),
                experiment_id=execution.experiment_id,
            )
        )
    return points


def _funnel(
    experiments: list[Experiment], executions: list[ExperimentExecution]
) -> tuple[list[tuple[str, int]], list[str]]:
    def reached(stage: BenchmarkStage) -> set[str]:
        return {e.experiment_id for e in executions if e.benchmark_stage is stage}

    screening, full, validation = (
        reached(BenchmarkStage.SCREENING),
        reached(BenchmarkStage.FULL),
        reached(BenchmarkStage.VALIDATION),
    )
    validated = {
        e.experiment_id
        for e in experiments
        if e.status in (ExperimentStatus.VALIDATED, ExperimentStatus.IMPLEMENTATION_READY)
    }
    dropped = {
        e.experiment_id: e.status.value
        for e in experiments
        if e.status
        in (
            ExperimentStatus.REJECTED,
            ExperimentStatus.FAILED_SETUP,
            ExperimentStatus.FAILED_EXECUTION,
            ExperimentStatus.CANCELLED,
        )
    }

    def drop_note(survivors: set[str]) -> str:
        lost = sorted(set(dropped) & survivors)
        return ", ".join(f"{eid} {dropped[eid]}" for eid in lost)

    stages = [
        ("imported", len(experiments)),
        ("screening", len(screening)),
        ("full benchmark", len(full)),
        ("validation", len(validation)),
        ("validated", len(validated)),
    ]
    notes = ["", drop_note(screening - full), drop_note(full - validation), "", ""]
    return stages, notes


def build_dashboard(
    conn: sqlite3.Connection,
    run: ExperimentRunGroup | None,
    link_base: str | None = None,
) -> str:
    """Assemble the dashboard HTML from stored records."""
    from researchforge.storage.baseline_repository import get_latest_successful_baseline
    from researchforge.storage.contract_repository import get_active_contract
    from researchforge.storage.deliverable_repository import list_deliverables
    from researchforge.storage.experiment_repository import list_executions, list_experiments
    from researchforge.storage.project_repository import get_project

    project = get_project(conn)
    contract = get_active_contract(conn)
    baseline = get_latest_successful_baseline(conn)
    assert project is not None and contract is not None and baseline is not None
    assert baseline.metrics is not None
    spec = contract.spec
    primary = spec.objective.primary_metric.name

    experiments = list_experiments(conn, run.plan_id) if run is not None else []
    executions = list_executions(conn, run_id=run.run_id) if run is not None else []
    ranking = None
    if run is not None and experiments:
        ranking = build_ranking_report(
            run.run_id,
            baseline,
            experiments,
            executions,
            spec,
            tradeoff_material_pct=load_settings().tradeoff_material_pct,
        )

    sections = [_header_section(project.name, spec, contract, baseline, run)]

    all_experiments = list_experiments(conn)
    all_executions = list_executions(conn)
    stats = build_stats(
        baseline,
        all_experiments,
        all_executions,
        ranking,
        spec.objective.primary_metric.direction,
    )
    stat_cards = "".join(
        f"<div class='card stat'><div class='k'>{escape(label)}</div>"
        f"<div class='v'>{escape(value)}</div><div class='d'>{escape(detail)}</div></div>"
        for label, value, detail in stats
    )
    sections.append(f"<div class='cards'>{stat_cards}</div>")

    nodes = graph_nodes(all_experiments, all_executions, ranking)
    if nodes:
        chart = graph_chart(
            nodes,
            baseline.metrics.primary_metric.value,
            primary,
            link_base=link_base,
            best_experiment_id=ranking.candidates[0].experiment_id
            if ranking is not None and ranking.candidates
            else None,
        )
        sections.append(
            f"<h2>Experiment graph</h2>{chart}"
            "<p class='sub'>Each experiment builds on the baseline, on one parent, or "
            "— dotted edges — on several at once. Values are measured against the "
            "frozen baseline, and the green chain traces the ancestry of the best "
            "result. Hover a card for its full title, parents and observation.</p>"
        )

    all_points = progress_points(
        list_experiments(conn),
        list_executions(conn),
        baseline.metrics.primary_metric.value,
        spec.objective.primary_metric.direction,
    )
    if all_points:
        kept = sum(1 for p in all_points if p.kept)
        chart = progress_chart(
            all_points,
            baseline.metrics.primary_metric.value,
            primary,
            lower_is_better=spec.objective.primary_metric.direction is MetricDirection.MINIMIZE,
        )
        sections.append(
            f"<h2>Progress — {len(all_points)} experiment(s), {kept} kept improvement(s)</h2>"
            f"{chart}<p class='sub'>Every full-benchmark measurement across all runs, in "
            "order. Green: improved the running best and survived; gray: discarded (worse, "
            "rejected on a constraint, or failed later).</p>"
        )

    sections.append(_economics_section(build_economics(conn), primary))

    if run is None:
        sections.append(
            "<p class='empty'>No experiment runs recorded yet — run "
            "<code>researchforge experiment run &lt;plan-id&gt;</code>, then rebuild this "
            "dashboard.</p>"
        )
    else:
        sections.append(_bar_section(experiments, executions, ranking, baseline, primary))
        sections.append(_scatter_section(spec, baseline, ranking, primary))
        sections.append(_funnel_section(experiments, executions))
        sections.append(_spread_section(spec, baseline, experiments, executions, primary))
        sections.append(_table_section(experiments, executions, primary))
        if not any(e.benchmark_stage is BenchmarkStage.VALIDATION for e in executions):
            sections.append(f"<div class='caveat'>{escape(ONE_OFF_CAVEAT)}</div>")

    deliverables = list_deliverables(conn)
    if deliverables:
        items = "".join(
            f"<li>{escape(d.kind.value)}: <code>{escape(d.location)}</code></li>"
            for d in deliverables
        )
        sections.append(f"<h2>Deliverables</h2><ul>{items}</ul>")

    sections.append(
        "<footer>Generated "
        f"{datetime.now(UTC).isoformat(timespec='seconds')} by ResearchForge {__version__} "
        f"from recorded data in <code>.researchforge/researchforge.db</code>. Results were "
        f"measured in {escape(baseline.execution_mode.value)} mode on one machine and may not "
        "generalize beyond the tested conditions.</footer>"
    )

    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>ResearchForge dashboard — {escape(project.name)}</title>"
        f"<style>{DASHBOARD_CSS}</style></head><body>{body}</body></html>"
    )


def _header_section(
    name: str,
    spec: ContractSpec,
    contract: ExperimentContract,
    baseline: BaselineRun,
    run: ExperimentRunGroup | None,
) -> str:
    assert baseline.metrics is not None
    objective = contract.spec.objective
    cards = [
        ("objective", escape(objective.description), ""),
        (
            f"baseline {escape(objective.primary_metric.name)}",
            f"{baseline.metrics.primary_metric.value:g}",
            f"commit {escape(contract.baseline_commit[:12])}",
        ),
    ]
    for metric_name, value in baseline.metrics.secondary_metrics.items():
        cards.append((f"baseline {escape(metric_name)}", f"{value:g}", ""))
    cards.append(
        (
            "contract",
            f"v{contract.contract_version}",
            f"{escape(baseline.execution_mode.value)} mode",
        )
    )
    if run is not None:
        cards.append(("run", escape(run.run_id), escape(run.status.value)))
    rendered = "".join(
        f"<div class='card'><div class='k'>{key}</div><div class='v'>{value}</div>"
        f"<div class='d'>{detail}</div></div>"
        for key, value, detail in cards
    )
    return (
        "<div class='rf-masthead'>"
        f"{_logo_html_inline(80)}"
        "<div>"
        "<div class='rf-brand-name'>ResearchForge</div>"
        f"<h1>dashboard &mdash; {escape(name)}</h1>"
        "<p class='sub'>Experiments vs the frozen baseline, from recorded data only.</p>"
        "</div>"
        "</div>"
        f"<div class='cards'>{rendered}</div>"
    )


def _bar_section(
    experiments: list[Experiment],
    executions: list[ExperimentExecution],
    ranking: RankingReport | None,
    baseline: BaselineRun,
    primary: str,
) -> str:
    assert baseline.metrics is not None
    bars = _bars(experiments, executions, ranking)
    if not bars:
        return "<h2>Primary metric</h2><p class='empty'>No measured experiments yet.</p>"
    chart = bar_chart(bars, baseline.metrics.primary_metric.value, primary)
    return f"<h2>Primary metric — every experiment vs baseline</h2>{chart}"


def _scatter_section(
    spec: ContractSpec, baseline: BaselineRun, ranking: RankingReport | None, primary: str
) -> str:
    assert baseline.metrics is not None
    if ranking is None:
        return ""
    rows = [*ranking.candidates, *ranking.rejected]
    charts = []
    secondaries = list(baseline.metrics.secondary_metrics)
    thresholds = {c.name: c for c in spec.objective.hard_constraints}
    for secondary in secondaries:
        points = [
            Point(
                label=row.experiment_id,
                x=row.secondary_values[secondary],
                y=row.primary_value,
                status=row.status.value,
                pareto=row.experiment_id in ranking.pareto_ids,
            )
            for row in rows
            if row.primary_value is not None and secondary in row.secondary_values
        ]
        if not points:
            continue
        constraint = thresholds.get(secondary)
        chart = scatter_chart(
            points,
            baseline=Point(
                label="baseline",
                x=baseline.metrics.secondary_metrics[secondary],
                y=baseline.metrics.primary_metric.value,
                status="baseline",
            ),
            x_label=secondary,
            y_label=primary,
            x_threshold=constraint.value if constraint is not None else None,
            threshold_note=(
                f"{constraint.name} {constraint.operator.value} {constraint.value:g}"
                if constraint is not None
                else None
            ),
        )
        charts.append(f"<h2>Trade-off — {escape(primary)} vs {escape(secondary)}</h2>{chart}")
    return "".join(charts)


def _funnel_section(experiments: list[Experiment], executions: list[ExperimentExecution]) -> str:
    stages, notes = _funnel(experiments, executions)
    return f"<h2>Funnel</h2>{funnel_chart(stages, notes)}"


def _spread_section(
    spec: ContractSpec,
    baseline: BaselineRun,
    experiments: list[Experiment],
    executions: list[ExperimentExecution],
    primary: str,
) -> str:
    assert baseline.metrics is not None
    rows = []
    for experiment in experiments:
        attempts = [
            e
            for e in executions
            if e.experiment_id == experiment.experiment_id
            and e.benchmark_stage is BenchmarkStage.VALIDATION
        ]
        if not attempts:
            continue
        summary = summarize_validation(
            experiment, attempts, baseline, spec.objective.primary_metric.direction
        )
        full_value = _latest_full_value(executions, experiment.experiment_id)
        rows.append(
            SpreadRow(
                label=experiment.experiment_id,
                values=summary.values,
                mean=summary.mean,
                stdev=summary.stdev,
                outcome=summary.outcome.value,
                extra_values=[full_value] if full_value is not None else [],
            )
        )
    if not rows:
        return ""
    chart = spread_chart(rows, baseline.metrics.primary_metric.value, primary)
    return (
        "<h2>Validation spread — repeated runs, not one-offs</h2>"
        f"{chart}<p class='sub'>Filled dots: validation attempts. Hollow dot: the original "
        "full-benchmark value. Tick: mean.</p>"
    )


def _economics_section(economics: Economics, primary: str) -> str:
    """Where the compute went, and which runs never had to happen.

    Deliberately makes no claim about what a person would have spent instead:
    the avoided-work figure is a count of skipped runs times this project's own
    measured cost per run, and both halves are shown so the reader can check the
    multiplication rather than take it on faith.
    """
    stages = economics.stages
    if stages.total <= 0:
        return ""

    parts = [
        ("baseline", stages.baseline),
        ("screening", stages.screening),
        ("full benchmarks", stages.full),
        ("validation", stages.validation),
    ]
    used = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(humanize_seconds(value))}</td>"
        f"<td>{value / stages.total * 100:.0f}%</td></tr>"
        for label, value in parts
        if value > 0
    )
    # Percentages here are of experiment compute, not of the total: the baseline
    # has no outcome, so sharing the "Compute used" denominator would leave these
    # rows summing to less than 100% in a table that looks like it should.
    experiment_total = sum(economics.by_outcome.values())
    by_outcome = "".join(
        f"<tr><td>{escape(outcome)}</td><td>{escape(humanize_seconds(value))}</td>"
        f"<td>{value / experiment_total * 100:.0f}%</td></tr>"
        for outcome, value in economics.by_outcome.items()
        if experiment_total > 0
    )

    avoided = economics.avoided
    if avoided.runs == 0:
        avoided_html = (
            "<p class='sub'>Nothing was screened out or stopped early, so every "
            "experiment cost a full benchmark.</p>"
        )
    elif avoided.seconds is None:
        avoided_html = (
            f"<p class='sub'>{avoided.runs} run(s) were skipped, but no full benchmark "
            "has completed yet — there is no measured cost per run to price them at.</p>"
        )
    else:
        avoided_html = (
            f"<p class='big'>{escape(humanize_seconds(avoided.seconds))}</p>"
            f"<p class='sub'>{avoided.runs} full benchmark(s) never ran — "
            f"{avoided.screened_out} screened out, {avoided.cancelled} stopped by the "
            f"stall rule — priced at this project's own average full benchmark of "
            f"{escape(humanize_seconds(avoided.mean_full_seconds))}.</p>"
        )

    spend = economics.tokens
    estimate_note = ""
    if spend.has_estimates:
        estimate_note = (
            f"<p class='sub'>Plus about {spend.estimated_tokens:,} token(s) across "
            f"{spend.estimated_calls} planning exchange(s) driven from an IDE. Those "
            "were spent in your editor's session, so they are sized from the context "
            "handed over rather than metered, and are not priced.</p>"
        )
    if spend.calls:
        by_purpose = "".join(
            f"<tr><td>{escape(label)}</td><td>{tokens:,}</td></tr>"
            for label, tokens in spend.by_purpose.items()
        )
        if spend.total_tokens and spend.usd > 0:
            headline = f"<p class='big'>${spend.usd:,.2f}</p>"
        else:
            headline = f"<p class='big'>{spend.total_tokens:,}</p>"
        caveat = ""
        if not spend.fully_priced:
            caveat = (
                "<p class='sub'>No rate configured for "
                f"{escape(', '.join(spend.unpriced_models))}, so those tokens are "
                "counted but not priced — the dollar figure is a floor.</p>"
            )
        tokens_html = (
            "<div class='card econ'><h3>Model calls</h3>"
            f"{headline}"
            f"<p class='sub'>{spend.calls:,} call(s) · {spend.input_tokens:,} in / "
            f"{spend.output_tokens:,} out</p>"
            f"<table><tbody>{by_purpose}</tbody></table>{caveat}{estimate_note}</div>"
        )
    elif spend.has_estimates:
        tokens_html = (
            "<div class='card econ'><h3>Model calls</h3>"
            f"<p class='big'>~{spend.estimated_tokens:,}</p>"
            f"<p class='sub'>{spend.estimated_calls} planning exchange(s) driven from "
            "an IDE. The tokens were spent in your editor's session, so this is sized "
            "from the context handed over and what came back — an estimate, not a "
            "metered count, and not priced.</p></div>"
        )
    else:
        # An empty card that explains itself, rather than no card at all: a
        # reader looking for token spend should learn why there is none instead
        # of wondering whether the feature is broken.
        tokens_html = (
            "<div class='card econ'><h3>Model calls</h3><p class='big'>—</p>"
            "<p class='sub'>No model calls recorded for this project. Usage is "
            "captured as calls happen, so runs made before token accounting "
            "existed have none to report, and a loop driven from an IDE spends "
            "its tokens there rather than here.</p></div>"
        )

    if economics.compute_usd is not None:
        compute_note = (
            f"<p class='sub'>Compute priced at the configured hourly rate: "
            f"${economics.compute_usd:,.2f} for {escape(humanize_seconds(stages.total))}."
            "</p>"
        )
    else:
        compute_note = (
            "<p class='sub'>Compute is reported in hours, not dollars: what an hour "
            "of this machine costs is specific to your setup. Set "
            "<code>local_compute_usd_per_hour</code> in "
            "<code>.researchforge/config.json</code> to convert it.</p>"
        )

    record = economics.record
    caught = economics.caught
    kept_note = (
        f"{record.experiments} experiment(s) recorded, {record.with_lineage} building on "
        f"an earlier result. {record.kept} kept, {record.rejected} rejected, "
        f"{record.failed} failed, {record.cancelled} cancelled — the ones that did not "
        "work are kept too, so the same idea is not retried by the next person."
    )
    caught_rows = []
    if caught.constraint_violations:
        caught_rows.append(
            f"<li><strong>{len(caught.constraint_violations)}</strong> change(s) broke a "
            "hard limit and were stopped by it, whatever they did to "
            f"{escape(primary)}: {escape(', '.join(caught.constraint_violations))}.</li>"
        )
    if caught.no_ops:
        caught_rows.append(
            f"<li><strong>{len(caught.no_ops)}</strong> change(s) measured exactly what "
            "they inherited, contributing nothing of their own: "
            f"{escape(', '.join(caught.no_ops))}.</li>"
        )
    caught_html = (
        f"<h3>Caught before it shipped</h3><ul>{''.join(caught_rows)}</ul>" if caught_rows else ""
    )

    return (
        "<h2>Time economics</h2>"
        "<div class='cards'>"
        f"<div class='card econ'><h3>Compute used — {escape(humanize_seconds(stages.total))}"
        f"</h3><table><tbody>{used}</tbody></table></div>"
        f"<div class='card econ'><h3>By outcome — {escape(humanize_seconds(experiment_total))}"
        f" of experiments</h3><table><tbody>{by_outcome}</tbody></table></div>"
        f"<div class='card econ'><h3>Work avoided</h3>{avoided_html}</div>"
        f"{tokens_html}"
        "</div>"
        f"{compute_note}"
        f"<h3>The record</h3><p class='sub'>{escape(kept_note)}</p>"
        f"{caught_html}"
    )


def _table_section(
    experiments: list[Experiment], executions: list[ExperimentExecution], primary: str
) -> str:
    rows = []
    for experiment in experiments:
        value = _latest_full_value(executions, experiment.experiment_id)
        reason = experiment.decision.reason if experiment.decision else ""
        rows.append(
            "<tr>"
            f"<td>{escape(experiment.experiment_id)}</td>"
            f"<td>{escape(experiment.title)}</td>"
            f"<td>{_badge(experiment.status.value)}</td>"
            f"<td>{escape(_stage_reached(executions, experiment.experiment_id))}</td>"
            f"<td>{value if value is not None else '—'}</td>"
            f"<td>{escape(reason)}</td>"
            "</tr>"
        )
    return (
        "<h2>All experiments</h2><table><thead><tr>"
        f"<th>id</th><th>title</th><th>status</th><th>stage reached</th><th>{escape(primary)} "
        "(full)</th><th>decision</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_stats(
    baseline: BaselineRun,
    experiments: list[Experiment],
    executions: list[ExperimentExecution],
    ranking: RankingReport | None,
    direction: MetricDirection,
) -> list[tuple[str, str, str]]:
    """Headline stat cards: (label, value, detail) — recorded data only."""
    assert baseline.metrics is not None
    base_value = baseline.metrics.primary_metric.value
    survivors = [e for e in experiments if e.status in _SURVIVOR_STATUSES]
    best_value: float | None = None
    for experiment in survivors:
        value = _latest_full_value(executions, experiment.experiment_id)
        if value is None:
            continue
        if best_value is None or signed_improvement(best_value, value, direction) > 0:
            best_value = value
    if best_value is not None and base_value:
        delta = (best_value - base_value) / abs(base_value)
        best_text = f"{best_value:g}"
        best_detail = f"{delta:+.1%} vs baseline {base_value:g}"
    else:
        best_text = f"{base_value:g}"
        best_detail = "baseline (no surviving experiment yet)"

    errs = sum(1 for e in experiments if e.status.value.startswith("failed_"))
    discarded = sum(
        1
        for e in experiments
        if e.status in (ExperimentStatus.REJECTED, ExperimentStatus.CANCELLED)
    )
    frontier = len(ranking.pareto_ids) if ranking is not None else 0
    return [
        ("best score", best_text, best_detail),
        (
            "experiments",
            str(len(experiments)),
            f"{len(survivors)} kept · {discarded} discarded · {errs} err",
        ),
        ("frontier", str(frontier), "Pareto candidates"),
    ]


def graph_nodes(
    experiments: list[Experiment],
    executions: list[ExperimentExecution],
    ranking: RankingReport | None,
) -> list[GraphNode]:
    deltas: dict[str, float | None] = {}
    if ranking is not None:
        for row in [*ranking.candidates, *ranking.rejected]:
            deltas[row.experiment_id] = row.primary_delta_pct
    return [
        GraphNode(
            experiment_id=e.experiment_id,
            title=e.title,
            status=e.status.value,
            value=_latest_full_value(executions, e.experiment_id),
            delta_pct=deltas.get(e.experiment_id),
            parent_ids=list(e.parent_experiment_ids),
            observation=e.observation,
        )
        for e in experiments
    ]
