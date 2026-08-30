"""Deterministic inline-SVG chart renderers for the local dashboard.

Pure functions, no dependencies: each returns an ``<svg>…</svg>`` string
meant to be embedded in the dashboard HTML. Colors reference CSS variables
(``var(--chart-…)``) defined by the page so the charts follow light/dark
mode. Kept intentionally simple — these visualize a handful of experiments,
not arbitrary datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from researchforge.reporting.graph_layout import (
    ROOT_KEY,
    Box,
    Metrics,
    layout_graph,
)
from researchforge.reporting.graph_layout import Layout as GraphLayout

FONT = "font-family='ui-sans-serif, system-ui, sans-serif'"

# Status -> CSS variable used as fill.
STATUS_COLOR = {
    "validated": "var(--chart-good)",
    "implementation_ready": "var(--chart-good)",
    "promising": "var(--chart-info)",
    "rejected": "var(--chart-bad)",
    "failed_setup": "var(--chart-muted)",
    "failed_execution": "var(--chart-muted)",
    "cancelled": "var(--chart-muted)",
}
DEFAULT_COLOR = "var(--chart-muted)"
BASELINE_COLOR = "var(--chart-baseline)"


def status_color(status: str) -> str:
    return STATUS_COLOR.get(status, DEFAULT_COLOR)


def _value_span(values: list[float], pad_ratio: float = 0.15) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if lo == hi:
        pad = abs(lo) * pad_ratio or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def _fmt(value: float) -> str:
    return f"{value:.4g}"


@dataclass
class Bar:
    label: str
    value: float
    status: str  # drives the fill color
    delta_pct: float | None = None
    note: str | None = None  # e.g. "screening"


def bar_chart(bars: list[Bar], baseline_value: float, metric_name: str) -> str:
    """Baseline + one bar per experiment, with a dashed baseline line."""
    width, height, pad_left, pad_bottom, pad_top = 760, 300, 60, 56, 24
    plot_w, plot_h = width - pad_left - 16, height - pad_top - pad_bottom
    all_bars = [Bar(label="baseline", value=baseline_value, status="baseline"), *bars]
    lo, hi = _value_span([b.value for b in all_bars] + [0.0 if baseline_value > 0 else 0.0])
    lo = min(lo, 0.0) if baseline_value >= 0 else lo

    def y_of(value: float) -> float:
        return pad_top + plot_h * (1 - (value - lo) / (hi - lo))

    slot = plot_w / max(len(all_bars), 1)
    bar_w = min(72.0, slot * 0.6)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' xmlns='http://www.w3.org/2000/svg'>",
        f"<text x='{pad_left}' y='14' {FONT} font-size='12' fill='var(--fg-muted)'>"
        f"{escape(metric_name)}</text>",
    ]
    # Gridlines + axis labels.
    for i in range(5):
        gval = lo + (hi - lo) * i / 4
        gy = y_of(gval)
        parts.append(
            f"<line x1='{pad_left}' y1='{gy:.1f}' x2='{width - 16}' y2='{gy:.1f}' "
            "stroke='var(--grid)' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{pad_left - 6}' y='{gy + 4:.1f}' {FONT} font-size='11' "
            f"text-anchor='end' fill='var(--fg-muted)'>{_fmt(gval)}</text>"
        )
    # Baseline reference line.
    by = y_of(baseline_value)
    parts.append(
        f"<line x1='{pad_left}' y1='{by:.1f}' x2='{width - 16}' y2='{by:.1f}' "
        f"stroke='{BASELINE_COLOR}' stroke-width='1.5' stroke-dasharray='6 4'/>"
    )
    for index, bar in enumerate(all_bars):
        x = pad_left + slot * index + (slot - bar_w) / 2
        top = y_of(max(bar.value, min(0.0, hi)))
        zero_y = y_of(max(lo, 0.0))
        bar_top, bar_h = (top, abs(zero_y - top)) if bar.value >= 0 else (zero_y, abs(top - zero_y))
        fill = BASELINE_COLOR if bar.status == "baseline" else status_color(bar.status)
        parts.append(
            f"<rect x='{x:.1f}' y='{bar_top:.1f}' width='{bar_w:.1f}' height='{max(bar_h, 1):.1f}' "
            f"rx='3' fill='{fill}' data-bar='{escape(bar.label)}'/>"
        )
        cx = x + bar_w / 2
        value_text = _fmt(bar.value)
        if bar.delta_pct is not None:
            value_text += f" ({bar.delta_pct:+.1f}%)"
        parts.append(
            f"<text x='{cx:.1f}' y='{y_of(bar.value) - 6:.1f}' {FONT} font-size='11' "
            f"text-anchor='middle' fill='var(--fg)'>{escape(value_text)}</text>"
        )
        parts.append(
            f"<text x='{cx:.1f}' y='{height - pad_bottom + 16}' {FONT} font-size='11' "
            f"text-anchor='middle' fill='var(--fg)'>{escape(bar.label)}</text>"
        )
        note = "baseline" if bar.status == "baseline" else (bar.note or bar.status)
        parts.append(
            f"<text x='{cx:.1f}' y='{height - pad_bottom + 31}' {FONT} font-size='10' "
            f"text-anchor='middle' fill='var(--fg-muted)'>{escape(note)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


@dataclass
class ProgressPoint:
    index: int  # chronological experiment number (0 = baseline)
    value: float
    kept: bool  # improved the running best (and survived)
    label: str  # annotation shown for kept points
    experiment_id: str = ""


def progress_chart(
    points: list[ProgressPoint], baseline_value: float, metric_name: str, lower_is_better: bool
) -> str:
    """Autoresearch-style progress: every experiment as a dot in chronological
    order, kept improvements annotated, and a step line tracking the running
    best. Inspired by karpathy/autoresearch's progress plot."""
    width, height, pad_left, pad_bottom, pad_top = 760, 340, 60, 44, 40
    plot_w, plot_h = width - pad_left - 24, height - pad_top - pad_bottom
    values = [p.value for p in points] + [baseline_value]
    lo, hi = _value_span(values)
    max_index = max((p.index for p in points), default=1)

    def sx(index: float) -> float:
        # +0.6 keeps the newest point (and its label) off the right edge.
        return pad_left + plot_w * index / (max(max_index, 1) + 0.6)

    def sy(value: float) -> float:
        return pad_top + plot_h * (1 - (value - lo) / (hi - lo))

    direction = "lower is better" if lower_is_better else "higher is better"
    kept_count = sum(1 for p in points if p.kept)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' xmlns='http://www.w3.org/2000/svg'>",
        f"<text x='{pad_left}' y='16' {FONT} font-size='12' fill='var(--fg-muted)'>"
        f"{escape(metric_name)} ({escape(direction)}) — {len(points)} experiment(s), "
        f"{kept_count} kept improvement(s)</text>",
        f"<text x='{pad_left + plot_w / 2:.1f}' y='{height - 6}' {FONT} font-size='11' "
        "text-anchor='middle' fill='var(--fg-muted)'>experiment # (chronological, all runs)"
        "</text>",
    ]
    for i in range(5):
        gval = lo + (hi - lo) * i / 4
        gy = sy(gval)
        parts.append(
            f"<line x1='{pad_left}' y1='{gy:.1f}' x2='{width - 24}' y2='{gy:.1f}' "
            "stroke='var(--grid)' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{pad_left - 6}' y='{gy + 4:.1f}' {FONT} font-size='10' text-anchor='end' "
            f"fill='var(--fg-muted)'>{_fmt(gval)}</text>"
        )
    # Running-best step line through baseline + kept points.
    steps = [(0, baseline_value)] + [(p.index, p.value) for p in points if p.kept]
    path = f"M {sx(steps[0][0]):.1f} {sy(steps[0][1]):.1f}"
    for (_, prev_value), (next_index, next_value) in zip(steps, steps[1:], strict=False):
        path += f" L {sx(next_index):.1f} {sy(prev_value):.1f}"
        path += f" L {sx(next_index):.1f} {sy(next_value):.1f}"
    path += f" L {sx(max_index):.1f} {sy(steps[-1][1]):.1f}"
    parts.append(
        f"<path d='{path}' fill='none' stroke='var(--chart-good)' stroke-width='2' "
        "data-role='running-best'/>"
    )
    # Baseline point + label.
    bx, by = sx(0), sy(baseline_value)
    parts.append(
        f"<circle cx='{bx:.1f}' cy='{by:.1f}' r='6' fill='{BASELINE_COLOR}' "
        "data-progress='baseline'/>"
    )
    parts.append(
        f"<text x='{bx + 8:.1f}' y='{by - 8:.1f}' {FONT} font-size='10' "
        f"fill='{BASELINE_COLOR}' transform='rotate(-30 {bx + 8:.1f} {by - 8:.1f})'>"
        "baseline</text>"
    )
    for point in points:
        px, py = sx(point.index), sy(point.value)
        if point.kept:
            parts.append(
                f"<circle cx='{px:.1f}' cy='{py:.1f}' r='6' fill='var(--chart-good)' "
                f"stroke='var(--bg)' stroke-width='1' data-progress='kept' "
                f"data-value='{_fmt(point.value)}'/>"
            )
            # Flip labels near the right edge so they never clip.
            flip = px > pad_left + plot_w * 0.75
            lx = px - 8 if flip else px + 8
            anchor = " text-anchor='end'" if flip else ""
            parts.append(
                f"<text x='{lx:.1f}' y='{py - 8:.1f}' {FONT} font-size='10'{anchor} "
                f"fill='var(--chart-good)' transform='rotate(-30 {lx:.1f} {py - 8:.1f})'>"
                f"{escape(point.label)}</text>"
            )
        else:
            parts.append(
                f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3.5' fill='var(--chart-muted)' "
                f"opacity='0.55' data-progress='discarded' data-value='{_fmt(point.value)}'/>"
            )
    parts.append("</svg>")
    return "".join(parts)


@dataclass
class Point:
    label: str
    x: float
    y: float
    status: str
    pareto: bool = False


def scatter_chart(
    points: list[Point],
    baseline: Point,
    x_label: str,
    y_label: str,
    x_threshold: float | None = None,
    threshold_note: str | None = None,
) -> str:
    """Primary metric (y) vs a secondary metric (x); optional constraint line."""
    width, height, pad_left, pad_bottom, pad_top = 760, 320, 60, 48, 24
    plot_w, plot_h = width - pad_left - 24, height - pad_top - pad_bottom
    everything = [baseline, *points]
    xs = [p.x for p in everything] + ([x_threshold] if x_threshold is not None else [])
    ys = [p.y for p in everything]
    x_lo, x_hi = _value_span(xs)
    y_lo, y_hi = _value_span(ys)

    def sx(value: float) -> float:
        return pad_left + plot_w * (value - x_lo) / (x_hi - x_lo)

    def sy(value: float) -> float:
        return pad_top + plot_h * (1 - (value - y_lo) / (y_hi - y_lo))

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect x='{pad_left}' y='{pad_top}' width='{plot_w}' height='{plot_h}' fill='none' "
        "stroke='var(--grid)'/>",
        f"<text x='{pad_left + plot_w / 2:.1f}' y='{height - 8}' {FONT} font-size='12' "
        f"text-anchor='middle' fill='var(--fg-muted)'>{escape(x_label)}</text>",
        f"<text x='14' y='{pad_top + plot_h / 2:.1f}' {FONT} font-size='12' text-anchor='middle' "
        f"fill='var(--fg-muted)' transform='rotate(-90 14 {pad_top + plot_h / 2:.1f})'>"
        f"{escape(y_label)}</text>",
    ]
    for i in range(5):
        gx = x_lo + (x_hi - x_lo) * i / 4
        gy = y_lo + (y_hi - y_lo) * i / 4
        parts.append(
            f"<text x='{sx(gx):.1f}' y='{height - pad_bottom + 16}' {FONT} font-size='10' "
            f"text-anchor='middle' fill='var(--fg-muted)'>{_fmt(gx)}</text>"
        )
        parts.append(
            f"<text x='{pad_left - 6}' y='{sy(gy) + 3:.1f}' {FONT} font-size='10' "
            f"text-anchor='end' fill='var(--fg-muted)'>{_fmt(gy)}</text>"
        )
    if x_threshold is not None:
        tx = sx(x_threshold)
        # Shade the violating side (assumes an upper bound, the common case).
        shade_w = max(pad_left + plot_w - tx, 0)
        parts.append(
            f"<rect x='{tx:.1f}' y='{pad_top}' width='{shade_w:.1f}' height='{plot_h}' "
            "fill='var(--chart-bad)' opacity='0.08' data-role='violation-region'/>"
        )
        parts.append(
            f"<line x1='{tx:.1f}' y1='{pad_top}' x2='{tx:.1f}' y2='{pad_top + plot_h}' "
            "stroke='var(--chart-bad)' stroke-width='1.5' stroke-dasharray='5 4' "
            "data-role='constraint-line'/>"
        )
        if threshold_note:
            # Flip the label to the left of the line when it would clip.
            near_right_edge = tx > pad_left + plot_w * 0.7
            note_x = tx - 4 if near_right_edge else tx + 4
            anchor = "end" if near_right_edge else "start"
            parts.append(
                f"<text x='{note_x:.1f}' y='{pad_top + 12}' {FONT} font-size='10' "
                f"text-anchor='{anchor}' fill='var(--chart-bad)'>{escape(threshold_note)}</text>"
            )
    bx, by = sx(baseline.x), sy(baseline.y)
    parts.append(
        f"<rect x='{bx - 5:.1f}' y='{by - 5:.1f}' width='10' height='10' "
        f"fill='{BASELINE_COLOR}' transform='rotate(45 {bx:.1f} {by:.1f})' data-point='baseline'/>"
    )
    parts.append(
        f"<text x='{bx + 9:.1f}' y='{by + 4:.1f}' {FONT} font-size='11' fill='var(--fg)'>"
        "baseline</text>"
    )
    for point in points:
        px, py = sx(point.x), sy(point.y)
        if point.pareto:
            parts.append(
                f"<circle cx='{px:.1f}' cy='{py:.1f}' r='9' fill='none' "
                "stroke='var(--chart-good)' stroke-width='1.5' data-role='pareto-ring'/>"
            )
        parts.append(
            f"<circle cx='{px:.1f}' cy='{py:.1f}' r='5.5' fill='{status_color(point.status)}' "
            f"data-point='{escape(point.label)}'/>"
        )
        parts.append(
            f"<text x='{px + 9:.1f}' y='{py + 4:.1f}' {FONT} font-size='11' fill='var(--fg)'>"
            f"{escape(point.label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def funnel_chart(stages: list[tuple[str, int]], drop_notes: list[str] | None = None) -> str:
    """Horizontal funnel: count reaching each stage, widths proportional."""
    width, row_h, pad_left = 760, 44, 170
    top = max(count for _, count in stages) if stages else 0
    height = row_h * len(stages) + 20
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' xmlns='http://www.w3.org/2000/svg'>"]
    notes = drop_notes or []
    for index, (label, count) in enumerate(stages):
        y = 10 + row_h * index
        bar_w = (width - pad_left - 140) * (count / top) if top else 0
        parts.append(
            f"<text x='{pad_left - 8}' y='{y + 22}' {FONT} font-size='12' text-anchor='end' "
            f"fill='var(--fg)'>{escape(label)}</text>"
        )
        parts.append(
            f"<rect x='{pad_left}' y='{y + 6}' width='{max(bar_w, 2):.1f}' height='24' rx='4' "
            f"fill='var(--chart-info)' opacity='{1 - index * 0.12:.2f}' "
            f"data-funnel='{escape(label)}' data-count='{count}'/>"
        )
        note = f" — {notes[index]}" if index < len(notes) and notes[index] else ""
        parts.append(
            f"<text x='{pad_left + max(bar_w, 2) + 8:.1f}' y='{y + 22}' {FONT} font-size='12' "
            f"fill='var(--fg)'>{count}{escape(note)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


@dataclass
class SpreadRow:
    label: str
    values: list[float]  # one per validation attempt
    mean: float | None = None
    outcome: str = ""
    stdev: float | None = None
    extra_values: list[float] = field(default_factory=list)  # e.g. the full-run value


def spread_chart(rows: list[SpreadRow], baseline_value: float, metric_name: str) -> str:
    """Per finalist: a dot per validation attempt, a mean tick, baseline line."""
    width, row_h, pad_left, pad_top = 760, 54, 170, 30
    height = pad_top + row_h * len(rows) + 30
    all_values = [v for row in rows for v in row.values + row.extra_values] + [baseline_value]
    lo, hi = _value_span(all_values)

    def x_of(value: float) -> float:
        return pad_left + (width - pad_left - 30) * (value - lo) / (hi - lo)

    bx = x_of(baseline_value)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' xmlns='http://www.w3.org/2000/svg'>",
        f"<text x='{pad_left}' y='16' {FONT} font-size='12' fill='var(--fg-muted)'>"
        f"{escape(metric_name)} per validation attempt</text>",
        f"<line x1='{bx:.1f}' y1='{pad_top}' x2='{bx:.1f}' y2='{height - 24}' "
        f"stroke='{BASELINE_COLOR}' stroke-width='1.5' stroke-dasharray='6 4'/>",
        f"<text x='{bx:.1f}' y='{height - 8}' {FONT} font-size='10' text-anchor='middle' "
        f"fill='var(--fg-muted)'>baseline {_fmt(baseline_value)}</text>",
    ]
    for index, row in enumerate(rows):
        cy = pad_top + row_h * index + row_h / 2
        outcome = f" [{row.outcome}]" if row.outcome else ""
        stdev = f" ±{_fmt(row.stdev)}" if row.stdev is not None else ""
        parts.append(
            f"<text x='{pad_left - 8}' y='{cy + 4:.1f}' {FONT} font-size='12' text-anchor='end' "
            f"fill='var(--fg)'>{escape(row.label + outcome + stdev)}</text>"
        )
        if row.mean is not None:
            mx = x_of(row.mean)
            parts.append(
                f"<line x1='{mx:.1f}' y1='{cy - 14:.1f}' x2='{mx:.1f}' y2='{cy + 14:.1f}' "
                "stroke='var(--fg)' stroke-width='2' data-role='mean-tick'/>"
            )
        for value in row.values:
            parts.append(
                f"<circle cx='{x_of(value):.1f}' cy='{cy:.1f}' r='5' fill='var(--chart-info)' "
                f"opacity='0.85' data-attempt-value='{_fmt(value)}'/>"
            )
        for value in row.extra_values:
            parts.append(
                f"<circle cx='{x_of(value):.1f}' cy='{cy:.1f}' r='4' fill='none' "
                f"stroke='var(--chart-info)' data-role='full-run-value'/>"
            )
    parts.append("</svg>")
    return "".join(parts)


@dataclass
class GraphNode:
    experiment_id: str
    title: str
    status: str
    value: float | None = None
    delta_pct: float | None = None
    parent_ids: list[str] = field(default_factory=list)
    """Empty = child of the baseline root. More than one = a merge."""

    round_num: int | None = None  # autorun round (None = single-round run)
    observation: str | None = None

    @property
    def parent_id(self) -> str | None:
        """The first parent — the single-lineage view, for callers that want it."""
        return self.parent_ids[0] if self.parent_ids else None

    @property
    def is_merge(self) -> bool:
        return len(self.parent_ids) > 1


STATUS_BADGE = {
    "implementation_ready": "SHIP",
    "validated": "VALIDATED",
    "promising": "KEPT",
    "rejected": "REJECTED",
    "failed_setup": "SETUP FAILED",
    "failed_execution": "RUN FAILED",
    "validating": "VALIDATING",
    "cancelled": "CANCELLED",
    "planned": "PLANNED",
    "approved": "APPROVED",
    "preparing": "PREPARING",
    "running": "RUNNING",
}

LEGEND = (
    ("baseline", BASELINE_COLOR),
    ("shipped / validated", "var(--chart-good)"),
    ("kept", "var(--chart-info)"),
    ("rejected", "var(--chart-bad)"),
    ("failed / cancelled", "var(--chart-muted)"),
)

WINNING_STATUSES = frozenset({"implementation_ready", "validated", "promising"})


def _delta_text(delta_pct: float | None) -> str:
    if delta_pct is None:
        return ""
    return f"{delta_pct:+.1f}%"


def _inherited_from(nodes: list[GraphNode], baseline_value: float) -> dict[str, str]:
    """Nodes that measured exactly what they were built on, and what that was.

    Every card's percentage is against the frozen baseline, which is what makes
    the graph comparable — but it means a child that changed nothing still
    wears its parent's gain. This is how a card can say so: an exact tie with
    everything it builds on is a change that the benchmark could not see, which
    is worth reading as a result rather than as a win.
    """
    value_of = {node.experiment_id: node.value for node in nodes}
    inherited: dict[str, str] = {}
    for node in nodes:
        if node.value is None:
            continue
        parents = node.parent_ids or [ROOT_KEY]
        references = [
            baseline_value if parent == ROOT_KEY else value_of.get(parent) for parent in parents
        ]
        if any(reference is None for reference in references):
            continue
        if all(node.value == reference for reference in references):
            inherited[node.experiment_id] = " + ".join(
                "the baseline" if parent == ROOT_KEY else parent for parent in parents
            )
    return inherited


def _tooltip(
    node: GraphNode, layout: GraphLayout, metric_name: str, inherited_from: str | None = None
) -> str:
    """The full detail a card has no room for, as a native SVG tooltip.

    `<title>` is what a browser shows on hover without any script, which keeps
    the dashboard a single file that works from disk.
    """
    lines = [f"{node.experiment_id} — {node.title}", f"status: {node.status}"]
    if node.value is not None:
        delta = _delta_text(node.delta_pct)
        against = f"  ({delta} vs baseline)" if delta else ""
        lines.append(f"{metric_name}: {_fmt(node.value)}{against}")
        if inherited_from is not None:
            lines.append(
                f"no change vs {inherited_from}: this {metric_name} is inherited, "
                "the experiment's own change measured nothing"
            )
    else:
        lines.append(f"{metric_name}: not measured")
    if node.round_num is not None:
        lines.append(f"autorun round: {node.round_num}")
    chain = [route.parent for route in layout.routes if route.child == node.experiment_id]
    lines.append(
        "builds on: "
        + ", ".join("baseline" if p == ROOT_KEY else p for p in chain)
    )
    if node.observation:
        lines.append(f"observed: {node.observation}")
    return f"<title>{escape(chr(10).join(lines))}</title>"


def _legend(width: float, y: float) -> str:
    parts = [
        f"<text x='18' y='{y}' {FONT} font-size='10' fill='var(--fg-muted)' "
        "letter-spacing='0.4'>LEGEND</text>"
    ]
    x = 76.0
    for label, color in LEGEND:
        parts.append(
            f"<rect x='{x}' y='{y - 8}' width='9' height='9' rx='2' fill='{color}'/>"
            f"<text x='{x + 15}' y='{y}' {FONT} font-size='10' fill='var(--fg-muted)'>"
            f"{escape(label)}</text>"
        )
        x += 26 + len(label) * 5.6
    parts.append(
        f"<line x1='{x + 6}' y1='{y - 4}' x2='{x + 32}' y2='{y - 4}' "
        "stroke='var(--fg-muted)' stroke-width='1.5' stroke-dasharray='4 3'/>"
        f"<text x='{x + 40}' y='{y}' {FONT} font-size='10' fill='var(--fg-muted)'>"
        "merge (multi-parent)</text>"
    )
    return "".join(parts)


def graph_chart(
    nodes: list[GraphNode],
    baseline_value: float,
    metric_name: str,
    link_base: str | None = None,
    best_experiment_id: str | None = None,
) -> str:
    """The experiment DAG: baseline root, one card per experiment, edges to parents.

    Positions come from `graph_layout`, so a node with several parents and an
    edge that skips a layer are drawn as they are rather than approximated.
    `best_experiment_id` traces that node's ancestors in the winning colour,
    which is how the picture answers "what actually got us here".
    """
    # Layer membership is fixed by the graph, but the order within a layer is
    # only a tie-break — so spend it on grouping each autorun round together.
    keys = [
        node.experiment_id
        for node in sorted(nodes, key=lambda n: (n.round_num or 0, n.experiment_id))
    ]
    layout = layout_graph(keys, {node.experiment_id: node.parent_ids for node in nodes})
    positions = {box.key: box for box in layout.boxes}
    metrics = Metrics()

    winning_path = _path_to_best(layout, best_experiment_id)
    inherited = _inherited_from(nodes, baseline_value)
    legend_y = layout.height + 4
    height = legend_y + 14

    parts = [
        f"<svg viewBox='0 0 {layout.width:.0f} {height:.0f}' role='img' "
        "aria-label='Experiment graph' xmlns='http://www.w3.org/2000/svg'>"
    ]

    edges: list[str] = []
    for route in layout.routes:
        child = next((n for n in nodes if n.experiment_id == route.child), None)
        on_path = (route.parent, route.child) in winning_path
        if on_path:
            stroke, width_px = "var(--chart-good)", "2.5"
        elif child is not None and child.status in WINNING_STATUSES:
            stroke, width_px = status_color(child.status), "2"
        else:
            stroke, width_px = "var(--grid)", "1.5"
        dashes = " stroke-dasharray='5 4'" if child is not None and child.is_merge else ""
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in route.points)
        label = "baseline" if route.parent == ROOT_KEY else route.parent
        edges.append(
            f"<polyline points='{points}' fill='none' stroke='{stroke}' "
            f"stroke-width='{width_px}' stroke-linecap='round'{dashes} "
            f"data-graph-edge='{escape(label)}->{escape(route.child)}'/>"
        )
    parts.append("".join(edges))

    root = positions[ROOT_KEY]
    parts.append(
        f"<rect x='{root.x}' y='{root.y}' width='{metrics.node_w}' "
        f"height='{metrics.node_h}' rx='10' fill='var(--card)' "
        f"stroke='{BASELINE_COLOR}' stroke-width='2' data-graph-node='baseline'/>"
        f"<text x='{root.x + 12}' y='{root.y + 22}' {FONT} font-size='11' font-weight='700' "
        f"fill='{BASELINE_COLOR}' letter-spacing='0.5'>BASELINE</text>"
        f"<text x='{root.x + 12}' y='{root.y + 42}' {FONT} font-size='12' fill='var(--fg)'>"
        f"{escape(metric_name)} = {_fmt(baseline_value)}</text>"
        f"<text x='{root.x + 12}' y='{root.y + 61}' {FONT} font-size='10' "
        "fill='var(--fg-muted)'>frozen reference for every measurement</text>"
    )

    for node in nodes:
        box = positions.get(node.experiment_id)
        if box is None:
            continue
        card = _graph_card(
            node,
            box,
            metrics,
            metric_name,
            layout,
            on_winning_path=node.experiment_id in {c for _, c in winning_path},
            inherited_from=inherited.get(node.experiment_id),
        )
        if link_base is not None:
            card = f"<a href='{link_base}/{escape(node.experiment_id)}'>{card}</a>"
        parts.append(card)

    parts.append(_legend(layout.width, legend_y))
    parts.append("</svg>")
    return "".join(parts)


def _path_to_best(layout: GraphLayout, best_experiment_id: str | None) -> set[tuple[str, str]]:
    """The edges on the ancestor chain of the best node, as (parent, child).

    Every ancestor counts, not just one lineage: a merge got there through both
    its parents and highlighting only the first would misreport the history.
    """
    if best_experiment_id is None:
        return set()
    edges: set[tuple[str, str]] = set()
    frontier = [best_experiment_id]
    seen = {best_experiment_id}
    while frontier:
        current = frontier.pop()
        for route in layout.routes:
            if route.child != current:
                continue
            edges.add((route.parent, route.child))
            if route.parent not in seen:
                seen.add(route.parent)
                frontier.append(route.parent)
    return edges


def _graph_card(
    node: GraphNode,
    box: Box,
    metrics: Metrics,
    metric_name: str,
    layout: GraphLayout,
    on_winning_path: bool,
    inherited_from: str | None = None,
) -> str:
    color = status_color(node.status)
    badge = STATUS_BADGE.get(node.status, node.status.upper())
    if node.is_merge:
        badge = f"{badge} · MERGE"
    if inherited_from is not None:
        badge = f"{badge} · NO CHANGE"
    title_room = 26 if node.round_num is not None else 31
    title = node.title if len(node.title) <= title_room else node.title[: title_room - 1] + "…"
    left, top = box.x, box.y

    parts = [
        f"<rect x='{left}' y='{top}' width='{metrics.node_w}' height='{metrics.node_h}' "
        f"rx='10' fill='var(--card)' stroke='{color}' "
        f"stroke-width='{2.5 if on_winning_path else 1.5}' "
        f"data-graph-node='{escape(node.experiment_id)}' "
        f"data-status='{escape(node.status)}'"
        + (f" data-round='{node.round_num}'" if node.round_num is not None else "")
        + "/>",
        _tooltip(node, layout, metric_name, inherited_from),
        f"<text x='{left + 12}' y='{top + 21}' {FONT} font-size='11' font-weight='700' "
        f"fill='var(--fg)'>{escape(node.experiment_id)}</text>",
        f"<text x='{left + 12}' y='{top + 40}' {FONT} font-size='10' "
        f"fill='var(--fg-muted)'>{escape(title)}</text>",
        f"<text x='{left + 12}' y='{top + 61}' {FONT} font-size='9.5' font-weight='600' "
        f"fill='{color}' letter-spacing='0.3'>{escape(badge)}</text>",
    ]

    delta = _delta_text(node.delta_pct)
    if delta:
        pill_w = 14 + len(delta) * 6.0
        parts.append(
            f"<rect x='{left + metrics.node_w - 12 - pill_w}' y='{top + 10}' "
            f"width='{pill_w:.1f}' height='16' rx='8' fill='{color}' opacity='0.16'/>"
            f"<text x='{left + metrics.node_w - 12 - pill_w / 2}' y='{top + 21.5}' {FONT} "
            f"font-size='10' font-weight='700' text-anchor='middle' fill='{color}' "
            f"data-graph-delta='{escape(node.experiment_id)}'>{escape(delta)}</text>"
        )
    if node.round_num is not None:
        parts.append(
            f"<text x='{left + metrics.node_w - 12}' y='{top + 40}' {FONT} font-size='9.5' "
            f"text-anchor='end' fill='var(--fg-muted)'>R{node.round_num}</text>"
        )
    value_text = _fmt(node.value) if node.value is not None else "not measured"
    parts.append(
        f"<text x='{left + metrics.node_w - 12}' y='{top + 61}' {FONT} font-size='10' "
        f"text-anchor='end' fill='var(--fg)'>{escape(value_text)}</text>"
    )
    return "".join(parts)
