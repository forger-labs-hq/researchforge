"""The hub: one machine-wide, read-only server for every registered project.

The home page lists all projects from the global registry with their folder
locations; each project's full monitor UI is served under `/p/<slug>/` by
reading that project's own database read-only on demand. No per-project
server processes, no port juggling — new projects appear as soon as they
register (which `researchforge init` does automatically).
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from researchforge import __version__
from researchforge.config.registry import RegistryEntry, find_by_slug, load_registry
from researchforge.server.app import render_dashboard_page
from researchforge.server.data import ProjectState, read_state
from researchforge.server.pages import (
    _PAGE_CSS,
    experiment_page,
    experiments_page,
    overview_page,
    research_page,
    run_page,
    session_page,
)

_LOGO_SEARCH_PATHS = [
    Path.home() / ".researchforge" / "logo.png",
    Path.home() / ".researchforge" / "logo.jpeg",
    Path.home() / ".researchforge" / "logo.jpg",
]


def _logo_path() -> Path | None:
    return next((p for p in _LOGO_SEARCH_PATHS if p.is_file()), None)


def _logo_html(size: int = 56, extra_style: str = "") -> str:
    """Return an <img> tag when a logo is installed, ASCII fox otherwise."""
    if _logo_path() is not None:
        style = f"width:{size}px;height:{size}px;object-fit:contain;border-radius:8px"
        if extra_style:
            style += f";{extra_style}"
        return (
            f"<img src='/logo' alt='ResearchForge' style='{style}' "
            "onerror=\"this.style.display='none'\">\n"
        )
    return "<pre class='rf-fox'>  /\\   /\\ \n ( o   o )\n  \\_____/ </pre>"


HUB_REFRESH_SECONDS = 15


def _compute_pill(state: ProjectState) -> str:
    """How much machine time this project has spent, at a glance.

    The hub answers "which of my projects is worth looking at?", and counts of
    papers and hypotheses do not distinguish a project that has been thinking
    from one that has actually been running.
    """
    from researchforge.config.settings import load_settings
    from researchforge.reporting.economics import compute_economics, humanize_seconds

    settings = load_settings()
    economics = compute_economics(
        state.experiments,
        state.executions,
        [state.baseline] if state.baseline is not None else [],
        None,
        calls=state.ai_calls,
        prices=settings.model_prices,
        usd_per_hour=settings.local_compute_usd_per_hour,
    )

    pills = ""
    total = economics.stages.total
    if total > 0:
        pills += f"<span class='stat-pill'>⏱ {escape(humanize_seconds(total))} compute</span>"
        avoided = economics.avoided.seconds
        if avoided is not None:
            pills += f"<span class='stat-pill'>⏭ {escape(humanize_seconds(avoided))} avoided</span>"

    spend = economics.tokens
    if spend.total_tokens:
        pills += f"<span class='stat-pill'>◇ {_compact(spend.total_tokens)} tokens</span>"
        # A dollar figure that silently excludes unpriced models would be worse
        # than none, so it appears only when every model had a rate.
        if spend.usd > 0 and spend.fully_priced:
            pills += f"<span class='stat-pill'>${spend.usd:,.2f}</span>"
    if spend.has_estimates:
        # The tilde is the whole point: this project's planning ran in an IDE,
        # so the figure is sized rather than metered.
        pills += (
            f"<span class='stat-pill'>◇ ~{_compact(spend.estimated_tokens)} tokens (IDE)</span>"
        )
    return pills


def _compact(count: int) -> str:
    """Token counts at a glance: 1.2M reads faster than 1,234,567 on a card."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}k"
    return str(count)


def _project_card(entry: RegistryEntry, state: ProjectState | None) -> str:
    """One project on the hub home page; honest about unreadable entries."""
    last_active = entry.last_active[:16].replace("T", " ")
    if state is None:
        reason = "folder moved or deleted" if not entry.exists else "database unreadable"
        return (
            "<div class='card project-card s-missing'>"
            f"<div style='display:flex;align-items:center;gap:8px;justify-content:space-between'>"
            f"<span style='font-weight:700;color:var(--fg)'>{escape(entry.name)}</span>"
            f"<span class='badge' style='background:var(--chart-bad)'>missing</span></div>"
            f"<div class='d' style='margin-top:6px'><code>{escape(entry.path)}</code></div>"
            f"<div class='d'>{escape(reason)} · last active {escape(last_active)}</div>"
            "</div>"
        )
    project = state.project
    mode = project.mode.value.replace("_", " ") if project.mode else "mode unset"
    status_val = project.status.value
    slug = entry.slug
    live_badge = (
        "<span class='live' style='color:var(--chart-good);font-weight:700;margin-right:6px'>"
        "● live</span>"
        if state.run_in_progress
        else ""
    )
    papers = len(state.papers)
    hypotheses = len(state.hypotheses)
    experiments = len(state.experiments)
    compute = _compute_pill(state)
    return (
        f"<div class='card project-card s-{escape(status_val)}'>"
        "<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:12px'>"
        f"<div>{live_badge}"
        f"<a href='/p/{escape(slug)}/' class='card-name'>{escape(entry.name)}</a></div>"
        f"<span class='badge' style='background:var(--chart-info);flex-shrink:0'>"
        f"{escape(status_val)}</span></div>"
        f"<div class='d' style='margin-top:6px'>{escape(mode)}</div>"
        "<div class='card-stats'>"
        f"<span class='stat-pill'>📄 {papers} paper{'s' if papers != 1 else ''}</span>"
        f"<span class='stat-pill'>💡 {hypotheses} hypothesis</span>"
        f"<span class='stat-pill'>⚗ {experiments} exp{'s' if experiments != 1 else ''}</span>"
        f"{compute}"
        "</div>"
        f"<div class='d' style='margin-top:8px'><code>{escape(entry.path)}</code></div>"
        f"<div class='d'>last active {escape(last_active)}</div>"
        "</div>"
    )


def hub_home_page() -> str:
    entries = sorted(load_registry(), key=lambda e: e.last_active, reverse=True)
    active_cards: list[str] = []
    missing_cards: list[str] = []
    for entry in entries:
        state: ProjectState | None = None
        try:
            state = read_state(Path(entry.path))
        except Exception:  # noqa: BLE001
            state = None
        (missing_cards if state is None else active_cards).append(_project_card(entry, state))

    if active_cards:
        main_body = f"<div class='project-grid'>{''.join(active_cards)}</div>"
    else:
        main_body = (
            "<div class='empty-state'>"
            f"{_logo_html(72, 'margin-bottom:16px;display:block')}"
            "<p style='font-size:1.1rem;font-weight:600;color:var(--fg)'>No projects yet</p>"
            "<p>Initialize one anywhere and it appears here automatically.</p>"
            "<p><code>researchforge init</code> &nbsp;or&nbsp; "
            "<code>researchforge -C ~/my-project init</code></p>"
            "</div>"
        )

    missing_section = ""
    if missing_cards:
        missing_section = (
            "<details style='margin-top:24px'>"
            f"<summary style='cursor:pointer;color:var(--fg-muted);font-size:0.85rem;"
            f"padding:8px 0;user-select:none'>"
            f"<span style='color:var(--chart-bad)'>▸</span>"
            f" {len(missing_cards)} missing project"
            f"{'s' if len(missing_cards) != 1 else ''}"
            " — folder moved or deleted &nbsp;·&nbsp; "
            "<code style='font-size:0.8rem'>researchforge hub --prune</code> to remove</summary>"
            f"<div class='project-grid' style='margin-top:12px;opacity:.6'>"
            f"{''.join(missing_cards)}</div>"
            "</details>"
        )

    project_count = len(active_cards)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>ResearchForge Hub</title>"
        f"<meta http-equiv='refresh' content='{HUB_REFRESH_SECONDS}'>"
        f"<style>{_PAGE_CSS}"
        ".project-card .card-name{{color:var(--fg);text-decoration:none}}"
        ".project-card .card-name:hover{{color:var(--brand)}}"
        "details>summary::marker{{color:var(--fg-muted)}}"
        "details[open]>summary span{{transform:rotate(90deg);display:inline-block}}"
        "</style></head>"
        "<body>"
        "<nav>"
        "<span class='nav-brand'>⬥ ResearchForge</span>"
        "<a href='/' class='active'>Hub</a>"
        f"<span style='margin-left:auto' class='sub'>v{__version__} · read-only · "
        f"refreshes every {HUB_REFRESH_SECONDS}s</span>"
        "</nav>"
        "<div class='rf-masthead'>"
        f"{_logo_html(80)}"
        "<div>"
        "<div class='rf-brand-name'>ResearchForge</div>"
        "<h1 style='margin:0 0 4px'>All Projects</h1>"
        f"<p class='sub' style='margin:0'>{project_count} active project"
        f"{'s' if project_count != 1 else ''} · "
        "newly initialized projects appear automatically</p>"
        "</div>"
        "<span style='margin-left:auto;color:var(--chart-good);font-weight:700' "
        "class='live'>● hub</span>"
        "</div>"
        f"{main_body}{missing_section}"
        f"<footer>ResearchForge {__version__} · hub · read-only · 127.0.0.1 only</footer>"
        "</body></html>"
    )


def create_hub_app() -> FastAPI:
    app = FastAPI(title="ResearchForge hub", docs_url=None, redoc_url=None)

    def project_state(slug: str) -> ProjectState:
        entry = find_by_slug(slug)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Unknown project: {slug}")
        try:
            state = read_state(Path(entry.path))
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Project '{slug}' is registered at {entry.path} but no longer "
                "readable there (moved or deleted?).",
            ) from exc
        state.link_prefix = f"/p/{slug}"
        return state

    def project_base(slug: str) -> Path:
        entry = find_by_slug(slug)
        assert entry is not None  # project_state ran first
        return Path(entry.path)

    @app.get("/logo")
    def logo() -> FileResponse:
        path = _logo_path()
        if path is None:
            raise HTTPException(status_code=404, detail="No logo installed.")
        media = "image/png" if path.suffix == ".png" else "image/jpeg"
        return FileResponse(str(path), media_type=media)

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return hub_home_page()

    @app.get("/p/{slug}/", response_class=HTMLResponse)
    @app.get("/p/{slug}", response_class=HTMLResponse)
    def overview(slug: str) -> str:
        return overview_page(project_state(slug))

    @app.get("/p/{slug}/research", response_class=HTMLResponse)
    def research(slug: str) -> str:
        return research_page(project_state(slug))

    @app.get("/p/{slug}/experiments", response_class=HTMLResponse)
    def experiments(slug: str) -> str:
        return experiments_page(project_state(slug))

    @app.get("/p/{slug}/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(slug: str, run_id: str) -> str:
        state = project_state(slug)
        if not any(run.run_id == run_id for run in state.runs):
            raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}")
        return run_page(state, run_id)

    @app.get("/p/{slug}/sessions/{search_run_id}", response_class=HTMLResponse)
    def session_detail(slug: str, search_run_id: str) -> str:
        state = project_state(slug)
        if not any(str(run["run_id"]) == search_run_id for run in state.search_runs):
            raise HTTPException(status_code=404, detail=f"Unknown session: {search_run_id}")
        return session_page(state, search_run_id)

    @app.get("/p/{slug}/experiments/{experiment_id}", response_class=HTMLResponse)
    def experiment_detail(slug: str, experiment_id: str) -> str:
        state = project_state(slug)
        if not any(e.experiment_id == experiment_id for e in state.experiments):
            raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment_id}")
        return experiment_page(state, experiment_id)

    @app.get("/p/{slug}/dashboard", response_class=HTMLResponse)
    def dashboard(slug: str, run: str | None = None) -> str:
        state = project_state(slug)
        return render_dashboard_page(state, project_base(slug), run)

    @app.get("/p/{slug}/api/state")
    def api_state(slug: str) -> JSONResponse:
        return JSONResponse(project_state(slug).model_dump(mode="json"))

    return app
