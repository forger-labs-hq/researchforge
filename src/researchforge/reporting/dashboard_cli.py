"""`researchforge dashboard` — self-contained HTML dashboard command."""

from __future__ import annotations

import json
import webbrowser
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from researchforge.config.paths import reports_dir
from researchforge.storage.db import open_project_db
from researchforge.utils.output import JsonOption

DASHBOARD_FILENAME = "dashboard.html"


def _hub_hint() -> None:
    """Point at the always-on hub — the live view most people want."""
    from researchforge.server.monitor import read_hub

    record = read_hub()
    if record is not None:
        typer.echo(f"Looking for the live view of all projects? The hub is at {record.url}")


def dashboard_command(
    run_id: Annotated[
        str | None,
        typer.Option("--run", help="Run to visualize (default: the latest run)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write the dashboard here instead of the default path."),
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the dashboard in the default browser.")
    ] = False,
    json_output: JsonOption = False,
    legacy: Annotated[
        bool, typer.Option("--legacy", help="Force the old static HTML report (repo projects).")
    ] = False,
    port: Annotated[
        int, typer.Option("--port", help="Port for the live dashboard [default: 7842]")
    ] = 7842,
) -> None:
    """Open the live ResearchForge dashboard at localhost:7842.

    One dashboard for every project on this machine — Kaggle overnight dirs
    (.overnight_*/) and repo projects (.researchforge/) — found through the
    project registries and a scan of every known workspace root, wherever you
    run this from. Views: Home, Automation, Runs, DAG, Hypotheses,
    Submissions, Timeline, Insights.

    Pass --legacy (or any of --json / --run / --output) to write the static
    HTML report for the repo project in the current directory.

    Examples:
        researchforge dashboard              # auto-detect, open live dashboard
        researchforge dashboard --open       # same + open browser
        researchforge dashboard --port 8080  # custom port
        researchforge dashboard --legacy     # old static HTML for repo project
    """
    # The live dashboard discovers every project on this machine (registries +
    # workspace scan), so it no longer depends on what the cwd contains.
    # --json / --run / --output are report options: they imply the static report.
    cwd = Path.cwd()
    if not (legacy or json_output or run_id or output):
        from researchforge.kaggle.dashboard import is_running, serve

        running, dash_port = is_running(port)
        if running:
            url = f"http://localhost:{dash_port}"
            typer.echo(f"✅ Dashboard already running → {url}")
            if open_browser:
                webbrowser.open(url)
        else:
            serve(port=port, open_browser=open_browser, cwd=cwd)
        return

    # --legacy path: old static HTML report for repo project

    from researchforge.domain.deliverable import Deliverable, DeliverableKind
    from researchforge.reporting.dashboard import build_dashboard
    from researchforge.storage.baseline_repository import get_latest_successful_baseline
    from researchforge.storage.contract_repository import get_active_contract
    from researchforge.storage.deliverable_repository import record_deliverable_once
    from researchforge.storage.experiment_repository import get_run, list_runs
    from researchforge.storage.project_repository import get_project

    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None or project.objective is None:
            typer.echo(
                "This builds a static chart snapshot for ONE project with recorded "
                "data — define the project first: `researchforge project create`."
            )
            _hub_hint()
            raise typer.Exit(code=1)
        contract = get_active_contract(conn)
        baseline = get_latest_successful_baseline(conn)
        if contract is None or baseline is None:
            typer.echo(
                "The dashboard needs an approved contract and a successful baseline — "
                "run `researchforge contract approve` and `researchforge baseline run` first."
            )
            _hub_hint()
            raise typer.Exit(code=1)

        if run_id is not None:
            run = get_run(conn, run_id)
            if run is None:
                typer.echo(f"Unknown run: {run_id}.")
                raise typer.Exit(code=1)
        else:
            runs = list_runs(conn)
            run = runs[-1] if runs else None

        html = build_dashboard(conn, run)

        path = output or (reports_dir() / DASHBOARD_FILENAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

        record_deliverable_once(
            conn,
            project.id,
            Deliverable(
                deliverable_id=uuid4().hex,
                kind=DeliverableKind.DASHBOARD,
                location=str(path),
                created_at=datetime.now(UTC),
            ),
        )

    if open_browser:
        webbrowser.open(path.resolve().as_uri())

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "path": str(path),
                    "run_id": run.run_id if run is not None else None,
                    "bytes": len(html.encode("utf-8")),
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"Dashboard written to {path}")
        if not open_browser:
            typer.echo("Open it with: researchforge dashboard --open")
