"""ResearchForge CLI shell."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from researchforge.analytics.cli import analytics_app
from researchforge.analytics.service import record_event
from researchforge.claude.cli import claude_app
from researchforge.config.paths import is_initialized, researchforge_dir
from researchforge.config.paths_cli import paths_command
from researchforge.contract.cli import contract_app
from researchforge.domain.project import Project, ProjectMode, ProjectStatus
from researchforge.execution.cli import baseline_app
from researchforge.experiments.cli import experiment_app, results_app, validate_command
from researchforge.hypotheses.cli import hypotheses_app
from researchforge.project.cli import project_app
from researchforge.reporting.cli import report_app
from researchforge.reporting.dashboard_cli import dashboard_command
from researchforge.reporting.paper_cli import paper_app
from researchforge.repository.cli import repo_app
from researchforge.research.cli import papers_app, research_app
from researchforge.server.cli import hub_command, serve_command
from researchforge.shipping.cli import ship_app
from researchforge.storage.db import open_project_db
from researchforge.storage.project_repository import get_project, insert_project
from researchforge.utils.output import JsonOption
from researchforge.utils.system_checks import run_all_checks

app = typer.Typer(
    name="researchforge",
    no_args_is_help=True,
    add_completion=False,
    help="From papers to proof.",
)


@app.callback()
def _root(
    ctx: typer.Context,
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            "-C",
            help=(
                "Run as if started in this directory — the project (its database, "
                "worktrees, artifacts, dashboard) lives wherever you point this. "
                "Example: researchforge -C ~/Desktop/some_new_folder init"
            ),
            exists=True,
            file_okay=False,
        ),
    ] = None,
) -> None:
    if directory is not None:
        os.chdir(directory)

    # Git-style walk-up: run from any subfolder of a project (e.g. inside the
    # cloned repo being improved) and the command still finds the project root.
    # `init` is exempt — it creates projects and must honor the exact cwd.
    if ctx.invoked_subcommand != "init" and not is_initialized():
        from researchforge.config.paths import find_project_root

        root = find_project_root()
        if root is not None:
            os.chdir(root)
            typer.echo(f"Using project at {root}", err=True)

    if is_initialized():
        from researchforge.config.registry import touch_project

        with suppress(OSError):  # registry is best-effort bookkeeping
            touch_project(Path.cwd())

    if ctx.invoked_subcommand not in ("claude", None):
        with suppress(Exception):  # a convenience offer must never break a command
            _offer_skills_once()

    if ctx.invoked_subcommand not in ("hub", "serve", None) and not os.environ.get(
        "RESEARCHFORGE_NO_HUB"
    ):
        from researchforge.server.monitor import ensure_hub

        with suppress(Exception):  # the hub must never break a command
            ensure_hub()


def _is_interactive() -> bool:
    import sys

    return sys.stdin.isatty() and sys.stdout.isatty()


def _offer_skills_once() -> None:
    """First interactive run: offer the user-level Claude skills install.

    Consent-based and asked exactly once ever — the answer (either way) is
    recorded so the CLI never nags. Skipped entirely when not on a terminal,
    or when skills are already installed for this user or this project.
    """
    if not _is_interactive():
        return
    from researchforge.claude.installer import manifest_path
    from researchforge.config.registry import researchforge_home

    marker = researchforge_home() / "skills-offer-answered"
    if marker.exists() or manifest_path(user=True).exists() or manifest_path().exists():
        return
    wanted = typer.confirm(
        "Install the Claude Code skills for all sessions (~/.claude/skills), "
        "so /researchforge-start works everywhere?",
        default=True,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(("yes" if wanted else "no") + "\n", encoding="utf-8")
    if wanted:
        from researchforge.claude.installer import install_skills

        report = install_skills(user=True)
        typer.echo(
            f"Claude skills installed in {report.skills_dir} — open any Claude Code "
            "session and try /researchforge-start."
        )
    else:
        typer.echo("Skipped — run `researchforge claude install --user` anytime.")


app.add_typer(project_app, name="project")
app.add_typer(repo_app, name="repo")
app.add_typer(research_app, name="research")
app.add_typer(papers_app, name="papers")
app.add_typer(hypotheses_app, name="hypotheses")
app.add_typer(report_app, name="report")
app.add_typer(contract_app, name="contract")
app.add_typer(baseline_app, name="baseline")
app.add_typer(experiment_app, name="experiment")
app.add_typer(results_app, name="results")
app.command("validate")(validate_command)
app.command("dashboard")(dashboard_command)
app.command("serve")(serve_command)
app.command("hub")(hub_command)
app.command("paths")(paths_command)
app.add_typer(ship_app, name="ship")
app.add_typer(paper_app, name="paper")
app.add_typer(claude_app, name="claude")
app.add_typer(analytics_app, name="analytics")


@app.command()
def doctor(json_output: JsonOption = False) -> None:
    """Check that required and optional dependencies are available."""
    results = run_all_checks()

    if json_output:
        typer.echo(json.dumps([r.model_dump() for r in results], indent=2))
    else:
        for result in results:
            marker = "✓" if result.ok else ("✗" if result.required else "-")
            typer.echo(f"{marker} {result.name}: {result.detail}")
            if not result.ok and result.hint:
                typer.echo(f"    hint: {result.hint}")

    if any(not result.ok and result.required for result in results):
        raise typer.Exit(code=1)
    record_event("doctor_passed")


@app.command()
def init(
    claude: bool = typer.Option(
        False, "--claude", help="Also install the Claude Code skills into .claude/skills/."
    ),
    json_output: JsonOption = False,
) -> None:
    """Initialize a ResearchForge project in the current directory."""
    from researchforge.claude.installer import InstallReport, install_skills
    from researchforge.config.paths import find_project_root
    from researchforge.config.registry import touch_project

    already = is_initialized()
    ancestor = None if already else find_project_root(Path.cwd().parent)
    if ancestor is not None:
        typer.echo(
            f"Note: an initialized project already exists at {ancestor} — "
            "continuing creates a separate, nested project here.",
            err=True,
        )
    project: Project | None = None
    if not already:
        researchforge_dir().mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        project = Project(
            id=uuid4().hex,
            name=Path.cwd().name,
            status=ProjectStatus.INITIALIZED,
            created_at=now,
            updated_at=now,
        )
        with closing(open_project_db()) as conn:
            insert_project(conn, project)
        record_event("initialized")
        with suppress(OSError):
            touch_project(Path.cwd())

    skills: InstallReport | None = install_skills() if claude else None

    if json_output:
        payload: dict[str, object]
        if project is not None:
            payload = project.model_dump(mode="json")
        else:
            payload = {"status": "already_initialized"}
        if skills is not None:
            payload["skills"] = skills.model_dump()
        typer.echo(json.dumps(payload, indent=2))
        return

    if already:
        typer.echo("Already initialized.")
    else:
        assert project is not None
        typer.echo(f"Initialized ResearchForge project '{project.name}' in {researchforge_dir()}")
    if skills is not None:
        typer.echo(f"Claude skills installed in {skills.skills_dir}:")
        for result in skills.results:
            typer.echo(f"  /{result.skill} ({result.action.value})")
        if skills.conflicts:
            typer.echo(
                "Modified skills were left untouched; `researchforge claude install --force` "
                "overwrites them."
            )
        typer.echo(
            "Start in Claude Code with /researchforge-start, or from the CLI:\n"
            "  researchforge project create --mode explore_research_idea --objective ...\n"
            "  researchforge project create --mode improve_repository --objective ..."
        )


_COUNT_QUERIES = {
    "papers": "SELECT COUNT(*) AS n FROM papers",
    "hypotheses": "SELECT COUNT(*) AS n FROM hypotheses",
    "landscape": "SELECT COUNT(*) AS n FROM landscape",
}


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(_COUNT_QUERIES[table]).fetchone()
    return int(row["n"])


def _experiment_next_action(conn: sqlite3.Connection | None = None) -> str:
    """Next step once a baseline exists (Phase 1C planning surface).

    Uses `conn` when given (lets read-only callers like the monitoring
    server reuse this logic); otherwise opens the project db itself.
    """
    if conn is None:
        with closing(open_project_db()) as owned:
            return _experiment_next_action(owned)

    from researchforge.domain.deliverable import DeliverableKind
    from researchforge.domain.experiment import ExperimentStatus, PlanStatus
    from researchforge.storage.deliverable_repository import list_deliverables
    from researchforge.storage.experiment_repository import list_experiments, list_plans

    plans = list_plans(conn)
    experiments = list_experiments(conn)
    if not plans:
        return "researchforge experiment plan <hyp-id>  # plan experiment variants"
    latest = plans[-1]
    if latest.status is PlanStatus.PLANNED:
        return f"researchforge experiment approve {latest.plan_id}"
    if latest.status is PlanStatus.APPROVED:
        return f"researchforge experiment run {latest.plan_id}"
    branch_deliverables = list_deliverables(conn, kind=DeliverableKind.BRANCH)
    if any(e.status is ExperimentStatus.VALIDATED for e in experiments):
        return "researchforge ship branch  # reconstruct the validated winner as a clean branch"
    if any(e.status is ExperimentStatus.IMPLEMENTATION_READY for e in experiments):
        reports = list_deliverables(conn, kind=DeliverableKind.ENGINEERING_REPORT)
        prs = list_deliverables(conn, kind=DeliverableKind.DRAFT_PR)
        if not reports:
            return "researchforge report build  # engineering report for the shipped change"
        if branch_deliverables and not prs:
            return "researchforge ship pr  # optional draft PR — or: researchforge paper package"
        return (
            "Phase 1 complete — `researchforge paper package` builds the research "
            "bundle; `researchforge dashboard --open` visualizes the results."
        )
    return f"researchforge experiment run {latest.plan_id}  # or plan a new batch"


def _next_action(
    project: Project,
    papers: int,
    hypotheses: int,
    landscape: int,
    *,
    contract_version: int | None,
    contract_drifted: bool,
    baseline_failed: bool = False,
    conn: sqlite3.Connection | None = None,
) -> str:
    if project.mode is None or project.objective is None:
        return "researchforge project create"
    if project.mode is ProjectMode.IMPROVE_REPOSITORY and project.repository.path is None:
        return "researchforge repo scan"
    if contract_version is not None:
        if contract_drifted:
            return "researchforge contract approve  # researchforge.yaml changed since approval"
        if project.status not in (ProjectStatus.BASELINED, ProjectStatus.VALIDATED):
            if baseline_failed:
                return (
                    "Baseline failed — inspect .researchforge/artifacts/baseline/ and "
                    "re-run `researchforge baseline run`"
                )
            return "researchforge baseline run"
        return _experiment_next_action(conn)
    if papers == 0:
        return "researchforge research search"
    if landscape == 0 or hypotheses == 0:
        return (
            "researchforge research context — then ask Claude to write the synthesis "
            "artifacts and import them"
        )
    if project.status not in (
        ProjectStatus.REPORTED,
        ProjectStatus.CONTRACTED,
        ProjectStatus.BASELINED,
    ):
        return "researchforge report build"
    if project.mode is ProjectMode.IMPROVE_REPOSITORY:
        return "researchforge contract generate"
    return "Research complete — report generated. Attach a repository to run experiments."


@app.command()
def status(json_output: JsonOption = False) -> None:
    """Show the status of the current ResearchForge project."""
    from researchforge.config.paths import contract_path
    from researchforge.contract.service import check_contract_drift
    from researchforge.storage.contract_repository import get_active_contract

    if not is_initialized():
        typer.echo("Not an initialized ResearchForge project. Run `researchforge init`.")
        raise typer.Exit(code=1)

    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None:
            typer.echo("Project database exists but no project record was found.")
            raise typer.Exit(code=1)
        papers = _count(conn, "papers")
        hypotheses = _count(conn, "hypotheses")
        landscape = _count(conn, "landscape")
        contract = get_active_contract(conn)
        repo_root = Path(project.repository.path) if project.repository.path else Path.cwd()
        drifted = check_contract_drift(conn, contract_path(repo_root))
        from researchforge.domain.baseline import BaselineStatus
        from researchforge.storage.baseline_repository import get_latest_baseline

        latest_baseline = get_latest_baseline(conn)
        baseline_failed = (
            latest_baseline is not None and latest_baseline.status is not BaselineStatus.SUCCEEDED
        )

    next_action = _next_action(
        project,
        papers,
        hypotheses,
        landscape,
        contract_version=contract.contract_version if contract else None,
        contract_drifted=drifted,
        baseline_failed=baseline_failed,
    )

    if json_output:
        payload = project.model_dump(mode="json")
        payload["counts"] = {"papers": papers, "hypotheses": hypotheses, "landscape": landscape}
        payload["contract_version"] = contract.contract_version if contract else None
        payload["contract_drifted"] = drifted
        payload["next_action"] = next_action
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"Name:       {project.name}")
        typer.echo(f"Mode:       {project.mode.value if project.mode else 'unset'}")
        typer.echo(f"Objective:  {project.objective or 'unset'}")
        typer.echo(f"Status:     {project.status.value}")
        typer.echo(f"Papers:     {papers}")
        typer.echo(f"Hypotheses: {hypotheses}")
        if contract is not None:
            drift_note = " (drifted — re-approve)" if drifted else ""
            typer.echo(f"Contract:   v{contract.contract_version}{drift_note}")
        typer.echo(f"Created:    {project.created_at.isoformat()}")
        typer.echo(f"Updated:    {project.updated_at.isoformat()}")
        typer.echo(f"Next:       {next_action}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
