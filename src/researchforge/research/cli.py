"""`researchforge research` and `researchforge papers` sub-apps."""

from __future__ import annotations

import json
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from researchforge.analytics.service import record_event
from researchforge.config.settings import load_settings
from researchforge.domain.paper import Paper
from researchforge.research.arxiv_client import ArxivClient, ArxivError
from researchforge.research.context_export import build_context, write_context
from researchforge.research.importers import import_landscape
from researchforge.research.search_service import CitedPapersError, run_search
from researchforge.storage.db import open_project_db
from researchforge.storage.paper_repository import (
    get_paper,
    list_papers,
    paper_ids,
    upsert_paper,
)
from researchforge.storage.project_repository import get_project
from researchforge.storage.scan_repository import get_latest_scan
from researchforge.storage.synthesis_repository import get_landscape
from researchforge.utils.output import JsonOption, echo_import_result, echo_json, echo_model

research_app = typer.Typer(name="research", no_args_is_help=True, help="Paper discovery.")
papers_app = typer.Typer(name="papers", no_args_is_help=True, help="Stored paper records.")


@research_app.command()
def search(
    query: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Search query (repeatable). Omit to auto-generate."),
    ] = None,
    max_candidates: Annotated[int | None, typer.Option("--max-candidates", min=10)] = None,
    select: Annotated[int | None, typer.Option("--select", "--n", "-n", min=1)] = None,
    categories: Annotated[
        list[str] | None,
        typer.Option(
            "--categories",
            "-c",
            help="arXiv category filter (repeatable). E.g. cs.CV cs.LG",
        ),
    ] = None,
    min_score: Annotated[
        float | None,
        typer.Option("--min-score", min=0.0, max=1.0, help="Minimum relevance score (0–1)."),
    ] = None,
    since: Annotated[
        datetime | None,
        typer.Option(
            "--since",
            formats=["%Y-%m-%d"],
            help="Only papers submitted on or after this date, e.g. 2024-01-01.",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace papers already cited by hypotheses.")
    ] = False,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider for query generation: anthropic|google|openai. "
            "Auto-detected from env when omitted.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Discover, deduplicate, rank, and store relevant arXiv papers.

    When an AI provider is configured (ANTHROPIC_API_KEY / GEMINI_API_KEY /
    OPENAI_API_KEY), queries are generated automatically from your objective.
    Pass --query to override with your own queries.
    """
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None or project.objective is None:
            typer.echo("Define the project first: `researchforge project create`.")
            raise typer.Exit(code=1)
        scan = get_latest_scan(conn)
        settings = load_settings()
        if max_candidates is not None:
            settings = settings.model_copy(update={"max_candidates": max_candidates})

        effective_queries = list(query) if query else None

        # AI-powered query generation when no explicit queries given
        if effective_queries is None:
            try:
                from researchforge.ai.providers import get_provider
                from researchforge.ai.query_gen import generate_queries_with_ai

                ai_provider = get_provider(provider_hint=provider)
                if ai_provider is not None:
                    if not json_output:
                        typer.echo(f"Generating search queries via {ai_provider.name}…")
                    ai_queries = generate_queries_with_ai(
                        project.objective, scan, settings, ai_provider
                    )
                    if ai_queries:
                        effective_queries = ai_queries
                        if not json_output:
                            typer.echo(f"AI-generated {len(ai_queries)} queries:")
                            for q in ai_queries:
                                typer.echo(f"  - {q}")
            except Exception:  # noqa: BLE001
                pass  # fall back to algorithmic query generation silently

        if not json_output:
            typer.echo("Searching arXiv…")
        try:
            outcome = run_search(
                conn,
                project,
                scan,
                queries=effective_queries,
                settings=settings,
                client=ArxivClient(
                    category_filter=categories,
                    submitted_since=since.date() if since is not None else None,
                ),
                select=select,
                force=force,
                min_score=min_score,
            )
        except CitedPapersError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None
        except ArxivError as exc:
            typer.echo(f"arXiv retrieval failed: {exc}")
            raise typer.Exit(code=1) from None

    record_event("papers_retrieved")
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "run_id": outcome.run_id,
                    "queries": outcome.queries,
                    "fetched_count": outcome.fetched_count,
                    "deduped_count": outcome.deduped_count,
                    "selected_count": len(outcome.selected),
                    "papers": [p.model_dump(mode="json") for p in outcome.selected],
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"Queries ({len(outcome.queries)}):")
        for q in outcome.queries:
            typer.echo(f"  - {q}")
        typer.echo(
            f"Fetched {outcome.fetched_count}, deduplicated to {outcome.deduped_count}, "
            f"selected {len(outcome.selected)}."
        )
        for paper in outcome.selected[:10]:
            typer.echo(f"  {paper.relevance_score:.3f}  {paper.paper_id}  {paper.title[:70]}")
        if len(outcome.selected) > 10:
            typer.echo(f"  ... and {len(outcome.selected) - 10} more (`researchforge papers list`)")
        if provider:
            typer.echo("Next: researchforge research synthesize")
        else:
            typer.echo(
                "Next: researchforge research context  "
                "(or research synthesize --provider anthropic|google|openai)"
            )


@research_app.command()
def context(
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write the bundle here instead of the default path."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Export the synthesis context bundle for Claude to read."""
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None or project.objective is None:
            typer.echo("Define the project first: `researchforge project create`.")
            raise typer.Exit(code=1)
        scan = get_latest_scan(conn)
        bundle = build_context(conn, project, scan, load_settings())

    if not bundle.papers:
        typer.echo("No papers stored. Run `researchforge research search` first.")
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(bundle.model_dump_json(indent=2))
        return

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        path = output
    else:
        path = write_context(bundle)

    typer.echo(f"Synthesis context written to {path}")
    typer.echo("Options:")
    typer.echo(
        "  A) Auto-synthesize with AI:  researchforge research synthesize "
        "--provider anthropic|google|openai"
    )
    typer.echo("  B) Ask Claude / Cursor to read the context and write:")
    typer.echo(f"       - {bundle.expected_artifacts.landscape_path}")
    typer.echo(f"       - {bundle.expected_artifacts.hypotheses_path}")
    typer.echo("     Then import them:")
    typer.echo("       researchforge research landscape --import <landscape file>")
    typer.echo("       researchforge hypotheses import <hypotheses file>")


@research_app.command()
def synthesize(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider: anthropic|google|openai. Auto-detected from env when omitted.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Override model name (e.g. claude-opus-4-5, gemini-2.0-flash, gpt-4o).",
        ),
    ] = None,
    no_import: Annotated[
        bool,
        typer.Option("--no-import", help="Write YAML files but skip importing them."),
    ] = False,
    from_results: Annotated[
        bool,
        typer.Option(
            "--from-results",
            help="Ground the synthesis in what this project has already measured. "
            "New hypotheses are ADDED to the stored set, and candidates that "
            "restate one already on record are skipped.",
        ),
    ] = False,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run",
            help="With --from-results, use one run's outcomes instead of all of them.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Generate landscape and hypotheses using a built-in AI provider.

    Works without Claude Code or Cursor — just set an API key:
      export ANTHROPIC_API_KEY=sk-ant-...
      export GEMINI_API_KEY=...
      export OPENAI_API_KEY=sk-...

    ResearchForge will auto-detect the provider from the key. Override with
    --provider and --model if you want a specific one.

    Pass --from-results after experiments have run to re-synthesize from the
    measurements: the AI sees the research log and the recorded outcomes, and
    the hypotheses already on record are kept.
    """
    from researchforge.ai.service import build_results_context, resolve_provider, run_synthesis

    if run_id is not None and not from_results:
        typer.echo("--run only applies with --from-results.", err=True)
        raise typer.Exit(code=1)

    try:
        ai_provider = resolve_provider(provider_hint=provider, model_hint=model)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    results_context: str | None = None
    if from_results:
        try:
            results_context = build_results_context(run_id)
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from None
        if not results_context:
            typer.echo(
                "No results to synthesize from yet. Run experiments first, or drop "
                "--from-results for a fresh synthesis.",
                err=True,
            )
            raise typer.Exit(code=1)

    if not json_output:
        typer.echo(f"Synthesizing with {ai_provider.name}…")

    from researchforge.utils.progress import LiveProgress

    try:
        label = f"Synthesizing with {ai_provider.name}"
        with LiveProgress(label, enabled=not json_output) as progress:
            progress.phase("Reading papers and building context…")
            outcome = run_synthesis(
                ai_provider, do_import=not no_import, results_context=results_context
            )
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "landscape_path": str(outcome.landscape_path),
                    "hypotheses_path": str(outcome.hypotheses_path),
                    "landscape_ok": outcome.landscape_result.ok,
                    "hypotheses_ok": outcome.hypotheses_result.ok,
                    "added_hypotheses": outcome.added_hypotheses,
                    "restated_hypotheses": outcome.restated_hypotheses,
                    "errors": outcome.landscape_result.errors + outcome.hypotheses_result.errors,
                    "warnings": (
                        outcome.landscape_result.warnings + outcome.hypotheses_result.warnings
                    ),
                },
                indent=2,
            )
        )
        return

    if no_import:
        typer.echo(f"Wrote {outcome.landscape_path}")
        typer.echo(f"Wrote {outcome.hypotheses_path}")
        return

    if outcome.ok:
        record_event("landscape_imported")
        record_event("hypotheses_imported")
        typer.echo(f"✓ Wrote {outcome.landscape_path}")
        typer.echo(f"✓ Wrote {outcome.hypotheses_path}")
        if from_results:
            for warning in outcome.hypotheses_result.warnings:
                typer.echo(f"  {warning}")
            typer.echo(
                f"✓ {len(outcome.added_hypotheses)} new hypothesis(es) added: "
                f"{', '.join(outcome.added_hypotheses) or 'none'}"
            )
            typer.echo("Next: researchforge experiment plan <hypothesis-id>")
        else:
            typer.echo("✓ Landscape and hypotheses imported.")
            typer.echo("Next: researchforge contract approve")
    else:
        if not outcome.landscape_result.ok:
            typer.echo("✗ Landscape import failed:")
            for err in outcome.landscape_result.errors:
                typer.echo(f"  {err}")
        if not outcome.hypotheses_result.ok:
            typer.echo("✗ Hypotheses import failed:")
            for err in outcome.hypotheses_result.errors:
                typer.echo(f"  {err}")
        raise typer.Exit(code=1)


@research_app.command()
def landscape(
    import_file: Annotated[
        Path | None,
        typer.Option("--import", "-i", help="Validate and import a landscape artifact."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Show the stored research landscape, or import one with --import."""
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None:
            typer.echo("No project found. Run `researchforge project create` first.")
            raise typer.Exit(code=1)

        if import_file is not None:
            result = import_landscape(conn, import_file, project.id)
            if result.ok:
                record_event("landscape_imported")
            stored = get_landscape(conn) if result.ok else None
            success = (
                f"Landscape imported: {len(stored.directions)} direction(s), "
                f"{len(stored.paper_annotations)} paper annotation(s), "
                f"{len(stored.evidence)} evidence claim(s)."
                if stored is not None
                else "Landscape imported."
            )
            echo_import_result(result.errors, result.warnings, success, json_output)
            return

        stored = get_landscape(conn)

    if stored is None:
        typer.echo(
            "No landscape imported yet. Run `researchforge research context`, have Claude "
            "write the artifact, then re-run with --import <file>."
        )
        raise typer.Exit(code=1)

    if json_output:
        echo_model(stored)
        return

    typer.echo(f"Summary: {stored.summary}")
    for direction in stored.directions:
        typer.echo(f"\n[{direction.direction_id}] {direction.name}")
        typer.echo(f"  {direction.description}")
        typer.echo(f"  Papers: {', '.join(direction.paper_ids)}")
        for finding in direction.established_findings:
            typer.echo(f"  finding: {finding}")
        for contradiction in direction.contradictions:
            typer.echo(f"  contradiction: {contradiction}")
        for limitation in direction.limitations:
            typer.echo(f"  limitation: {limitation}")
        for aspect in direction.underexplored_aspects:
            typer.echo(f"  underexplored: {aspect}")
    typer.echo(
        f"\n{len(stored.paper_annotations)} paper annotation(s), "
        f"{len(stored.evidence)} evidence claim(s)."
    )


def _print_paper(paper: Paper) -> None:
    typer.echo(f"{paper.paper_id}  (relevance {paper.relevance_score:.3f})")
    typer.echo(f"Title:      {paper.title}")
    typer.echo(f"Authors:    {', '.join(paper.authors)}")
    typer.echo(f"Published:  {paper.published_at.date().isoformat()}")
    typer.echo(f"Categories: {', '.join(paper.categories)}")
    typer.echo(f"Link:       {paper.source_url}")
    if paper.method_summary:
        typer.echo(f"Method:     {paper.method_summary}")
    if paper.evidence_strength.value != "unknown":
        typer.echo(f"Evidence:   {paper.evidence_strength.value}")
    for finding in paper.reported_findings:
        typer.echo(f"  finding:    {finding}")
    for limitation in paper.limitations:
        typer.echo(f"  limitation: {limitation}")
    if paper.repository_relevance:
        typer.echo(f"Repo relevance: {paper.repository_relevance}")
    if paper.supports_hypotheses:
        typer.echo(f"Supports:   {', '.join(paper.supports_hypotheses)}")
    if paper.contradicts_hypotheses:
        typer.echo(f"Contradicts: {', '.join(paper.contradicts_hypotheses)}")
    typer.echo(f"Abstract:   {paper.abstract[:400]}{'…' if len(paper.abstract) > 400 else ''}")


@papers_app.command("list")
def list_command(
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    json_output: JsonOption = False,
) -> None:
    """List stored papers ordered by relevance."""
    with closing(open_project_db()) as conn:
        papers = list_papers(conn, limit=limit)
    if json_output:
        typer.echo(json.dumps([p.model_dump(mode="json") for p in papers], indent=2))
        return
    if not papers:
        typer.echo("No papers stored. Run `researchforge research search` first.")
        return
    for paper in papers:
        typer.echo(f"{paper.relevance_score:.3f}  {paper.paper_id}  {paper.title[:80]}")


@papers_app.command("show")
def show_command(
    paper_id: Annotated[str, typer.Argument(help="e.g. arxiv:2401.12345")],
    json_output: JsonOption = False,
) -> None:
    """Show one stored paper in full."""
    with closing(open_project_db()) as conn:
        paper = get_paper(conn, paper_id)
    if paper is None:
        typer.echo(f"Unknown paper id: {paper_id}. See `researchforge papers list`.")
        raise typer.Exit(code=1)
    if json_output:
        echo_model(paper)
    else:
        _print_paper(paper)


PAPERS_EXPORT_VERSION = 1


@papers_app.command("export")
def export_command(
    output: Annotated[Path, typer.Argument(help="Destination, e.g. papers.json")],
    json_output: JsonOption = False,
) -> None:
    """Write every stored paper to a JSON file.

    Papers normally arrive from arXiv, which a machine without outbound network
    access cannot reach. Exporting them makes the searched-and-scored set
    portable, so the search runs where there is a network and the research runs
    where the code is.
    """
    with closing(open_project_db()) as conn:
        papers = list_papers(conn)
    if not papers:
        typer.echo("No papers stored. Run `researchforge research search` first.")
        raise typer.Exit(code=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": PAPERS_EXPORT_VERSION,
                "papers": [paper.model_dump(mode="json") for paper in papers],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if json_output:
        echo_json({"path": str(output), "paper_count": len(papers)})
        return
    typer.echo(f"Exported {len(papers)} paper(s) to {output}")


@papers_app.command("import")
def import_papers_command(
    file: Annotated[Path, typer.Argument(help="A file written by `papers export`.")],
    json_output: JsonOption = False,
) -> None:
    """Import papers from an exported JSON file.

    Existing papers with the same id are replaced, so re-importing the same file
    is safe and never doubles the stored set.
    """
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None:
            typer.echo("No project found. Run `researchforge project create` first.")
            raise typer.Exit(code=1)

        try:
            papers = read_papers_export(file)
        except (OSError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from None

        known = paper_ids(conn)
        for paper in papers:
            upsert_paper(conn, project.id, paper)

    added = sum(1 for paper in papers if paper.paper_id not in known)
    if json_output:
        echo_json({"imported": len(papers), "added": added, "replaced": len(papers) - added})
        return
    typer.echo(f"Imported {len(papers)} paper(s): {added} new, {len(papers) - added} replaced.")
    typer.echo("Next: researchforge research synthesize")


def read_papers_export(file: Path) -> list[Paper]:
    """Validate an export file at the boundary and return its papers.

    Everything about the file is untrusted: it came from another machine, and a
    partially-valid file must be rejected whole rather than half-imported.
    """
    payload = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{file}: expected a JSON object, found {type(payload).__name__}.")

    version = payload.get("schema_version")
    if version != PAPERS_EXPORT_VERSION:
        raise ValueError(
            f"{file}: schema_version {version!r} is not supported "
            f"(this build reads version {PAPERS_EXPORT_VERSION})."
        )

    raw = payload.get("papers")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{file}: 'papers' must be a non-empty list.")

    papers = []
    for index, entry in enumerate(raw):
        try:
            papers.append(Paper.model_validate(entry))
        except ValidationError as exc:
            raise ValueError(f"{file}: papers[{index}] is not a valid paper — {exc}") from None
    return papers
