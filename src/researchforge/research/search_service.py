"""Orchestrates the retrieval pipeline: search -> dedup -> rank -> persist."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from researchforge.config.settings import ResearchSettings
from researchforge.domain.paper import Paper
from researchforge.domain.project import Project, ProjectStatus
from researchforge.domain.repo_scan import RepoScan
from researchforge.project.service import touch_project_status
from researchforge.research.arxiv_client import ArxivClient, ArxivEntry
from researchforge.research.dedup import deduplicate_entries
from researchforge.research.queries import generate_queries
from researchforge.research.ranking import build_query_document, rank_candidates
from researchforge.storage.paper_repository import (
    cited_paper_ids,
    delete_all_papers,
    record_search_run,
    upsert_paper,
)


class CitedPapersError(Exception):
    """Stored hypotheses cite existing papers; replacing them needs --force."""


@dataclass
class SearchOutcome:
    run_id: str
    queries: list[str]
    fetched_count: int
    deduped_count: int
    selected: list[Paper]


def _entry_to_paper(entry: ArxivEntry, score: float) -> Paper:
    return Paper(
        paper_id=entry.paper_id,
        title=entry.title,
        authors=entry.authors,
        published_at=entry.published_at,
        updated_at=entry.updated_at,
        abstract=entry.abstract,
        source_url=entry.source_url,
        pdf_url=entry.pdf_url,
        categories=entry.categories,
        relevance_score=round(score, 4),
    )


def run_search(
    conn: sqlite3.Connection,
    project: Project,
    scan: RepoScan | None,
    *,
    queries: list[str] | None,
    settings: ResearchSettings,
    client: ArxivClient,
    select: int | None = None,
    force: bool = False,
    min_score: float | None = None,
) -> SearchOutcome:
    """Run the full retrieval pipeline and persist the selected papers.

    Re-running replaces previously stored papers; refused when stored
    hypotheses already cite papers unless `force` is set.
    """
    if project.objective is None:
        raise ValueError("Project objective is required before searching.")

    cited = cited_paper_ids(conn)
    if cited and not force:
        raise CitedPapersError(
            f"{len(cited)} stored paper(s) are cited by hypotheses. "
            "Re-run with --force to replace the paper set."
        )

    effective_queries = queries or generate_queries(project.objective, scan, settings)
    per_query = max(settings.max_candidates // len(effective_queries), 1)

    fetched: list[ArxivEntry] = []
    for query in effective_queries:
        fetched.extend(client.search(query, max_results=per_query))

    deduped = deduplicate_entries(fetched)
    query_tokens = build_query_document(project.objective, scan, effective_queries)
    ranked = rank_candidates(deduped, query_tokens)

    # Apply minimum relevance score filter before selection
    if min_score is not None:
        ranked = [(entry, score) for entry, score in ranked if score >= min_score]

    selection_size = select or settings.selected_papers
    selected = [_entry_to_paper(entry, score) for entry, score in ranked[:selection_size]]

    delete_all_papers(conn)
    for paper in selected:
        upsert_paper(conn, project.id, paper)
    run_id = record_search_run(
        conn,
        project.id,
        queries=effective_queries,
        fetched_count=len(fetched),
        deduped_count=len(deduped),
        selected_count=len(selected),
        paper_ids=[paper.paper_id for paper in selected],
    )
    touch_project_status(conn, ProjectStatus.RESEARCHING)

    return SearchOutcome(
        run_id=run_id,
        queries=effective_queries,
        fetched_count=len(fetched),
        deduped_count=len(deduped),
        selected=selected,
    )
