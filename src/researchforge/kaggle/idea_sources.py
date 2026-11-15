"""Idea sources: collect seeds for the patch proposer from arxiv, top kernels,
past winner write-ups, and GitHub repositories.

Each source is distilled into a short list of concrete ideas so the patch
proposer can cite them and build on them without hallucinating.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchforge.ai.providers import AiProvider
    from researchforge.kaggle.competition_reader import CompetitionMetadata


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class IdeaSourceType(StrEnum):
    ARXIV = "arxiv"
    KAGGLE_KERNEL = "kaggle_kernel"
    WINNER_WRITEUP = "winner_writeup"
    GITHUB = "github"
    USER_NOTE = "user_note"


@dataclass
class IdeaSource:
    source_type: IdeaSourceType
    title: str
    url: str
    ideas: list[str] = field(default_factory=list)  # concrete technique ideas
    relevance_score: float = 0.0
    raw_text: str = ""  # first 500 chars of source (for debugging)


@dataclass
class IdeaCollection:
    competition_slug: str
    sources: list[IdeaSource] = field(default_factory=list)

    @property
    def all_ideas(self) -> list[str]:
        """Flat deduplicated list of all ideas across sources."""
        seen: set[str] = set()
        out: list[str] = []
        for src in self.sources:
            for idea in src.ideas:
                low = idea.lower().strip()
                if low not in seen:
                    seen.add(low)
                    out.append(idea)
        return out

    def save(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / "idea_sources.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {
            "competition_slug": self.competition_slug,
            "sources": [{**asdict(s), "source_type": s.source_type.value} for s in self.sources],
        }
        with path.open("w") as fh:
            json.dump(serialised, fh, indent=2)

    @classmethod
    def load(cls, artifacts_dir: Path) -> IdeaCollection | None:
        path = artifacts_dir / "idea_sources.json"
        if not path.exists():
            return None
        with path.open() as fh:
            data = json.load(fh)
        sources = [
            IdeaSource(
                source_type=IdeaSourceType(s["source_type"]),
                title=s["title"],
                url=s["url"],
                ideas=s.get("ideas", []),
                relevance_score=s.get("relevance_score", 0.0),
                raw_text=s.get("raw_text", ""),
            )
            for s in data.get("sources", [])
        ]
        return cls(competition_slug=data["competition_slug"], sources=sources)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_sources(
    metadata: CompetitionMetadata,
    artifacts_dir: Path,
    ai_provider: AiProvider | None = None,
    force_refresh: bool = False,
) -> IdeaCollection:
    """Collect ideas from all available sources.

    Loads from cache if present and not force_refresh.
    """
    if not force_refresh:
        cached = IdeaCollection.load(artifacts_dir)
        if cached is not None:
            return cached

    collection = IdeaCollection(competition_slug=metadata.slug)

    # --- Kaggle kernels -------------------------------------------------------
    kernel_sources = _fetch_kernel_ideas(metadata.slug, ai_provider)
    collection.sources.extend(kernel_sources)

    # --- arXiv ----------------------------------------------------------------
    arxiv_sources = _fetch_arxiv_ideas(metadata, ai_provider)
    collection.sources.extend(arxiv_sources)

    # Sort by relevance descending
    collection.sources.sort(key=lambda s: s.relevance_score, reverse=True)

    collection.save(artifacts_dir)
    return collection


# ---------------------------------------------------------------------------
# Kernel idea extraction
# ---------------------------------------------------------------------------


_KERNEL_EXTRACTION_SYSTEM = """\
You are a Kaggle expert. Given a list of top public kernel titles for a competition,
extract 3-6 distinct, concrete technique ideas that appear to be working well.
Return ONLY a JSON array of strings — no prose, no explanation.
Example: ["LightGBM + CatBoost blend", "Target encoding for high-cardinality columns",
          "5-fold stratified CV", "Feature: lag features for time-series data"]
"""


def _fetch_kernel_ideas(
    slug: str,
    ai_provider: AiProvider | None,
) -> list[IdeaSource]:
    """Pull top kernel titles and extract ideas from them."""
    from researchforge.kaggle.api import KaggleAPIError, list_competition_kernels

    try:
        kernels = list_competition_kernels(slug, sort_by="scoreDescending", page_size=20)
    except KaggleAPIError:
        return []

    if not kernels:
        return []

    titles_text = "\n".join(
        f"- {k.title} (by {k.author}, votes: {k.total_votes})" for k in kernels[:15]
    )

    ideas: list[str] = []
    if ai_provider is not None:
        try:
            user_prompt = f"Competition: {slug}\n\nTop kernel titles:\n{titles_text}"
            raw = ai_provider.generate(_KERNEL_EXTRACTION_SYSTEM, user_prompt, max_tokens=512)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            ideas = json.loads(raw)
            if not isinstance(ideas, list):
                ideas = []
        except (json.JSONDecodeError, Exception):
            ideas = []

    # Fallback: use kernel titles directly as ideas
    if not ideas:
        ideas = [k.title for k in kernels[:6] if k.title]

    return [
        IdeaSource(
            source_type=IdeaSourceType.KAGGLE_KERNEL,
            title=f"Top Kaggle kernels for {slug}",
            url=f"https://www.kaggle.com/competitions/{slug}/code",
            ideas=ideas,
            relevance_score=0.9,
            raw_text=titles_text[:500],
        )
    ]


# ---------------------------------------------------------------------------
# arXiv idea extraction
# ---------------------------------------------------------------------------


_ARXIV_QUERY_SYSTEM = """\
You are a research librarian. Given a Kaggle competition's task type and metric,
generate a precise arXiv search query (5-10 words) that would find the most useful
recent papers for improving a model. Return ONLY the query string, nothing else.
"""

_ARXIV_EXTRACT_SYSTEM = """\
You are a machine learning researcher. Given arXiv paper titles and abstracts,
extract 3-6 concrete technique ideas that could improve a Kaggle competition model.
Return ONLY a JSON array of strings (technique names/ideas), no prose.
"""


def _fetch_arxiv_ideas(
    metadata: CompetitionMetadata,
    ai_provider: AiProvider | None,
) -> list[IdeaSource]:
    """Search arXiv for relevant papers and extract ideas."""
    try:
        from researchforge.research.arxiv import (  # type: ignore[import-not-found]
            search as arxiv_search,
        )
    except ImportError:
        return _arxiv_via_api(metadata, ai_provider)

    # Generate query from competition metadata
    if ai_provider is not None:
        try:
            query = (
                ai_provider.generate(
                    _ARXIV_QUERY_SYSTEM,
                    f"Task: {metadata.task_type}, Metric: {metadata.metric_name}, Slug: "
                    f"{metadata.slug}",
                    max_tokens=64,
                )
                .strip()
                .strip('"')
                .strip("'")
            )
        except Exception:
            query = f"{metadata.task_type} {metadata.metric_name}"
    else:
        query = f"{metadata.task_type} machine learning"

    try:
        papers = arxiv_search(query, max_results=10)
    except Exception:
        return _arxiv_via_api(metadata, ai_provider)

    if not papers:
        return []

    papers_text = "\n".join(
        f"- {p.get('title', '')} | {p.get('summary', '')[:200]}" for p in papers[:8]
    )

    ideas: list[str] = []
    if ai_provider is not None:
        try:
            raw = ai_provider.generate(_ARXIV_EXTRACT_SYSTEM, papers_text, max_tokens=512)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            ideas = json.loads(raw)
            if not isinstance(ideas, list):
                ideas = []
        except Exception:
            ideas = []

    if not ideas:
        ideas = [p.get("title", "") for p in papers[:5] if p.get("title")]

    paper_urls = [p.get("url", p.get("id", "")) for p in papers[:3]]

    return [
        IdeaSource(
            source_type=IdeaSourceType.ARXIV,
            title=f"arXiv: {query}",
            url=paper_urls[0] if paper_urls else "https://arxiv.org/search/",
            ideas=ideas,
            relevance_score=0.7,
            raw_text=papers_text[:500],
        )
    ]


def _arxiv_via_api(
    metadata: CompetitionMetadata,
    ai_provider: AiProvider | None,
) -> list[IdeaSource]:
    """Minimal arXiv search via the public API (no SDK dependency)."""
    import urllib.error
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    query = f"{metadata.task_type} kaggle competition"
    encoded = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{encoded}&max_results=8&sortBy=relevance"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchForge/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return []

    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns)[:8]:
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            papers.append(
                {
                    "title": (title_el.text or "").strip() if title_el is not None else "",
                    "summary": (summary_el.text or "").strip()[:200]
                    if summary_el is not None
                    else "",
                    "url": (id_el.text or "").strip() if id_el is not None else "",
                }
            )
    except ET.ParseError:
        return []

    if not papers:
        return []

    papers_text = "\n".join(f"- {p['title']} | {p['summary']}" for p in papers)

    ideas: list[str] = []
    if ai_provider is not None:
        try:
            raw = ai_provider.generate(_ARXIV_EXTRACT_SYSTEM, papers_text, max_tokens=512)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            ideas = json.loads(raw)
            if not isinstance(ideas, list):
                ideas = []
        except Exception:
            ideas = []

    if not ideas:
        ideas = [p["title"] for p in papers[:5] if p["title"]]

    return [
        IdeaSource(
            source_type=IdeaSourceType.ARXIV,
            title=f"arXiv: {query}",
            url=papers[0]["url"] if papers else "https://arxiv.org",
            ideas=ideas,
            relevance_score=0.7,
            raw_text=papers_text[:500],
        )
    ]
