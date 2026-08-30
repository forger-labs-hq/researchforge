"""arXiv API client (Atom feed over HTTPS).

Politeness per arXiv guidance: at least 3 seconds between requests, an
identifying User-Agent, and modest page sizes. Responses are metadata and
abstracts only — paper text is never downloaded or redistributed.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import date, datetime

import httpx
from pydantic import BaseModel

from researchforge import __version__

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
_ABS_URL_PATTERN = re.compile(r"/abs/(?P<id>\d{4}\.\d{4,5})(?:v(?P<version>\d+))?$")
_WHITESPACE = re.compile(r"\s+")

MAX_RESPONSE_BYTES = 10_000_000


class ArxivError(Exception):
    """Retrieval or parsing failure, with the offending query in the message."""


class ArxivEntry(BaseModel):
    """A raw retrieval record, before ranking/selection."""

    arxiv_id: str  # canonical, version stripped: "2401.12345"
    version: int | None = None
    title: str
    abstract: str
    authors: list[str]
    published_at: datetime
    updated_at: datetime | None = None
    categories: list[str]
    primary_category: str | None = None
    source_url: str
    pdf_url: str | None = None

    @property
    def paper_id(self) -> str:
        return f"arxiv:{self.arxiv_id}"


def _clean(text: str | None) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def _parse_entry(entry: ET.Element) -> ArxivEntry | None:
    raw_id = _clean(entry.findtext("atom:id", namespaces=_NS))
    match = _ABS_URL_PATTERN.search(raw_id)
    if match is None:
        return None  # e.g. legacy pre-2007 ids; out of scope for Phase 1A

    published_raw = entry.findtext("atom:published", namespaces=_NS)
    if not published_raw:
        return None
    updated_raw = entry.findtext("atom:updated", namespaces=_NS)

    pdf_url = None
    for link in entry.findall("atom:link", namespaces=_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
            break

    primary = entry.find("arxiv:primary_category", namespaces=_NS)

    return ArxivEntry(
        arxiv_id=match.group("id"),
        version=int(match.group("version")) if match.group("version") else None,
        title=_clean(entry.findtext("atom:title", namespaces=_NS)),
        abstract=_clean(entry.findtext("atom:summary", namespaces=_NS)),
        authors=[
            _clean(name.text)
            for name in entry.findall("atom:author/atom:name", namespaces=_NS)
            if _clean(name.text)
        ],
        published_at=datetime.fromisoformat(published_raw.replace("Z", "+00:00")),
        updated_at=(
            datetime.fromisoformat(updated_raw.replace("Z", "+00:00")) if updated_raw else None
        ),
        categories=[
            term
            for cat in entry.findall("atom:category", namespaces=_NS)
            if (term := cat.get("term")) is not None
        ],
        primary_category=primary.get("term") if primary is not None else None,
        source_url=raw_id,
        pdf_url=pdf_url,
    )


def parse_atom_feed(xml_text: str) -> tuple[list[ArxivEntry], int | None]:
    """Parse an arXiv Atom feed into entries plus the reported total-result count."""
    try:
        root = ET.fromstring(xml_text)  # noqa: S314 — trusted origin, stdlib parser without DTD handling
    except ET.ParseError as exc:
        raise ArxivError(f"Unparseable Atom feed: {exc}") from exc

    total_raw = root.findtext("opensearch:totalResults", namespaces=_NS)
    total = int(total_raw) if total_raw and total_raw.isdigit() else None

    entries = []
    for element in root.findall("atom:entry", namespaces=_NS):
        parsed = _parse_entry(element)
        if parsed is not None:
            entries.append(parsed)
    return entries, total


ARXIV_DATE_CEILING = "999912312359"
"""An upper bound no submission can exceed. arXiv's date range needs both ends."""


def build_search_query(
    query: str, categories: list[str], submitted_since: date | None
) -> str:
    """The arXiv `search_query` for one search, with its filters applied.

    Filtering by date at the API rather than after fetching matters: the fetch
    is capped, so a post-fetch filter would spend that cap on papers it then
    discards and return fewer recent papers than asked for.
    """
    clauses = [f"({query})"]
    if categories:
        clauses.append("(" + " OR ".join(f"cat:{category}" for category in categories) + ")")
    if submitted_since is not None:
        start = submitted_since.strftime("%Y%m%d") + "0000"
        clauses.append(f"submittedDate:[{start} TO {ARXIV_DATE_CEILING}]")
    return " AND ".join(clauses) if len(clauses) > 1 else query


class ArxivClient:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        request_interval_s: float = 3.0,
        max_retries: int = 2,
        category_filter: list[str] | None = None,
        submitted_since: date | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": (
                    f"researchforge/{__version__} (https://github.com/forger-labs-hq/researchforge)"
                )
            },
        )
        self._sleep = sleep
        self._interval = request_interval_s
        self._max_retries = max_retries
        self._last_request_at: float | None = None
        self._category_filter = category_filter or []
        self._submitted_since = submitted_since

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._interval:
                self._sleep(self._interval - elapsed)
        self._last_request_at = time.monotonic()

    def _get(self, params: dict[str, str | int]) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                self._sleep(self._interval * attempt)
            self._throttle()
            try:
                response = self._client.get(self.BASE_URL, params=params)
            except httpx.TimeoutException as exc:
                last_error = exc
                continue
            if response.status_code >= 500:
                last_error = ArxivError(f"arXiv server error {response.status_code}")
                continue
            if response.status_code != 200:
                raise ArxivError(
                    f"arXiv request failed with HTTP {response.status_code} "
                    f"for query {params.get('search_query')!r}"
                )
            if len(response.content) > MAX_RESPONSE_BYTES:
                raise ArxivError("arXiv response exceeded the size limit.")
            return response.text
        raise ArxivError(
            f"arXiv request failed after {self._max_retries + 1} attempts "
            f"for query {params.get('search_query')!r}: {last_error}"
        )

    def search(self, query: str, *, max_results: int, page_size: int = 100) -> list[ArxivEntry]:
        """Fetch up to `max_results` entries for `query`, paging politely."""
        collected: list[ArxivEntry] = []
        total: int | None = None
        search_q = build_search_query(query, self._category_filter, self._submitted_since)
        while len(collected) < max_results:
            remaining = max_results - len(collected)
            request_size = min(page_size, remaining)
            text = self._get(
                {
                    "search_query": search_q,
                    "start": len(collected),
                    "max_results": request_size,
                    "sortBy": "relevance",
                }
            )
            entries, total = parse_atom_feed(text)
            collected.extend(entries)
            if len(entries) < request_size:
                break  # final page
            if total is not None and len(collected) >= total:
                break
        return collected[:max_results]
