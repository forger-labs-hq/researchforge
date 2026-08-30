from datetime import date
from pathlib import Path

import httpx
import pytest

from researchforge.research.arxiv_client import (
    ARXIV_DATE_CEILING,
    ArxivClient,
    ArxivError,
    build_search_query,
    parse_atom_feed,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "arxiv"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestParseAtomFeed:
    def test_parses_entries_and_total(self) -> None:
        entries, total = parse_atom_feed(_fixture("search_page1.xml"))

        assert total == 5
        assert len(entries) == 3
        first = entries[0]
        assert first.arxiv_id == "2401.12345"
        assert first.version == 2
        assert first.paper_id == "arxiv:2401.12345"
        assert first.title == "Uncertainty-Aware Routing for Cost-Efficient LLM Inference"
        assert "predictive entropy" in first.abstract
        assert first.authors == ["Alice Author", "Bob Builder"]
        assert first.primary_category == "cs.LG"
        assert first.categories == ["cs.LG", "cs.AI"]
        assert first.pdf_url == "http://arxiv.org/pdf/2401.12345v2"
        assert first.published_at.year == 2024

    def test_whitespace_normalized_in_title_and_abstract(self) -> None:
        entries, _ = parse_atom_feed(_fixture("search_page1.xml"))
        assert "\n" not in entries[0].title
        assert "  " not in entries[0].abstract

    def test_empty_feed(self) -> None:
        entries, total = parse_atom_feed(_fixture("search_empty.xml"))
        assert entries == []
        assert total == 0

    def test_malformed_feed_raises(self) -> None:
        with pytest.raises(ArxivError):
            parse_atom_feed(_fixture("malformed.xml"))


def _client_with(handler: httpx.MockTransport, sleeps: list[float] | None = None) -> ArxivClient:
    recorded = sleeps if sleeps is not None else []
    return ArxivClient(
        client=httpx.Client(transport=handler),
        sleep=recorded.append,
    )


class TestBuildSearchQuery:
    def test_an_unfiltered_query_is_passed_through_untouched(self) -> None:
        assert build_search_query("all:routing", [], None) == "all:routing"

    def test_categories_are_joined_with_or(self) -> None:
        query = build_search_query("all:routing", ["cs.LG", "cs.AI"], None)
        assert query == "(all:routing) AND (cat:cs.LG OR cat:cs.AI)"

    def test_a_since_date_becomes_an_arxiv_date_range(self) -> None:
        query = build_search_query("all:routing", [], date(2024, 6, 1))
        assert query == f"(all:routing) AND submittedDate:[202406010000 TO {ARXIV_DATE_CEILING}]"

    def test_categories_and_a_date_apply_together(self) -> None:
        query = build_search_query("all:routing", ["cs.LG"], date(2025, 1, 31))
        assert "cat:cs.LG" in query
        assert "submittedDate:[202501310000" in query
        assert query.count(" AND ") == 2

    def test_the_date_starts_at_midnight_so_the_day_itself_is_included(self) -> None:
        query = build_search_query("all:routing", [], date(2024, 6, 1))
        assert "202406010000" in query


class TestArxivClientSearch:
    def test_single_page_search(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, text=_fixture("search_page1.xml"))

        client = _client_with(httpx.MockTransport(handler))
        entries = client.search("all:routing", max_results=3, page_size=3)

        assert len(entries) == 3
        assert len(requests) == 1
        assert requests[0].url.params["search_query"] == "all:routing"
        assert requests[0].url.params["start"] == "0"

    def test_paging_issues_correct_start_offsets(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            page = "search_page1.xml" if request.url.params["start"] == "0" else "search_page2.xml"
            return httpx.Response(200, text=_fixture(page))

        client = _client_with(httpx.MockTransport(handler))
        entries = client.search("all:routing", max_results=5, page_size=3)

        assert len(entries) == 5
        assert [r.url.params["start"] for r in requests] == ["0", "3"]

    def test_rate_limit_sleeps_between_requests_not_before_first(self) -> None:
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            page = "search_page1.xml" if request.url.params["start"] == "0" else "search_page2.xml"
            return httpx.Response(200, text=_fixture(page))

        client = _client_with(httpx.MockTransport(handler), sleeps)
        client.search("all:routing", max_results=5, page_size=3)

        assert len(sleeps) == 1  # only between the two requests
        assert sleeps[0] > 0

    def test_retries_on_server_error_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, text=_fixture("search_page1.xml"))

        client = _client_with(httpx.MockTransport(handler))
        entries = client.search("all:routing", max_results=3, page_size=3)

        assert len(entries) == 3
        assert calls["n"] == 2

    def test_client_error_raises_without_retry(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400)

        client = _client_with(httpx.MockTransport(handler))
        with pytest.raises(ArxivError, match="HTTP 400"):
            client.search("bad query", max_results=3)
        assert calls["n"] == 1

    def test_exhausted_retries_raise(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        client = _client_with(httpx.MockTransport(handler))
        with pytest.raises(ArxivError, match="after 3 attempts"):
            client.search("all:routing", max_results=3)

    def test_since_filters_at_the_api_not_after_fetching(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, text=_fixture("search_page1.xml"))

        client = ArxivClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep=lambda s: None,
            submitted_since=date(2024, 6, 1),
        )
        client.search("all:routing", max_results=3, page_size=3)
        assert "submittedDate:[202406010000 TO" in requests[0].url.params["search_query"]

    def test_user_agent_header_sent(self) -> None:
        # The default client carries the UA; with an injected client the
        # header comes from the request default. Build the real default
        # client config but swap the transport via mounting is complex;
        # instead verify the constant on a default instance.
        client = ArxivClient()
        assert "researchforge/" in client._client.headers["User-Agent"]
