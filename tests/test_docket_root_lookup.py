"""Docket lookup retains raw searches and makes only a number-based shortlist."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from typing import Literal

import pytest

from mellea_lrc.api import Document, grow_roots
from mellea_lrc.courtlistener import CourtListenerHTTPError
from mellea_lrc.courtlistener.models import CourtListenerSearchPage
from mellea_lrc.model import FullDocketCitation

lookup_module = importlib.import_module("mellea_lrc.validation.docket_root_lookup")


class FakeSearchClient:
    def __init__(
        self,
        respond: Callable[[str, Literal["d", "o"], str | None], CourtListenerSearchPage],
    ) -> None:
        self.respond = respond
        self.calls: list[tuple[str, Literal["d", "o"], str | None]] = []

    def search(
        self, q: str, search_type: Literal["d", "o"], cursor: str | None = None
    ) -> CourtListenerSearchPage:
        self.calls.append((q, search_type, cursor))
        return self.respond(q, search_type, cursor)


def _rooted(source: str = "Acme v. Reed, Case No. 2:31-cv-45821 (D. Mass. 2031).") -> Document:
    return asyncio.run(grow_roots(Document.from_source(source)))


def _page(
    *results: dict[str, object], next_url: str | None = None, marker: str = "upstream"
) -> CourtListenerSearchPage:
    return CourtListenerSearchPage.model_validate(
        {
            "count": len(results),
            "next": next_url,
            "previous": None,
            "results": list(results),
            "uninterpreted_page_field": {"marker": marker},
        }
    )


def test_raw_and_numeric_queries_keep_every_hit_but_shortlist_by_number_only() -> None:
    before = _rooted()
    assert len(before.roots) == 1
    full = r"docketNumber:(2\:31\-cv\-45821)"
    broad = "docketNumber:(31 45821)"
    payloads = {
        (full, "d"): _page(
            {
                "docket_id": 101,
                "docketNumber": "2:31-cv-45821",
                "caseName": "Unrelated Parties",
                "court_id": "scotus",
                "dateFiled": "1999-01-01",
                "unknown_result_field": {"retain": [1, 2]},
            },
            {"docket_id": 102, "docketNumber": "88-ccc-9999"},
            marker="full-docket",
        ),
        (full, "o"): _page({"cluster_id": 301, "docketNumber": "2:31-cv-45821"}),
        (broad, "d"): _page(
            {"docket_id": 101, "docketNumber": "2:31-cv-45821"},
            {"docket_id": 103, "docketNumber": "31-45821"},
        ),
        (broad, "o"): _page({"cluster_id": 301, "docketNumber": "2:31-cv-45821"}),
    }
    client = FakeSearchClient(lambda q, kind, _cursor: payloads[(q, kind)])

    after = lookup_module.docket_root_lookup(before, client=client)

    assert client.calls == [(full, "d", None), (full, "o", None), (broad, "d", None), (broad, "o", None)]
    assert after.stage_runs == (*before.stage_runs, lookup_module.STAGE)
    assert after.get_stage("10_roots") == before
    assert after.get_stage(lookup_module.STAGE) == after
    root = after.roots[0]
    assert isinstance(root, FullDocketCitation)
    lookup = root.docket_lookup
    assert lookup is not None
    assert lookup.node_id == root.nodes[-1].id
    assert [(attempt.query, attempt.source_type) for attempt in lookup.attempts] == [
        (full, "d"),
        (full, "o"),
        (broad, "d"),
        (broad, "o"),
    ]
    assert lookup.attempts[0].pages[0]["uninterpreted_page_field"] == {"marker": "full-docket"}
    assert lookup.attempts[0].pages[0]["results"][0]["unknown_result_field"] == {"retain": [1, 2]}
    assert len(lookup.candidates) == 6
    assert [(item.attempt_index, item.page_index, item.result_index) for item in lookup.candidates] == [
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 0),
        (2, 0, 0),
        (2, 0, 1),
        (3, 0, 0),
    ]
    assert [item.record_id for item in lookup.candidates] == ["101", "102", "301", "101", "103", "301"]
    assert lookup.candidates[1].docket_similarity < lookup_module.MINIMUM_SIMILARITY_PERCENT
    assert lookup.candidates[4].docket_similarity >= lookup_module.MINIMUM_SIMILARITY_PERCENT
    assert lookup.shortlisted_candidate_indices == (0, 2, 4)
    assert root.identity_judgments == ()
    assert root.case_name_judgments == root.court_judgments == root.date_judgments == ()
    assert Document.model_validate_json(after.model_dump_json()) == after

    with pytest.raises(ValueError, match="already completed"):
        lookup_module.docket_root_lookup(after, client=client)


def test_forty_percent_boundary_and_hits_without_ids() -> None:
    before = _rooted("Case No. 12-ab-3456.")
    query = r"docketNumber:(12\-ab\-3456)"
    client = FakeSearchClient(
        lambda q, kind, _cursor: (
            _page(
                {"docketNumber": "17-bb-6915"},
                {"docketNumber": "17-bb-6915"},
                {"docketNumber": "65-yb-4943"},
            )
            if (q, kind) == (query, "d")
            else _page()
        )
    )

    after = lookup_module.docket_root_lookup(before, client=client)
    lookup = after.roots[0].docket_lookup
    assert lookup is not None
    assert [candidate.docket_similarity for candidate in lookup.candidates] == pytest.approx(
        [40.0, 40.0, 30.0]
    )
    assert [candidate.record_id for candidate in lookup.candidates] == [None, None, None]
    assert lookup.shortlisted_candidate_indices == (0, 1)


def test_every_distinct_docket_root_receives_its_own_lookup() -> None:
    before = _rooted("Case No. 24-cv-123. Case No. 24-cv-124.")
    assert len(before.roots) == 2
    client = FakeSearchClient(lambda *_args: _page())

    after = lookup_module.docket_root_lookup(before, client=client)

    assert len(client.calls) == 8
    assert len(after.roots) == 2
    assert all(root.docket_lookup is not None for root in after.roots)
    assert [root.docket_lookup.node_id for root in after.roots] == [root.nodes[-1].id for root in after.roots]


def test_partial_page_failure_is_saved_and_other_queries_continue() -> None:
    first_query = r"docketNumber:(24\-cv\-123)"
    next_url = "https://www.courtlistener.com/api/rest/v4/search/?cursor=abc%2B%2F%3D"
    first_page = _page({"docket_id": 1, "docketNumber": "24-cv-123"}, next_url=next_url)

    def respond(q: str, kind: Literal["d", "o"], cursor: str | None) -> CourtListenerSearchPage:
        if (q, kind, cursor) == (first_query, "d", None):
            return first_page
        if (q, kind, cursor) == (first_query, "d", "abc+/="):
            raise CourtListenerHTTPError(
                "CourtListener search returned HTTP 429",
                failure_type="http_error",
                upstream_status_code=429,
                url="https://proxy.example/search/?cursor=abc",
                upstream_detail=[{"loc": ("results", 0), "message": "try later"}],
            )
        return _page()

    client = FakeSearchClient(respond)
    after = lookup_module.docket_root_lookup(_rooted("Case No. 24-cv-123."), client=client)
    lookup = after.roots[0].docket_lookup
    assert lookup is not None
    assert client.calls[:2] == [(first_query, "d", None), (first_query, "d", "abc+/=")]
    assert len(lookup.attempts) == 4
    assert lookup.attempts[0].pages == (first_page.raw_json,)
    failure = lookup.attempts[0].failure
    assert failure is not None
    assert (failure.failure_type, failure.upstream_status_code) == ("http_error", 429)
    assert failure.upstream_detail == [{"loc": ["results", 0], "message": "try later"}]
    assert all(attempt.failure is None for attempt in lookup.attempts[1:])
    assert lookup.candidates[0].record_id == "1"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_proxy_rate_limit_retries_same_page_and_retains_failure_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lookup_module.time, "sleep", lambda _seconds: None)
    query = r"docketNumber:(24\-cv\-123)"
    failures = 0

    def respond(q: str, kind: Literal["d", "o"], cursor: str | None) -> CourtListenerSearchPage:
        nonlocal failures
        if (q, kind, cursor) == (query, "d", None) and failures == 0:
            failures += 1
            raise CourtListenerHTTPError(
                "CourtListener search returned HTTP 429",
                failure_type="http_error",
                upstream_status_code=429,
                url="https://proxy.example/search/",
                upstream_detail='{"detail":"all tokens exhausted","retry_after_seconds":0}',
            )
        return _page({"docket_id": 1, "docketNumber": "24-cv-123"}) if kind == "d" else _page()

    client = FakeSearchClient(respond)
    after = lookup_module.docket_root_lookup(_rooted("Case No. 24-cv-123."), client=client)
    lookup = after.roots[0].docket_lookup
    assert lookup is not None
    assert client.calls[:2] == [(query, "d", None), (query, "d", None)]
    assert len(lookup.attempts[0].retry_failures) == 1
    assert lookup.attempts[0].retry_failures[0].upstream_status_code == 429
    assert lookup.attempts[0].failure is None
    assert lookup.candidates[0].record_id == "1"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_page_budget_records_truncation_and_keeps_last_next_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lookup_module, "MAX_PAGES_PER_ATTEMPT", 2)
    query = r"docketNumber:(24\-cv\-123)"
    pages = {
        None: _page(
            {"docket_id": 1, "docketNumber": "24-cv-123"},
            next_url="https://www.courtlistener.com/api/rest/v4/search/?cursor=one",
        ),
        "one": _page(
            {"docket_id": 2, "docketNumber": "24-cv-124"},
            next_url="https://www.courtlistener.com/api/rest/v4/search/?cursor=two",
        ),
    }

    def respond(q: str, kind: Literal["d", "o"], cursor: str | None) -> CourtListenerSearchPage:
        return pages[cursor] if (q, kind) == (query, "d") else _page()

    client = FakeSearchClient(respond)
    after = lookup_module.docket_root_lookup(_rooted("Case No. 24-cv-123."), client=client)
    lookup = after.roots[0].docket_lookup
    assert lookup is not None
    assert client.calls[:2] == [(query, "d", None), (query, "d", "one")]
    assert (query, "d", "two") not in client.calls
    assert len(lookup.attempts[0].pages) == 2
    assert lookup.attempts[0].pages[1]["next"] == pages["one"].next
    assert lookup.attempts[0].failure is not None
    assert lookup.attempts[0].failure.failure_type == "page_limit_reached"
    assert lookup.attempts[0].failure.url == pages["one"].next
    assert [(item.page_index, item.result_index) for item in lookup.candidates[:2]] == [(0, 0), (1, 0)]


def test_no_docket_roots_completes_without_calling_search() -> None:
    before = _rooted("Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007).")
    client = FakeSearchClient(lambda *_args: pytest.fail("No docket root should be searched"))

    after = lookup_module.docket_root_lookup(before, client=client)

    assert client.calls == []
    assert after.stage_runs == (*before.stage_runs, lookup_module.STAGE)
    assert after.roots == before.roots
