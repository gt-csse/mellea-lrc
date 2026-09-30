"""GovInfo docket fallback keeps raw search evidence and numeric shortlists."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable

import pytest

from mellea_lrc.api import Document, grow_roots
from mellea_lrc.courtlistener.models import CourtListenerSearchPage
from mellea_lrc.govinfo import GovInfoError, GovInfoSearchPage
from mellea_lrc.model import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookupCaseNameAssessment,
    DocketLookupFieldAssessment,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.fields.case_name import CaseName
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.span import Span
from mellea_lrc.validation.docket_root_lookup_courtlistener_retrieval import (
    docket_root_lookup_courtlistener_retrieval,
)
from mellea_lrc.validation.docket_root_lookup_courtlistener_llm_review import (
    docket_root_lookup_courtlistener_llm_review,
)

lookup_module = importlib.import_module("mellea_lrc.validation.docket_root_lookup_govinfo_retrieval")


class FakeGovInfoClient:
    def __init__(self, respond: Callable[[str, str, int], GovInfoSearchPage]) -> None:
        self.respond = respond
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, *, offset_mark: str = "*", page_size: int = 100) -> GovInfoSearchPage:
        self.calls.append((query, offset_mark, page_size))
        return self.respond(query, offset_mark, page_size)


def _page(
    *results: dict[str, object], count: int | None = None, next_mark: str | None = None
) -> GovInfoSearchPage:
    raw = {
        "count": len(results) if count is None else count,
        "offsetMark": next_mark,
        "results": list(results),
        "uninterpreted": {"preserve": True},
    }
    return GovInfoSearchPage(
        raw_json=raw, results=tuple(results), count=raw["count"], next_offset_mark=next_mark
    )


def _empty_cl_page() -> CourtListenerSearchPage:
    return CourtListenerSearchPage.model_validate({"count": 0, "next": None, "previous": None, "results": []})


def _before_govinfo(source: str = "Acme v. Reed, Case No. 2:31-cv-45821 (D. Mass. 2031).") -> Document:
    rooted = asyncio.run(grow_roots(Document.from_source(source)))
    searched = docket_root_lookup_courtlistener_retrieval(
        rooted,
        client=type("EmptyCL", (), {"search": lambda *_args, **_kw: _empty_cl_page()})(),
    )
    return asyncio.run(docket_root_lookup_courtlistener_llm_review(searched))


def test_raw_pages_and_package_ids_drive_shortlist_only_by_number() -> None:
    before = _before_govinfo()
    query = 'collection:uscourts casenumber:("2:31-cv-45821")'
    client = FakeGovInfoClient(
        lambda *_: _page(
            {
                "packageId": "USCOURTS-mad-2_31-cv-45821",
                "granuleId": "USCOURTS-mad-2_31-cv-45821-0",
                "title": "Unrelated title",
                "dateIssued": "1900-01-01",
                "unknown": {"retain": [1]},
            },
            {"packageId": "not-a-valid-id", "granuleId": "g1"},
        )
    )

    after = lookup_module.docket_root_lookup_govinfo_retrieval(before, client=client)

    assert client.calls == [(query, "*", 100)]
    assert after.stage_runs[-1] == lookup_module.STAGE
    root = after.roots[0]
    assert isinstance(root, FullDocketCitation)
    lookup = root.govinfo_docket_lookup
    assert lookup is not None
    assert lookup.node_id == root.nodes[-1].id
    assert lookup.attempts[0].query == query
    assert lookup.attempts[0].pages[0]["uninterpreted"] == {"preserve": True}
    assert lookup.attempts[0].pages[0]["results"][0]["unknown"] == {"retain": [1]}
    assert [
        (item.package_id, item.granule_id, item.court_code, item.docket_number) for item in lookup.candidates
    ] == [
        ("USCOURTS-mad-2_31-cv-45821", "USCOURTS-mad-2_31-cv-45821-0", "mad", "2:31-cv-45821"),
        ("not-a-valid-id", "g1", None, None),
    ]
    assert lookup.shortlisted_candidate_indices == (0,)
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_compact_year_sequence_adds_a_search_only_spelling() -> None:
    source = "Acme v. Reed, No. 2210920 (Bankr. S.D.N.Y. 2024)."
    before = Document.from_source(source)
    locator_start = before.text.index("No. 2210920")
    locator_end = locator_start + len("No. 2210920")
    number_start = before.text.index("2210920")
    citation = FullDocketCitation.from_locator(
        citation_id="docket:compact",
        stage="2_docket_locators",
        source=before.text,
        span=Span(locator_start, locator_end),
        number_span=Span(number_start, number_start + 7),
    )
    before = before.add_citation(citation).complete("2_docket_locators")
    before = before.replace_citation(citation.record("10_roots").with_root(citation.id))
    before = (
        before.complete("10_roots")
        .complete("16_docket_root_lookup_courtlistener_retrieval")
        .complete("17_docket_root_lookup_courtlistener_llm_review")
    )
    client = FakeGovInfoClient(
        lambda query, *_: (
            _page({"packageId": "USCOURTS-nysb-1_22-bk-10920", "title": "Acme v. Reed"})
            if 'casenumber:("22-10920")' in query
            else _page()
        )
    )

    after = lookup_module.docket_root_lookup_govinfo_retrieval(before, client=client)
    lookup = after.roots[0].govinfo_docket_lookup
    assert lookup is not None
    assert [call[0] for call in client.calls] == [
        'collection:uscourts casenumber:("2210920")',
        'collection:uscourts casenumber:("22-10920")',
    ]
    assert len(lookup.attempts) == 2
    assert lookup.candidates[0].attempt_index == 1
    assert lookup.shortlisted_candidate_indices == (0,)
    assert after.roots[0].locator == before.roots[0].locator


def test_stops_when_reported_count_is_reached_even_with_offset_mark() -> None:
    before = _before_govinfo("Case No. 24-cv-123.")
    client = FakeGovInfoClient(
        lambda *_: _page(
            {"packageId": "USCOURTS-mad-24-cv-123"},
            count=1,
            next_mark="next",
        )
    )

    after = lookup_module.docket_root_lookup_govinfo_retrieval(before, client=client)
    lookup = after.roots[0].govinfo_docket_lookup
    assert lookup is not None
    assert len(client.calls) == 1
    assert lookup.attempts[0].count == 1
    assert lookup.attempts[0].next_offset_marks == ("next",)
    assert lookup.attempts[0].failure is None


def test_paginates_and_records_terminal_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _before_govinfo("Case No. 24-cv-123.")
    delays: list[float] = []
    monkeypatch.setattr(lookup_module.time, "sleep", delays.append)

    def respond(_query: str, mark: str, _size: int) -> GovInfoSearchPage:
        if mark == "*":
            return _page({"packageId": "USCOURTS-mad-24-cv-123"}, count=2, next_mark="page-2")
        raise GovInfoError(
            "GovInfo unavailable",
            failure_type="http_error",
            upstream_status_code=400,
            url="https://api.govinfo.gov/search",
            upstream_detail={"reason": "maintenance"},
        )

    client = FakeGovInfoClient(respond)
    after = lookup_module.docket_root_lookup_govinfo_retrieval(before, client=client)
    lookup = after.roots[0].govinfo_docket_lookup
    assert lookup is not None
    attempt = lookup.attempts[0]
    assert len(attempt.pages) == 1
    assert attempt.failure is not None
    assert (attempt.failure.failure_type, attempt.failure.upstream_status_code) == ("http_error", 400)
    assert attempt.failure.upstream_detail == {"reason": "maintenance"}
    assert len(attempt.retry_failures) == 0
    assert delays == []
    assert lookup.candidates[0].docket_number == "24-cv-123"


def test_transport_retry_is_saved_and_successful_search_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _before_govinfo("Case No. 24-cv-123.")
    delays: list[float] = []
    monkeypatch.setattr(lookup_module.time, "sleep", delays.append)
    failures = 0

    def respond(*_args: object) -> GovInfoSearchPage:
        nonlocal failures
        if failures == 0:
            failures += 1
            raise GovInfoError("network", failure_type="transport_error")
        return _page({"packageId": "USCOURTS-mad-24-cv-123"})

    after = lookup_module.docket_root_lookup_govinfo_retrieval(before, client=FakeGovInfoClient(respond))
    attempt = after.roots[0].govinfo_docket_lookup.attempts[0]
    assert len(attempt.retry_failures) == 1
    assert attempt.failure is None
    assert delays == [1]


def test_server_errors_retry_and_keep_retry_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _before_govinfo("Case No. 24-cv-123.")
    delays: list[float] = []
    monkeypatch.setattr(lookup_module.time, "sleep", delays.append)
    calls = 0

    def respond(*_args: object) -> GovInfoSearchPage:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise GovInfoError("server error", failure_type="http_error", upstream_status_code=503)
        return _page({"packageId": "USCOURTS-mad-24-cv-123"})

    after = lookup_module.docket_root_lookup_govinfo_retrieval(before, client=FakeGovInfoClient(respond))
    attempt = after.roots[0].govinfo_docket_lookup.attempts[0]
    assert len(attempt.retry_failures) == 2
    assert all(failure.upstream_status_code == 503 for failure in attempt.retry_failures)
    assert attempt.failure is None
    assert delays == [1, 2]


def test_selected_courtlistener_review_skips_provider_but_completes_stage() -> None:
    source = "Acme v. Reed, Case No. 24-cv-123 (D. Mass. 2031)."
    rooted = asyncio.run(grow_roots(Document.from_source(source)))

    class FakeCL:
        def search(self, *_args: object, **_kwargs: object) -> CourtListenerSearchPage:
            return CourtListenerSearchPage.model_validate(
                {
                    "count": 1,
                    "next": None,
                    "previous": None,
                    "results": [
                        {
                            "docket_id": 7,
                            "docketNumber": "24-cv-123",
                            "caseName": "Acme v. Reed",
                            "court": "D. Mass.",
                            "court_id": "mad",
                            "dateFiled": "2031-01-01",
                        }
                    ],
                }
            )

    searched = docket_root_lookup_courtlistener_retrieval(rooted, client=FakeCL())
    case_name = CaseName.from_quote("Acme v. Reed")

    def assessment() -> DocketLookupFieldAssessment:
        return DocketLookupFieldAssessment(
            propose_replacement=False,
            quote=None,
            result=MatchResult.MATCH,
            reason="Both sides support a match.",
        )

    decision = DocketLookupReviewDecision(
        selected_candidate_index=0,
        docket_number=assessment(),
        case_name=DocketLookupCaseNameAssessment(
            propose_replacement=False,
            quote=None,
            normalized=case_name,
            result=MatchResult.MATCH,
            reason="Both sides support a match.",
        ),
        court=assessment(),
        date=assessment(),
        reason="The record identifies the cited case.",
    )

    async def reviewer(_context: object) -> DocketLookupReviewDecision:
        return decision

    reviewed = asyncio.run(docket_root_lookup_courtlistener_llm_review(searched, reviewer=reviewer))
    root = reviewed.roots[0]
    assert isinstance(root, FullDocketCitation)
    assert root.docket_lookup_review.decision.selected_candidate_index == 0
    client = FakeGovInfoClient(lambda *_: pytest.fail("GovInfo must not be called for a selected CL case"))

    after = lookup_module.docket_root_lookup_govinfo_retrieval(reviewed, client=client)
    assert after.stage_runs[-1] == lookup_module.STAGE
    assert after.roots[0].govinfo_docket_lookup is None
    assert client.calls == []


def test_rejects_repeated_stage_and_requires_prior_review() -> None:
    with pytest.raises(ValueError, match="Complete CourtListener"):
        lookup_module.docket_root_lookup_govinfo_retrieval(
            asyncio.run(grow_roots(Document.from_source("Case No. 24-cv-123.")))
        )
    after = lookup_module.docket_root_lookup_govinfo_retrieval(
        _before_govinfo(), client=FakeGovInfoClient(lambda *_: _page())
    )
    with pytest.raises(ValueError, match="already completed"):
        lookup_module.docket_root_lookup_govinfo_retrieval(
            after, client=FakeGovInfoClient(lambda *_: _page())
        )
