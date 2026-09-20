"""Contract tests for bounded CourtListener and GovInfo docket discovery."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

import mellea_lrc.validation.docket_search as docket_search
from mellea_lrc.api import (
    form_roots,
    search_courtlistener_docket_roots,
    search_govinfo_docket_roots,
)
from mellea_lrc.core.citations import CitationDate, DocketCitation, placed
from mellea_lrc.core.record import CitationRecord, Node, Question, Reads
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerSearchResult
from mellea_lrc.extraction import Document, ExtractionMetadata
from mellea_lrc.extraction.stages import CASE_NAME_STAGE
from mellea_lrc.govinfo import GovInfoSearchResult
from mellea_lrc.preprocessing import preprocess


class _CourtListenerClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        assert search_type == "d"
        assert cursor is None
        self.calls.append(query)
        results = (
            [
                {
                    "docket_id": 77,
                    "docketNumber": "25-11282",
                    "caseName": "In re Example Holdings, Inc.",
                    "court_id": "nysb",
                }
            ]
            if query == 'caseName:("Example" AND "Holdings")'
            else []
        )
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type="d",
            semantic=False,
            count=len(results),
            results=results,
            next_cursor=None,
            previous_cursor=None,
        )


class _GovInfoClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search_uscourts(self, query: str, *, page_size: int) -> GovInfoSearchResult:
        assert page_size == 100
        self.calls.append(query)
        results = (
            [
                {
                    "packageId": "USCOURTS-nysb-25-11282",
                    "title": "In re Example Holdings, Inc.",
                }
            ]
            if query == 'collection:uscourts title:("Example" AND "Holdings")'
            else []
        )
        return GovInfoSearchResult(
            query=query, count=len(results), results=tuple(results), next_offset_mark=None
        )


@pytest.fixture
def document() -> Document:
    text = "In re Example Holdings, Inc., Case No. 1:25-bk-11282 (S.D.N.Y. 2025)."
    locator = "1:25-bk-11282"
    preprocessed = preprocess(text)
    citation = DocketCitation(
        docket_number=locator,
        court="nysd",
        date=CitationDate(year="2025"),
    )
    record = CitationRecord(
        citation_id="cite-0001",
        source=placed(
            citation,
            span=Span(0, len(text)),
            locator_span=Span(text.index(locator), text.index(locator) + len(locator)),
            matched_text=locator,
        ),
    )
    formed = form_roots(
        Document(
            source_metadata=preprocessed.source_metadata,
            text=text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(record,),
            extraction_metadata=ExtractionMetadata(),
        )
    )
    root = formed.citations[0]
    decision = Node(
        node_id="cite-0001:semantic-deferred",
        reads=Reads.RECORD,
        stage="docket_root_semantic_resolution",
        made_by="test",
        outcome="deferred",
    )
    root.judge(decision, Question.IDENTITY, "deferred_to_future_implementation")
    return replace(
        formed,
        passes=(*formed.passes, CASE_NAME_STAGE, "docket_root_semantic_resolution"),
    )


@pytest.fixture
def planned_terms(monkeypatch: pytest.MonkeyPatch):
    async def fixed_plan(record, *, stage: str, session):
        return docket_search._TermPlan(
            terms=("Example", "Holdings"),
            node=Node(
                node_id=f"{record.citation_id}:{stage}:case_name_terms",
                reads=Reads.DOCUMENT,
                stage=stage,
                made_by="test",
                outcome="prepared",
                details={"terms": ["Example", "Holdings"]},
            ),
        )

    monkeypatch.setattr(docket_search, "_prepare_terms", fixed_plan)


def test_courtlistener_discovery_uses_fixed_query_family_and_keeps_each_attempt(
    document: Document, planned_terms
) -> None:
    client = _CourtListenerClient()

    searched = asyncio.run(search_courtlistener_docket_roots(document, client=client))
    restored = Document.from_serialized(searched.serialize())
    root = restored.citations[0]
    node = next(
        item
        for item in root.trace
        if item.stage == "courtlistener_docket_search" and item.reads is Reads.RECORD
    )
    validation = node.details["validation"]

    assert client.calls == [
        "1:25-bk-11282",
        'caseName:("Example" AND "Holdings") AND court_id:nysd',
        'caseName:("Example" AND "Holdings")',
        'caseName:("Example") AND court_id:nysd',
        'caseName:("Holdings") AND court_id:nysd',
    ]
    assert validation["candidate_count"] == 1
    assert [attempt["kind"] for attempt in validation["attempts"]] == [
        "literal_docket",
        "case_name_with_stated_court",
        "case_name",
        "case_name_term_with_stated_court",
        "case_name_term_with_stated_court",
    ]
    assert validation["candidates"][0]["docket_id"] == 77
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_future_implementation"
    assert root.judgement(Question.DOCKET_LOOKUP).outcome == "search_found"


def test_govinfo_discovery_is_independent_and_serializes_its_package_provenance(
    document: Document, planned_terms
) -> None:
    courtlistener = asyncio.run(search_courtlistener_docket_roots(document, client=_CourtListenerClient()))
    client = _GovInfoClient()

    searched = asyncio.run(search_govinfo_docket_roots(courtlistener, client=client))
    restored = Document.from_serialized(searched.serialize())
    root = restored.citations[0]
    node = next(
        item for item in root.trace if item.stage == "govinfo_docket_search" and item.reads is Reads.RECORD
    )
    validation = node.details["validation"]

    assert client.calls == [
        'collection:uscourts casenumber:("1:25-bk-11282")',
        'collection:uscourts title:("Example" AND "Holdings") courtCode:nysd',
        'collection:uscourts title:("Example" AND "Holdings")',
        'collection:uscourts title:("Example") courtCode:nysd',
        'collection:uscourts title:("Holdings") courtCode:nysd',
    ]
    assert validation["candidates"][0]["govinfo_package_id"] == "USCOURTS-nysb-25-11282"
    assert [attempt["kind"] for attempt in validation["attempts"]] == [
        "literal_docket",
        "case_name_with_stated_court",
        "case_name",
        "case_name_term_with_stated_court",
        "case_name_term_with_stated_court",
    ]
    assert searched.passes[-2:] == ("courtlistener_docket_search", "govinfo_docket_search")


def test_courtlistener_discovery_keeps_every_page_of_a_bounded_precise_docket_query(
    document: Document,
    planned_terms,
) -> None:
    class PaginatedClient(_CourtListenerClient):
        def search(
            self,
            query: str,
            search_type: str,
            cursor: str | None = None,
            *,
            semantic: bool = False,
        ):
            assert search_type == "d"
            if query != "1:25-bk-11282":
                return CourtListenerSearchResult.from_payload(
                    query=query,
                    search_type=search_type,
                    semantic=False,
                    count=0,
                    results=[],
                    next_cursor=None,
                    previous_cursor=None,
                )
            if cursor is None:
                rows = [{"docket_id": index, "docketNumber": f"other-{index}"} for index in range(20)]
                return CourtListenerSearchResult.from_payload(
                    query=query,
                    search_type=search_type,
                    semantic=False,
                    count=21,
                    results=rows,
                    next_cursor="next-page",
                    previous_cursor=None,
                )
            assert cursor == "next-page"
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=False,
                count=21,
                results=[{"docket_id": 99, "docketNumber": "1:25-bk-11282"}],
                next_cursor=None,
                previous_cursor=None,
            )

    # The term plans deliberately produce name queries too. Their attempts are
    # empty; this test isolates the multi-page literal lookup contract.
    client = PaginatedClient()
    searched = asyncio.run(search_courtlistener_docket_roots(document, client=client))
    root = searched.citations[0]
    node = next(item for item in root.trace if item.node_id == "cite-0001:courtlistener_docket_search")
    literal = node.details["validation"]["attempts"][0]
    assert literal["candidate_count"] == 21
    assert [candidate["docket_id"] for candidate in literal["candidates"]] == [*range(20), 99]
