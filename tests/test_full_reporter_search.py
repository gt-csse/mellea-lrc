"""Contracts for reporter-miss metadata discovery."""

from __future__ import annotations

import asyncio

import pytest

import mellea_lrc.validation.full_reporter_search as reporter_search
from mellea_lrc.api import (
    form_roots,
    lookup_full_reporter_locators_exact,
    resolve_case_names,
    search_courtlistener_full_reporter_roots,
    search_govinfo_full_reporter_roots,
    stable,
)
from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import CitationDate, FullCaseCitation, placed
from mellea_lrc.core.record import CitationRecord, Node, Question, Reads
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerCitationLookup, CourtListenerSearchResult
from mellea_lrc.extraction import Document, ExtractionMetadata
from mellea_lrc.govinfo import GovInfoSearchResult
from mellea_lrc.preprocessing import preprocess


class _CourtListenerClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        return CourtListenerCitationLookup(citation=f"{volume} {reporter} {page}", status=404, clusters=())

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        assert search_type == "d"
        assert cursor is None
        self.calls.append((query, search_type))
        rows = (
            [{"docket_id": 44, "docketNumber": "1:18-cv-00754", "caseName": "Example v. Holdings"}]
            if query == 'caseName:("Example" AND "Holdings") AND court_id:ncmd'
            else []
        )
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=False,
            count=len(rows),
            results=rows,
            next_cursor=None,
            previous_cursor=None,
        )


class _GovInfoClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search_uscourts(self, query: str, *, page_size: int) -> GovInfoSearchResult:
        assert page_size == 100
        self.calls.append(query)
        rows = (
            [{"packageId": "USCOURTS-ncmd-1_18-cv-00754", "title": "Example v. Holdings"}]
            if query == 'collection:uscourts title:("Example" AND "Holdings") courtCode:ncmd'
            else []
        )
        return GovInfoSearchResult(query=query, count=len(rows), results=tuple(rows), next_offset_mark=None)


@pytest.fixture
def document() -> Document:
    text = "Example v. Holdings, 426 F. Supp. 3d 151 (M.D.N.C. 2019)."
    preprocessed = preprocess(text)
    locator = "426 F. Supp. 3d 151"
    citation = FullCaseCitation(
        volume="426",
        reporter="F. Supp. 3d",
        page="151",
        court="ncmd",
        date=CitationDate(year="2019"),
        case_name=CaseName(span=Span(0, len("Example v. Holdings")), text="Example v. Holdings"),
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
    return resolve_case_names(formed, rules=stable())


@pytest.fixture
def planned_terms(monkeypatch: pytest.MonkeyPatch):
    # The simple model substitute writes the same node shape the production
    # stage persists, while keeping this retrieval contract offline.
    from mellea_lrc.validation.docket_search import _TermPlan

    async def fixed_plan(record, *, stage: str, session):
        return _TermPlan(
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

    monkeypatch.setattr(reporter_search, "_prepare_terms", fixed_plan)


def test_reporter_metadata_searches_are_independent_and_keep_exact_miss_unjudged(
    document: Document,
    planned_terms,
) -> None:
    courtlistener = _CourtListenerClient()
    exact = asyncio.run(lookup_full_reporter_locators_exact(document, client=courtlistener))
    searched = asyncio.run(search_courtlistener_full_reporter_roots(exact, client=courtlistener))
    govinfo = _GovInfoClient()
    completed = asyncio.run(search_govinfo_full_reporter_roots(searched, client=govinfo))
    restored = Document.from_serialized(completed.serialize())
    root = restored.citations[0]

    assert courtlistener.calls == [
        ('caseName:("Example" AND "Holdings") AND court_id:ncmd', "d"),
        ('caseName:("Example" AND "Holdings")', "d"),
    ]
    assert govinfo.calls == [
        'collection:uscourts title:("Example" AND "Holdings") courtCode:ncmd',
        'collection:uscourts title:("Example" AND "Holdings")',
    ]
    assert root.judgement(Question.IDENTITY).outcome == "unjudged"
    assert root.judgement(Question.LOCATOR_LOOKUP).outcome == "search_found"
    cl_node = next(
        node for node in root.trace if node.details.get("validation_node_type") == "FullReporterSearchNode"
    )
    gi_node = next(
        node
        for node in root.trace
        if node.details.get("validation_node_type") == "GovInfoFullReporterSearchNode"
    )
    assert cl_node.details["validation"]["candidates"][0]["docket_id"] == 44
    assert (
        gi_node.details["validation"]["candidates"][0]["govinfo_package_id"] == "USCOURTS-ncmd-1_18-cv-00754"
    )


def test_reporter_metadata_search_marks_name_free_miss_unavailable(
    document: Document,
    planned_terms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_terms(record, *, stage: str, session):
        return reporter_search.MetadataTermPlan(
            terms=(),
            node=Node(
                node_id=f"{record.citation_id}:{stage}:case_name_terms",
                reads=Reads.DOCUMENT,
                stage=stage,
                made_by="test",
                outcome="unavailable",
                details={"terms": []},
            ),
        )

    monkeypatch.setattr(reporter_search, "_prepare_terms", no_terms)
    client = _CourtListenerClient()
    exact = asyncio.run(lookup_full_reporter_locators_exact(document, client=client))
    completed = asyncio.run(search_courtlistener_full_reporter_roots(exact, client=client))
    root = completed.citations[0]
    node = next(
        node for node in root.trace if node.details.get("validation_node_type") == "FullReporterSearchNode"
    )

    assert client.calls == []
    assert node.details["validation"]["outcome"] == "unavailable"
    assert root.judgement(Question.LOCATOR_LOOKUP).outcome == "search_unavailable"
