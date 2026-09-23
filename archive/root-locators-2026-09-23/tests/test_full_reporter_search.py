"""Contracts for reporter-miss metadata discovery."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

import mellea_lrc.validation.search.reporter as reporter_search
from mellea_lrc.api import (
    form_roots,
    lookup_full_reporter_locators_exact,
    resolve_case_names,
    resolve_full_reporter_locator_ambiguities,
    resolve_full_reporter_search_candidates,
    search_courtlistener_full_reporter_roots,
    search_govinfo_full_reporter_roots,
    stable,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.courtlistener import CourtListenerCitationLookup, CourtListenerSearchResult
from mellea_lrc.govinfo import GovInfoSearchResult
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import CitationDate, FullCaseCitation, placed
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.record import CitationRecord, Node, Question, Reads
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from tests.record_fixtures import read_citation


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
            [
                {
                    "docket_id": 44,
                    "docketNumber": "1:18-cv-00754",
                    "caseName": "Example v. Holdings",
                    "decisionDate": "2019-06-01",
                }
            ]
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
    record = read_citation(
        citation_id="cite-0001",
        fields=placed(
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
    from mellea_lrc.validation.search.docket import _TermPlan

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
    restored = Document.model_validate(completed.model_dump(mode="json"))
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


def test_metadata_cutoff_excludes_later_and_undated_candidates_before_resolution(
    document: Document,
    planned_terms,
) -> None:
    courtlistener = _CourtListenerClient()
    cutoff = date(2018, 12, 31)
    exact = asyncio.run(
        lookup_full_reporter_locators_exact(document, client=courtlistener, retrospective_date=cutoff)
    )
    searched = asyncio.run(
        search_courtlistener_full_reporter_roots(exact, client=courtlistener, retrospective_date=cutoff)
    )
    searched = asyncio.run(
        search_govinfo_full_reporter_roots(searched, client=_GovInfoClient(), retrospective_date=cutoff)
    )
    restored = Document.model_validate(searched.model_dump(mode="json"))
    root = restored.citations[0]
    cl = next(
        node for node in root.trace if node.details.get("validation_node_type") == "FullReporterSearchNode"
    ).details["validation"]
    gi = next(
        node
        for node in root.trace
        if node.details.get("validation_node_type") == "GovInfoFullReporterSearchNode"
    ).details["validation"]
    assert cl["candidate_count"] == 0
    assert cl["raw_candidate_count"] == 1
    assert cl["excluded_candidate_count"] == 1
    assert cl["retrospective_date"] == "2018-12-31"
    assert gi["candidate_count"] == 0
    assert gi["excluded_candidate_count"] == 1
    assert len(cl["attempts"][0]["candidates"]) == 1
    unique = asyncio.run(validate_unique_full_reporter_locator_identities(restored, client=courtlistener))
    ambiguous = asyncio.run(resolve_full_reporter_locator_ambiguities(unique, client=courtlistener))
    resolved = asyncio.run(resolve_full_reporter_search_candidates(ambiguous))
    assert resolved.citations[0].judgement(Question.IDENTITY).outcome.startswith("deferred")


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


class _SingleTermFallbackClient(_CourtListenerClient):
    """Return one record only after a strict source-name conjunction misses."""

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        assert search_type == "d"
        assert cursor is None
        self.calls.append((query, search_type))
        rows = (
            [{"docket_id": 55, "docketNumber": "1:18-cv-00754", "caseName": "Example v. Holdings"}]
            if query == 'caseName:("Holdings") AND court_id:ncmd'
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


def test_reporter_metadata_search_relaxes_only_a_zero_result_name_conjunction(
    document: Document,
    planned_terms,
) -> None:
    client = _SingleTermFallbackClient()
    exact = asyncio.run(lookup_full_reporter_locators_exact(document, client=client))
    completed = asyncio.run(search_courtlistener_full_reporter_roots(exact, client=client))
    root = completed.citations[0]
    node = next(
        node for node in root.trace if node.details.get("validation_node_type") == "FullReporterSearchNode"
    )

    assert client.calls == [
        ('caseName:("Example" AND "Holdings") AND court_id:ncmd', "d"),
        ('caseName:("Example" AND "Holdings")', "d"),
        ('caseName:("Example") AND court_id:ncmd', "d"),
        ('caseName:("Holdings") AND court_id:ncmd', "d"),
        ('caseName:("Example")', "d"),
        ('caseName:("Holdings")', "d"),
    ]
    assert node.details["validation"]["candidate_count"] == 1
    assert node.details["validation"]["candidates"][0]["docket_id"] == 55


class _NoGovInfoClient:
    def search_uscourts(self, query: str, *, page_size: int) -> GovInfoSearchResult:
        assert page_size == 100
        return GovInfoSearchResult(query=query, count=0, results=(), next_offset_mark=None)


def test_metadata_candidate_resolution_defers_to_locator_evidence(
    document: Document,
    planned_terms,
) -> None:
    courtlistener = _CourtListenerClient()
    exact = asyncio.run(lookup_full_reporter_locators_exact(document, client=courtlistener))
    courtlistener_search = asyncio.run(search_courtlistener_full_reporter_roots(exact, client=courtlistener))
    searched = asyncio.run(
        search_govinfo_full_reporter_roots(courtlistener_search, client=_NoGovInfoClient())
    )
    unique = asyncio.run(validate_unique_full_reporter_locator_identities(searched, client=courtlistener))
    ambiguous = asyncio.run(resolve_full_reporter_locator_ambiguities(unique, client=courtlistener))
    completed = asyncio.run(resolve_full_reporter_search_candidates(ambiguous))
    restored = Document.model_validate(completed.model_dump(mode="json"))
    root = restored.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_search"
    assert root.found is None
    assert root.authority_id is None
    assert completed.passes[-1] == "full_reporter_search_candidate_resolution"
    evaluation = next(
        node for node in root.trace if node.details.get("validation_node_type") == "CandidateEvaluationNode"
    )
    assert evaluation.details["validation"]["source"] == "full_reporter_metadata_search"


def test_metadata_candidate_with_only_docket_date_defers_reporter_identity(
    document: Document,
    planned_terms,
) -> None:
    """Docket metadata cannot prove the date of a separately cited opinion."""

    class UndatedMetadataClient(_CourtListenerClient):
        def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
            result = super().search(query, search_type, cursor, semantic=semantic)
            rows = [
                {key: value for key, value in row.items() if key != "decisionDate"} for row in result.results
            ]
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=semantic,
                count=len(rows),
                results=rows,
                next_cursor=None,
                previous_cursor=None,
            )

    courtlistener = UndatedMetadataClient()
    exact = asyncio.run(lookup_full_reporter_locators_exact(document, client=courtlistener))
    courtlistener_search = asyncio.run(search_courtlistener_full_reporter_roots(exact, client=courtlistener))
    searched = asyncio.run(
        search_govinfo_full_reporter_roots(courtlistener_search, client=_NoGovInfoClient())
    )
    unique = asyncio.run(validate_unique_full_reporter_locator_identities(searched, client=courtlistener))
    ambiguous = asyncio.run(resolve_full_reporter_locator_ambiguities(unique, client=courtlistener))
    completed = asyncio.run(resolve_full_reporter_search_candidates(ambiguous))

    root = completed.citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_search"
    assert root.found is None


class _ConflictingCourtClient(_CourtListenerClient):
    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        result = super().search(query, search_type, cursor, semantic=semantic)
        rows = [dict(row, court_id="nyed") for row in result.results]
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=semantic,
            count=len(rows),
            results=rows,
            next_cursor=None,
            previous_cursor=None,
        )


def test_metadata_candidate_with_wrong_court_is_rejected_without_model_review(
    document: Document,
    planned_terms,
) -> None:
    courtlistener = _ConflictingCourtClient()
    exact = asyncio.run(lookup_full_reporter_locators_exact(document, client=courtlistener))
    courtlistener_search = asyncio.run(search_courtlistener_full_reporter_roots(exact, client=courtlistener))
    searched = asyncio.run(
        search_govinfo_full_reporter_roots(courtlistener_search, client=_NoGovInfoClient())
    )
    unique = asyncio.run(validate_unique_full_reporter_locator_identities(searched, client=courtlistener))
    ambiguous = asyncio.run(resolve_full_reporter_locator_ambiguities(unique, client=courtlistener))
    completed = asyncio.run(resolve_full_reporter_search_candidates(ambiguous))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_search"
    assert root.found is None
    assessment = next(
        node
        for node in root.trace
        if node.details.get("validation_node_type") == "LocatorCandidateAssessmentNode"
    )
    assert assessment.details["validation"]["court_outcome"] == "mismatch"
