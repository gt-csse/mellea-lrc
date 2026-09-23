"""Contract tests for bounded CourtListener and GovInfo docket discovery."""

from __future__ import annotations

import asyncio

import pytest

import mellea_lrc.validation.search.common as metadata_search_common
import mellea_lrc.validation.search.docket as docket_search
from mellea_lrc.api import (
    form_roots,
    resolve_docket_metadata_search_candidates,
    search_courtlistener_docket_roots,
    search_govinfo_docket_roots,
)
from mellea_lrc.courtlistener import CourtListenerError, CourtListenerSearchResult
from mellea_lrc.extraction.stages import CASE_NAME_STAGE
from mellea_lrc.govinfo import GovInfoSearchResult
from mellea_lrc.model.citations import CitationDate, DocketCitation, placed
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.operations import judge_citation
from mellea_lrc.model.record import CitationRecord, Node, Question, Reads
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation.search.common import (
    MetadataQuery,
    _is_compact_search_fragment,
    merge_metadata_candidates,
    run_courtlistener_metadata_attempt,
)
from mellea_lrc.validation.types import (
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    MetadataSearchAttempt,
    ValidationNodeStatus,
)
from tests.record_fixtures import read_citation, revise_citation


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
    root = formed.citations[0]
    decision = Node(
        node_id="cite-0001:semantic-deferred",
        reads=Reads.RECORD,
        stage="docket_root_semantic_resolution",
        made_by="test",
        outcome="deferred",
    )
    judge_citation(root, decision, Question.IDENTITY, "deferred_to_future_implementation")
    return formed.evolve(
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
    restored = Document.model_validate(searched.model_dump(mode="json"))
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
    restored = Document.model_validate(searched.model_dump(mode="json"))
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


def test_metadata_candidate_resolution_is_a_separate_recoverable_checkpoint(
    document: Document,
    planned_terms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    courtlistener = asyncio.run(search_courtlistener_docket_roots(document, client=_CourtListenerClient()))
    discovered = asyncio.run(search_govinfo_docket_roots(courtlistener, client=_GovInfoClient()))

    async def reject_shortlisted(validation, *, summary, shortlist, **kwargs):
        assert {item.candidate_index for item in shortlist.candidates} == {1, 2}
        assert {item.court_outcome.value for item in summary.candidates} == {"mismatch"}
        assert {item.case_name_outcome.value for item in summary.candidates} == {"unavailable"}
        assert {item.year_outcome for item in summary.candidates} == {None}
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_docket_metadata_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.NO_MATCH,
            candidate_indices=(1, 2),
            selected_candidate_index=None,
            rationale="The returned records disagree with the stated court.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr(docket_search, "run_mellea_docket_metadata_choice", reject_shortlisted)

    completed = asyncio.run(resolve_docket_metadata_search_candidates(discovered))
    restored = Document.model_validate(completed.model_dump(mode="json"))

    assert completed.passes[-1] == "docket_metadata_search_candidate_resolution"
    assert restored == completed
    assert restored.citations[0].judgement(Question.IDENTITY).outcome == "no_match"


def test_metadata_candidate_resolution_reuses_the_fuzzy_docket_shortlist(
    document: Document,
    planned_terms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later metadata discovery reaches the same bounded semantic boundary."""
    root = document.citations[0]
    revise_citation(root, court="nysb")
    courtlistener = asyncio.run(search_courtlistener_docket_roots(document, client=_CourtListenerClient()))
    discovered = asyncio.run(search_govinfo_docket_roots(courtlistener, client=_GovInfoClient()))

    async def choose_shortlisted(validation, *, summary, shortlist, **kwargs):
        assert tuple(candidate.candidate_index for candidate in shortlist.candidates) == (1, 2)
        assert shortlist.extracted_court_id == "nysb"
        assert shortlist.minimum_similarity_percent == 40.0
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_docket_metadata_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            candidate_indices=(1, 2),
            selected_candidate_index=1,
            rationale="The candidate passed the shared docket evidence boundary.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr(docket_search, "run_mellea_docket_metadata_choice", choose_shortlisted)
    completed = asyncio.run(resolve_docket_metadata_search_candidates(discovered))

    assert completed.citations[0].judgement(Question.IDENTITY).outcome == "resolved"


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


def test_failed_metadata_page_is_trace_only_not_a_candidate_for_identity() -> None:
    """A partial provider page must not be treated as reviewable evidence."""
    attempts = (
        MetadataSearchAttempt(
            kind="complete",
            query="complete query",
            court_id=None,
            status=ValidationNodeStatus.SUCCEEDED,
            candidate_count=1,
            candidates=({"docket_id": 1},),
        ),
        MetadataSearchAttempt(
            kind="partial",
            query="partial query",
            court_id=None,
            status=ValidationNodeStatus.FAILED,
            candidate_count=2,
            candidates=({"docket_id": 2},),
            error="Second result page was unavailable.",
        ),
    )

    assert merge_metadata_candidates(attempts, key="docket_id") == ({"docket_id": 1},)


def test_retryable_metadata_failure_is_retried_and_retained_in_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetryOnceClient:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
            self.calls += 1
            if self.calls == 1:
                raise CourtListenerError(
                    "rate limited",
                    failure_type="api_limit",
                    upstream_status_code=429,
                    retryable=True,
                )
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=False,
                count=1,
                results=[{"docket_id": 1, "docketNumber": "1:24-cv-00001"}],
                next_cursor=None,
                previous_cursor=None,
            )

    pauses: list[float] = []
    monkeypatch.setattr(metadata_search_common.time, "sleep", pauses.append)
    client = RetryOnceClient()
    metadata_search_common._LAST_METADATA_REQUEST_AT.pop(id(client), None)

    attempt = run_courtlistener_metadata_attempt(
        MetadataQuery(kind="direct_docket", query="1:24-cv-00001", court_id=None),
        client,
    )

    assert attempt.status is ValidationNodeStatus.SUCCEEDED
    assert attempt.retry_count == 1
    assert attempt.throttle_wait_seconds == 0.0
    assert client.calls == 2
    assert pauses == [0.5]


def test_case_name_search_fragments_remain_compact() -> None:
    """A paired metadata query stays tolerant when one party name is long."""
    assert _is_compact_search_fragment("New Mexico")
    assert _is_compact_search_fragment("Oak and Fort")
    assert not _is_compact_search_fragment("University of New Mexico Regents")
