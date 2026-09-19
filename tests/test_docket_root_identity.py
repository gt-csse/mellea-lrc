"""Contract tests for document-native docket-root identity stages."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

import mellea_lrc.validation.docket_roots as docket_roots
from mellea_lrc.api import (
    form_roots,
    lookup_govinfo_docket_roots,
    resolve_docket_root_ambiguities,
    resolve_docket_root_semantics,
    resolve_govinfo_docket_root_ambiguities,
    resolve_requeued_docket_root_ambiguities,
    review_and_requeue_unresolved_docket_roots,
    search_docket_roots,
    validate_unique_docket_root_identities,
    validate_unique_govinfo_docket_root_identities,
    validate_unique_requeued_docket_root_identities,
)
from mellea_lrc.core.citations import CitationDate, DocketCitation, placed
from mellea_lrc.core.record import CitationRecord, Question
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerSearchResult
from mellea_lrc.extraction import Document, ExtractionMetadata
from mellea_lrc.govinfo import GovInfoSearchResult
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation.types import (
    MelleaDocketCitationReextractionNode,
    MelleaDocketCitationReextractionOutcome,
    MelleaDocketNumberEquivalenceNode,
    MelleaDocketNumberEquivalenceOutcome,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    ValidationNodeStatus,
)


class _DocketSearchClient:
    def __init__(self, *, count: int, results: list[dict[str, object]]) -> None:
        self.count = count
        self.results = results
        self.calls: list[tuple[str, str, str | None]] = []

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        self.calls.append((query, search_type, cursor))
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=semantic,
            count=self.count,
            results=self.results,
            next_cursor=None,
            previous_cursor=None,
        )


class _CorrectedDocketSearchClient(_DocketSearchClient):
    """Returns a result only after the model has corrected the parsed number."""

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        self.calls.append((query, search_type, cursor))
        corrected = "1:24-cv-08760"
        results = [_candidate(docket=corrected)] if corrected in query else []
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=semantic,
            count=len(results),
            results=results,
            next_cursor=None,
            previous_cursor=None,
        )


class _GovInfoSearchClient:
    """Small fake for the independent published-opinion fallback."""

    def __init__(self, *, count: int, results: list[dict[str, object]]) -> None:
        self.count = count
        self.results = results
        self.calls: list[tuple[str, str | None, int]] = []

    def search_uscourts_docket(self, docket_number: str, *, court_id: str | None, page_size: int):
        self.calls.append((docket_number, court_id, page_size))
        return GovInfoSearchResult(
            query=f'collection:uscourts casenumber:("{docket_number}") courtCode:{court_id}',
            count=self.count,
            results=tuple(self.results),
            next_offset_mark=None,
        )


def _document(*, court: str | None = "nysd", date: str | None = "2024") -> Document:
    text = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)."
    locator = "1:24-cv-08760"
    locator_start = text.index(locator)
    preprocessed = preprocess(text)
    citation = DocketCitation(
        plaintiff="Smith",
        defendant="Jones",
        docket_number=locator,
        court=court,
        date=CitationDate(year=date) if date else None,
    )
    record = CitationRecord(
        citation_id="cite-0001",
        source=placed(
            citation,
            span=Span(0, len(text)),
            locator_span=Span(locator_start, locator_start + len(locator)),
            matched_text=locator,
        ),
    )
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(record,),
        extraction_metadata=ExtractionMetadata(),
    )


def _candidate(
    *,
    docket: str = "1:24-cv-08760",
    docket_id: int = 44,
    case_name: str = "Smith v. Jones",
) -> dict[str, object]:
    return {
        "docket_id": docket_id,
        "docketNumber": docket,
        "caseName": case_name,
        "court_id": "nysd",
        "dateFiled": "2024-01-05",
        "docket_absolute_url": f"/docket/{docket_id}/smith-v-jones/",
    }


def test_docket_search_and_unique_identity_write_one_resumable_root_decision() -> None:
    client = _DocketSearchClient(count=1, results=[_candidate()])
    formed = form_roots(_document(date=None))

    searched = asyncio.run(search_docket_roots(formed, client=client))
    checkpoint = Document.from_serialized(searched.serialize())
    unique = asyncio.run(validate_unique_docket_root_identities(checkpoint))
    completed = asyncio.run(resolve_docket_root_ambiguities(unique))
    restored = Document.from_serialized(completed.serialize())
    root = restored.citations[0]

    assert client.calls == [("1:24-cv-08760 court_id:nysd", "d", None)]
    assert root.judgement(Question.DOCKET_LOOKUP).outcome == "found"
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.found is not None and root.found.docket_id == "44"
    assert root.authority_id == "courtlistener:docket:44"
    search_node = next(node for node in root.trace if node.stage == "docket_root_search")
    assert search_node.details["validation"]["query"] == "1:24-cv-08760 court_id:nysd"
    assert completed.passes[-4:] == (
        "root_formation",
        "docket_root_search",
        "docket_root_unique_identity",
        "docket_root_ambiguity_resolution",
    )


def test_docket_search_is_not_blocked_by_a_missing_court() -> None:
    client = _DocketSearchClient(count=1, results=[_candidate()])
    searched = asyncio.run(search_docket_roots(form_roots(_document(court=None, date=None)), client=client))
    completed = asyncio.run(
        resolve_docket_root_ambiguities(asyncio.run(validate_unique_docket_root_identities(searched)))
    )

    assert client.calls == [("1:24-cv-08760", "d", None)]
    assert completed.citations[0].judgement(Question.IDENTITY).outcome == "resolved"


def test_govinfo_fallback_resolves_a_courtlistener_docket_miss() -> None:
    """GovInfo's package record is a separately persisted positive identity route."""
    courtlistener = _DocketSearchClient(count=0, results=[])
    govinfo = _GovInfoSearchClient(
        count=1,
        results=[
            {
                "packageId": "USCOURTS-nysd-1_24-cv-08760",
                "title": "Smith v. Jones",
                "dateIssued": "2024-01-06",
            }
        ],
    )
    searched = asyncio.run(search_docket_roots(form_roots(_document()), client=courtlistener))
    unique = asyncio.run(validate_unique_docket_root_identities(searched))
    ambiguous = asyncio.run(resolve_docket_root_ambiguities(unique))
    looked_up = asyncio.run(lookup_govinfo_docket_roots(ambiguous, client=govinfo))
    completed = asyncio.run(
        resolve_govinfo_docket_root_ambiguities(
            asyncio.run(validate_unique_govinfo_docket_root_identities(looked_up))
        )
    )
    root = completed.citations[0]

    assert govinfo.calls == [("1:24-cv-08760", "nysd", 20)]
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.authority_id == "govinfo:package:USCOURTS-nysd-1_24-cv-08760"
    assert root.found is not None
    assert root.found.govinfo_package_id == "USCOURTS-nysd-1_24-cv-08760"
    assert "govinfo_docket_root_search" in completed.passes


def test_semantic_docket_stage_uses_a_positive_govinfo_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonliteral GovInfo package number reaches semantic equivalence."""
    document = _document(date="2021")
    document.citations[0].stated = replace(document.citations[0].stated, docket_number="24-cv-8760")
    courtlistener = _DocketSearchClient(count=0, results=[])
    govinfo = _GovInfoSearchClient(
        count=1,
        results=[
            {
                "packageId": "USCOURTS-nysd-1_24-cv-08760",
                "title": "Smith v. Jones",
                "dateIssued": "2024-01-06",
            }
        ],
    )

    async def unchanged_review(record, **kwargs):
        return MelleaDocketCitationReextractionNode(
            node_id=f"{record.citation_id}:mellea_docket_citation_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.UNCHANGED,
            source_citation="Smith v. Jones, Case No. 24-cv-8760 (S.D.N.Y. 2024).",
            source_locator="24-cv-8760",
            extracted_docket_number="24-cv-8760",
            reparsed_case_name="Smith v. Jones",
            reparsed_docket_number="24-cv-8760",
            reparsed_court="S.D.N.Y.",
            reparsed_date="2024",
            reparsed_pin_cite=None,
            grounded_docket_number="24-cv-8760",
            reason="The source states the abbreviated docket form.",
            depends_on=("cite-0001:govinfo_docket_root_search:identity_resolution",),
        )

    async def matching_equivalence(validation, *, deterministic_check, candidate, **kwargs):
        return MelleaDocketNumberEquivalenceNode(
            node_id=f"{deterministic_check.node_id}:mellea_docket_number_equivalence",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketNumberEquivalenceOutcome.MATCH,
            extracted_docket_number=deterministic_check.extracted_docket_number,
            retrieved_docket_number=deterministic_check.retrieved_docket_number,
            reason="The retrieved form adds the district prefix and zero padding.",
            depends_on=(deterministic_check.node_id,),
        )

    async def choose_candidate(validation, *, summary, eligible_candidate_indices, **kwargs):
        assert eligible_candidate_indices == (1,)
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_candidate_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            candidate_indices=eligible_candidate_indices,
            selected_candidate_index=1,
            rationale="The case and equivalent docket forms identify candidate 1.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr(docket_roots, "run_mellea_docket_citation_reextraction", unchanged_review)
    monkeypatch.setattr(docket_roots, "run_mellea_docket_number_equivalence_check", matching_equivalence)
    monkeypatch.setattr(docket_roots, "run_mellea_locator_candidate_choice", choose_candidate)

    searched = asyncio.run(search_docket_roots(form_roots(document), client=courtlistener))
    unique = asyncio.run(validate_unique_docket_root_identities(searched))
    initial = asyncio.run(resolve_docket_root_ambiguities(unique))
    fallback = asyncio.run(lookup_govinfo_docket_roots(initial, client=govinfo))
    fallback = asyncio.run(validate_unique_govinfo_docket_root_identities(fallback))
    fallback = asyncio.run(resolve_govinfo_docket_root_ambiguities(fallback))
    reviewed = asyncio.run(review_and_requeue_unresolved_docket_roots(fallback, client=courtlistener))
    requeued = asyncio.run(validate_unique_requeued_docket_root_identities(reviewed))
    ready = asyncio.run(resolve_requeued_docket_root_ambiguities(requeued))
    completed = asyncio.run(resolve_docket_root_semantics(ready))

    root = Document.from_serialized(completed.serialize()).citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.authority_id == "govinfo:package:USCOURTS-nysd-1_24-cv-08760"
    assert any(
        node.stage == "docket_root_semantic_resolution"
        and node.details.get("validation_node_type") == MelleaDocketNumberEquivalenceNode.__name__
        for node in root.trace
    )
    year = next(node for node in root.trace if node.details.get("validation_node_type") == "YearCheckNode")
    assert year.details["validation"]["outcome"] == "unavailable"


def test_docket_identity_defers_when_case_filing_postdates_a_stated_decision() -> None:
    """A docket opened after the cited decision is contradictory evidence."""
    document = _document(date=None)
    document.citations[0].stated = replace(
        document.citations[0].stated,
        date=CitationDate(year="2024", month="Jan.", day="4"),
    )
    client = _DocketSearchClient(count=1, results=[_candidate()])

    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    completed = asyncio.run(validate_unique_docket_root_identities(searched))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "no_match"
    year = next(node for node in root.trace if node.details.get("validation_node_type") == "YearCheckNode")
    assert year.details["validation"]["outcome"] == "mismatch"


def test_docket_identity_admits_a_case_filed_before_a_stated_decision() -> None:
    """A case filing before the cited decision is compatible identity evidence."""
    document = _document(date=None)
    document.citations[0].stated = replace(
        document.citations[0].stated,
        date=CitationDate(year="2024", month="Jan.", day="6"),
    )
    client = _DocketSearchClient(count=1, results=[_candidate()])

    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    completed = asyncio.run(validate_unique_docket_root_identities(searched))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    year = next(node for node in root.trace if node.details.get("validation_node_type") == "YearCheckNode")
    assert year.details["validation"]["outcome"] == "match"


def test_docket_identity_rejects_a_retrieved_record_with_a_different_docket_number() -> None:
    """A model may not bridge a different docket identifier during identity review."""
    client = _DocketSearchClient(count=1, results=[_candidate(docket="2:99-cv-00001")])
    searched = asyncio.run(search_docket_roots(form_roots(_document()), client=client))
    completed = asyncio.run(validate_unique_docket_root_identities(searched))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "no_match"
    assert root.found is None
    assert not any("Mellea" in node.made_by for node in root.trace)


def test_programmatic_docket_identity_defers_nonliteral_docket_forms_to_semantics() -> None:
    """Equivalent-looking strings are not a first-pass identity decision."""
    document = _document(date=None)
    document.citations[0].stated = replace(document.citations[0].stated, docket_number="24-cv-8760")
    client = _DocketSearchClient(count=1, results=[_candidate()])

    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    completed = asyncio.run(validate_unique_docket_root_identities(searched))
    root = completed.citations[0]

    assert root.stated.docket_number == "24-cv-8760"
    assert root.judgement(Question.IDENTITY).outcome == "no_match"
    assert root.found is None
    assert not any("Mellea" in node.made_by for node in root.trace)


def test_docket_name_mismatch_is_extraction_reviewed_then_deferred_to_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A docket match cannot admit a different case name before semantic review."""
    document = _document(date=None)
    client = _DocketSearchClient(count=1, results=[_candidate(case_name="Other v. Case")])
    calls = 0

    async def unchanged_complete_reextraction(record, **kwargs):
        nonlocal calls
        calls += 1
        return MelleaDocketCitationReextractionNode(
            node_id=f"{record.citation_id}:mellea_docket_citation_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.UNCHANGED,
            source_citation="Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024).",
            source_locator="1:24-cv-08760",
            extracted_docket_number="1:24-cv-08760",
            reparsed_case_name="Smith v. Jones",
            reparsed_docket_number="1:24-cv-08760",
            reparsed_court="S.D.N.Y.",
            reparsed_date="2024",
            reparsed_pin_cite=None,
            grounded_docket_number="1:24-cv-08760",
            reason="The filing states Smith v. Jones, while the retrieved candidate names another case.",
            depends_on=("cite-0001:docket_root_search:identity_resolution",),
        )

    monkeypatch.setattr(
        docket_roots, "run_mellea_docket_citation_reextraction", unchanged_complete_reextraction
    )
    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    initial = asyncio.run(validate_unique_docket_root_identities(searched))
    initial = asyncio.run(resolve_docket_root_ambiguities(initial))
    assert initial.citations[0].judgement(Question.IDENTITY).outcome == "deferred_to_semantic_review"

    reviewed = asyncio.run(review_and_requeue_unresolved_docket_roots(initial, client=client))
    root = reviewed.citations[0]

    assert calls == 1
    assert root.extraction_reviewed_by_llm is True
    assert root.judgement(Question.EXTRACTION_REVIEW).outcome == "unchanged"
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_semantic_review"
    assert root.found is None
    reextraction = next(
        node
        for node in root.trace
        if node.details.get("validation_node_type") == MelleaDocketCitationReextractionNode.__name__
    )
    assert reextraction.details["validation"]["reparsed_docket_number"] == "1:24-cv-08760"
    assert reextraction.details["validation"]["reparsed_pin_cite"] is None


def test_bounded_docket_ambiguity_retains_all_candidates_then_resolves_one_exact_match() -> None:
    client = _DocketSearchClient(
        count=2, results=[_candidate(docket="2:99-cv-00001", docket_id=1), _candidate()]
    )
    searched = asyncio.run(search_docket_roots(form_roots(_document(date=None)), client=client))
    unique = asyncio.run(validate_unique_docket_root_identities(searched))
    completed = asyncio.run(resolve_docket_root_ambiguities(unique))
    root = completed.citations[0]

    assert root.judgement(Question.DOCKET_LOOKUP).outcome == "deferred_to_ambiguity"
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.authority_id == "courtlistener:docket:44"
    evaluations = [
        node for node in root.trace if node.details.get("validation_node_type") == "CandidateEvaluationNode"
    ]
    assert len(evaluations) == 2


def test_docket_search_defers_twenty_or_more_candidates_without_truncating_a_review() -> None:
    client = _DocketSearchClient(count=20, results=[_candidate(docket_id=index) for index in range(20)])
    searched = asyncio.run(search_docket_roots(form_roots(_document()), client=client))
    unique = asyncio.run(validate_unique_docket_root_identities(searched))
    completed = asyncio.run(resolve_docket_root_ambiguities(unique))
    root = completed.citations[0]

    assert root.judgement(Question.DOCKET_LOOKUP).outcome == "deferred_to_future_implementation"
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_semantic_review"
    search = next(node for node in root.trace if node.stage == "docket_root_search")
    assert search.details["validation"]["candidate_count"] == 20
    assert len(search.details["validation"]["candidates"]) == 20


def test_docket_search_does_not_page_an_out_of_bounds_result_set() -> None:
    """The review ceiling bounds retrieval work as well as candidate choice."""
    client = _DocketSearchClient(count=21, results=[_candidate()])
    searched = asyncio.run(search_docket_roots(form_roots(_document()), client=client))
    root = searched.citations[0]
    search = next(node for node in root.trace if node.stage == "docket_root_search")

    assert root.judgement(Question.DOCKET_LOOKUP).outcome == "deferred_to_future_implementation"
    assert search.details["validation"]["candidate_count"] == 21
    assert len(search.details["validation"]["candidates"]) == 1
    assert client.calls == [("1:24-cv-08760 court_id:nysd", "d", None)]


def test_failed_docket_lookup_is_reviewed_once_then_corrected_and_requeued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document(date=None)
    malformed = replace(document.citations[0].stated, docket_number="1:24-cv-0876O")
    document.citations[0].stated = malformed
    client = _CorrectedDocketSearchClient(count=0, results=[])
    review_calls = 0

    async def corrected_review(record, **kwargs):
        nonlocal review_calls
        review_calls += 1
        return MelleaDocketCitationReextractionNode(
            node_id=f"{record.citation_id}:mellea_docket_citation_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.CORRECTED,
            source_citation="Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024).",
            source_locator="1:24-cv-08760",
            extracted_docket_number="1:24-cv-0876O",
            reparsed_case_name="Smith v. Jones",
            reparsed_docket_number="1:24-cv-08760",
            reparsed_court="S.D.N.Y.",
            reparsed_date="2024",
            reparsed_pin_cite=None,
            grounded_docket_number="1:24-cv-08760",
            reason="The source locator ends in zero, not capital O.",
            depends_on=("cite-0001:locator_identity_resolution",),
        )

    monkeypatch.setattr(docket_roots, "run_mellea_docket_citation_reextraction", corrected_review)
    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    initial = asyncio.run(validate_unique_docket_root_identities(searched))
    initial = asyncio.run(resolve_docket_root_ambiguities(initial))
    requeued = asyncio.run(review_and_requeue_unresolved_docket_roots(initial, client=client))
    resolved = asyncio.run(validate_unique_requeued_docket_root_identities(requeued))
    resolved = asyncio.run(resolve_requeued_docket_root_ambiguities(resolved))
    root = Document.from_serialized(resolved.serialize()).citations[0]

    assert client.calls == [
        ("1:24-cv-0876O court_id:nysd", "d", None),
        ("1:24-cv-08760 court_id:nysd", "d", None),
    ]
    assert root.stated.docket_number == "1:24-cv-08760"
    assert root.extraction_reviewed_by_llm is True
    assert root.judgement(Question.EXTRACTION_REVIEW).outcome == "corrected"
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.authority_id == "courtlistener:docket:44"
    assert len(root.corrections) == 1

    rerun = asyncio.run(review_and_requeue_unresolved_docket_roots(resolved, client=client))
    assert rerun == resolved
    assert review_calls == 1


@pytest.mark.parametrize(
    ("date", "expected_identity"),
    [
        (None, "resolved"),
        ("2024", "resolved"),
    ],
)
def test_semantic_docket_stage_confirms_a_nonliteral_court_anchored_record(
    monkeypatch: pytest.MonkeyPatch,
    date: str | None,
    expected_identity: str,
) -> None:
    """A semantic docket form and matching court directly identify one record."""
    document = _document(date=date)
    document.citations[0].stated = replace(document.citations[0].stated, docket_number="24-cv-8760")
    client = _DocketSearchClient(count=1, results=[_candidate()])

    async def unchanged_review(record, **kwargs):
        return MelleaDocketCitationReextractionNode(
            node_id=f"{record.citation_id}:mellea_docket_citation_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.UNCHANGED,
            source_citation="Smith v. Jones, Case No. 24-cv-8760 (S.D.N.Y. 2024).",
            source_locator="24-cv-8760",
            extracted_docket_number="24-cv-8760",
            reparsed_case_name="Smith v. Jones",
            reparsed_docket_number="24-cv-8760",
            reparsed_court="S.D.N.Y.",
            reparsed_date="2024",
            reparsed_pin_cite=None,
            grounded_docket_number="24-cv-8760",
            reason="The source states the abbreviated docket form.",
            depends_on=("cite-0001:docket_root_search:identity_resolution",),
        )

    async def matching_equivalence(validation, *, deterministic_check, candidate, **kwargs):
        return MelleaDocketNumberEquivalenceNode(
            node_id=f"{deterministic_check.node_id}:mellea_docket_number_equivalence",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketNumberEquivalenceOutcome.MATCH,
            extracted_docket_number=deterministic_check.extracted_docket_number,
            retrieved_docket_number=deterministic_check.retrieved_docket_number,
            reason="The retrieved form adds the district prefix and zero padding.",
            depends_on=(deterministic_check.node_id,),
        )

    async def should_not_choose_candidate(*args, **kwargs):
        raise AssertionError("A unique court-anchored docket record must not require candidate selection")

    monkeypatch.setattr(docket_roots, "run_mellea_docket_citation_reextraction", unchanged_review)
    monkeypatch.setattr(docket_roots, "run_mellea_docket_number_equivalence_check", matching_equivalence)
    monkeypatch.setattr(docket_roots, "run_mellea_locator_candidate_choice", should_not_choose_candidate)

    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    unique = asyncio.run(validate_unique_docket_root_identities(searched))
    ambiguous = asyncio.run(resolve_docket_root_ambiguities(unique))
    reviewed = asyncio.run(review_and_requeue_unresolved_docket_roots(ambiguous, client=client))
    requeued_unique = asyncio.run(validate_unique_requeued_docket_root_identities(reviewed))
    ready = asyncio.run(resolve_requeued_docket_root_ambiguities(requeued_unique))
    resolved = asyncio.run(resolve_docket_root_semantics(ready))
    root = Document.from_serialized(resolved.serialize()).citations[0]

    assert root.judgement(Question.IDENTITY).outcome == expected_identity
    assert root.authority_id == "courtlistener:docket:44"
    assert "docket_root_semantic_resolution" in resolved.passes
    equivalence = next(
        node
        for node in root.trace
        if node.details.get("validation_node_type") == MelleaDocketNumberEquivalenceNode.__name__
    )
    assert equivalence.reads.value == "document"


def test_semantic_docket_stage_sends_caption_drift_to_representative_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A docket-and-court match cannot bypass a contradictory stated caption."""
    document = _document(date=None)
    client = _DocketSearchClient(count=1, results=[_candidate(case_name="Renamed Case, LLC")])

    async def unchanged_review(record, **kwargs):
        return MelleaDocketCitationReextractionNode(
            node_id=f"{record.citation_id}:mellea_docket_citation_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.UNCHANGED,
            source_citation="Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024).",
            source_locator="1:24-cv-08760",
            extracted_docket_number="1:24-cv-08760",
            reparsed_case_name="Smith v. Jones",
            reparsed_docket_number="1:24-cv-08760",
            reparsed_court="S.D.N.Y.",
            reparsed_date="2024",
            reparsed_pin_cite=None,
            grounded_docket_number="1:24-cv-08760",
            reason="The source confirms the docket locator.",
            depends_on=("cite-0001:docket_root_search:identity_resolution",),
        )

    async def choose_no_match(validation, *, summary, eligible_candidate_indices, **kwargs):
        assert eligible_candidate_indices == (1,)
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_candidate_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.NO_MATCH,
            candidate_indices=eligible_candidate_indices,
            selected_candidate_index=None,
            rationale="The stated caption and retrieved caption identify different matters.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr(docket_roots, "run_mellea_docket_citation_reextraction", unchanged_review)
    monkeypatch.setattr(docket_roots, "run_mellea_locator_candidate_choice", choose_no_match)

    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    unique = asyncio.run(validate_unique_docket_root_identities(searched))
    ambiguity = asyncio.run(resolve_docket_root_ambiguities(unique))
    reviewed = asyncio.run(review_and_requeue_unresolved_docket_roots(ambiguity, client=client))
    requeued = asyncio.run(validate_unique_requeued_docket_root_identities(reviewed))
    ready = asyncio.run(resolve_requeued_docket_root_ambiguities(requeued))
    completed = asyncio.run(resolve_docket_root_semantics(ready))

    root = completed.citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "no_match"
    assert root.authority_id is None
    assessment = next(
        node
        for node in root.trace
        if node.details.get("validation_node_type") == "LocatorCandidateAssessmentNode"
        and node.stage == "docket_root_semantic_resolution"
    )
    assert assessment.details["validation"]["case_name_outcome"] == "mismatch"


def test_semantic_docket_stage_leaves_no_candidate_and_review_limit_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A semantic stage cannot turn absent or unbounded retrieval into no-match."""
    document = _document()
    client = _DocketSearchClient(count=20, results=[_candidate(docket_id=index) for index in range(20)])

    async def unchanged_review(record, **kwargs):
        return MelleaDocketCitationReextractionNode(
            node_id=f"{record.citation_id}:mellea_docket_citation_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.UNCHANGED,
            source_citation="Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024).",
            source_locator="1:24-cv-08760",
            extracted_docket_number="1:24-cv-08760",
            reparsed_case_name="Smith v. Jones",
            reparsed_docket_number="1:24-cv-08760",
            reparsed_court="S.D.N.Y.",
            reparsed_date="2024",
            reparsed_pin_cite=None,
            grounded_docket_number="1:24-cv-08760",
            reason="The source confirms the extracted docket.",
            depends_on=("cite-0001:docket_root_search:identity_resolution",),
        )

    monkeypatch.setattr(docket_roots, "run_mellea_docket_citation_reextraction", unchanged_review)
    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    unique = asyncio.run(validate_unique_docket_root_identities(searched))
    ambiguous = asyncio.run(resolve_docket_root_ambiguities(unique))
    reviewed = asyncio.run(review_and_requeue_unresolved_docket_roots(ambiguous, client=client))
    requeued_unique = asyncio.run(validate_unique_requeued_docket_root_identities(reviewed))
    ready = asyncio.run(resolve_requeued_docket_root_ambiguities(requeued_unique))
    completed = asyncio.run(resolve_docket_root_semantics(ready))

    root = completed.citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_future_implementation"
    assert root.authority_id is None
    assert not any("Mellea" in node.made_by and node.stage == "docket_root_semantic_resolution" for node in root.trace)
