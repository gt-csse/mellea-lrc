"""Contract tests for document-native docket-root identity stages."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

import mellea_lrc.validation.docket_roots as docket_roots
from mellea_lrc.api import (
    form_roots,
    reextract_unresolved_docket_root_citations,
    relookup_reviewed_docket_roots,
    resolve_docket_root_ambiguities,
    resolve_relooked_up_docket_root_ambiguities,
    search_docket_roots,
    validate_unique_docket_root_identities,
    validate_unique_relooked_up_docket_root_identities,
)
from mellea_lrc.core.citations import CitationDate, DocketCitation, placed
from mellea_lrc.core.record import CitationRecord, Question
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerSearchResult
from mellea_lrc.extraction import Document, ExtractionMetadata
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
    formed = form_roots(_document())

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


def test_docket_identity_rejects_a_retrieved_record_with_a_different_docket_number() -> None:
    """A model may not bridge a different docket identifier during identity review."""
    client = _DocketSearchClient(count=1, results=[_candidate(docket="2:99-cv-00001")])
    searched = asyncio.run(search_docket_roots(form_roots(_document()), client=client))
    completed = asyncio.run(validate_unique_docket_root_identities(searched))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "no_match"
    assert root.found is None
    assert not any("Mellea" in node.made_by for node in root.trace)


def test_docket_identity_accepts_one_model_confirmed_equivalent_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The semantic check may admit equivalence without writing a normalized number."""
    document = _document(date=None)
    document.citations[0].stated = replace(document.citations[0].stated, docket_number="24-cv-8760")
    client = _DocketSearchClient(count=1, results=[_candidate()])
    calls = 0

    async def equivalent_number_check(validation, *, deterministic_check, **kwargs):
        nonlocal calls
        calls += 1
        return MelleaDocketNumberEquivalenceNode(
            node_id=f"{deterministic_check.node_id}:mellea_docket_number_equivalence",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketNumberEquivalenceOutcome.MATCH,
            extracted_docket_number="24-cv-8760",
            retrieved_docket_number="1:24-cv-08760",
            reason="The retrieved display adds court-context components while preserving the same case serial.",
            depends_on=(deterministic_check.node_id,),
        )

    monkeypatch.setattr(docket_roots, "run_mellea_docket_number_equivalence_check", equivalent_number_check)
    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    completed = asyncio.run(validate_unique_docket_root_identities(searched))
    restored = Document.from_serialized(completed.serialize())
    root = restored.citations[0]

    assert calls == 1
    assert root.stated.docket_number == "24-cv-8760"
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    semantic = next(
        node
        for node in root.trace
        if node.details.get("validation_node_type") == MelleaDocketNumberEquivalenceNode.__name__
    )
    assert semantic.details["validation"]["outcome"] == "match"
    assert semantic.details["validation"]["reason"]


def test_docket_identity_reextracts_the_complete_citation_before_admitting_a_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching docket string alone cannot admit a citation with another case's name."""
    document = _document(date=None)
    client = _DocketSearchClient(count=1, results=[_candidate(case_name="Other v. Case")])
    calls = 0

    async def no_match_after_combined_reextraction(
        validation,
        *,
        summary,
        document_text,
        eligible_candidate_indices,
        **kwargs,
    ):
        nonlocal calls
        calls += 1
        assert eligible_candidate_indices == (1,)
        assert document_text[
            document.citations[0].locator_span.start : document.citations[0].locator_span.end
        ]
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_candidate_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.NO_MATCH,
            candidate_indices=(1,),
            selected_candidate_index=None,
            reparsed_case_name="Smith v. Jones",
            reparsed_docket_number="1:24-cv-08760",
            reparsed_court="S.D.N.Y.",
            reparsed_date="2024",
            reparsed_pin_cite=None,
            rationale="The source case name and the retrieved case name identify different cases.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr(
        docket_roots, "run_mellea_locator_candidate_choice", no_match_after_combined_reextraction
    )
    searched = asyncio.run(search_docket_roots(form_roots(document), client=client))
    completed = asyncio.run(validate_unique_docket_root_identities(searched))
    root = completed.citations[0]

    assert calls == 1
    assert root.judgement(Question.IDENTITY).outcome == "no_match"
    choice = next(
        node
        for node in root.trace
        if node.details.get("validation_node_type") == MelleaLocatorCandidateChoiceNode.__name__
    )
    assert choice.details["validation"]["reparsed_docket_number"] == "1:24-cv-08760"
    assert choice.details["validation"]["reparsed_pin_cite"] is None


def test_bounded_docket_ambiguity_retains_all_candidates_then_selects_the_one_number_match() -> None:
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
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_future_implementation"
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
    document = _document()
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
    reviewed = asyncio.run(reextract_unresolved_docket_root_citations(initial))
    relooked_up = asyncio.run(relookup_reviewed_docket_roots(reviewed, client=client))
    resolved = asyncio.run(validate_unique_relooked_up_docket_root_identities(relooked_up))
    resolved = asyncio.run(resolve_relooked_up_docket_root_ambiguities(resolved))
    root = Document.from_serialized(resolved.serialize()).citations[0]

    assert client.calls == [
        ("1:24-cv-0876O court_id:nysd", "d", None),
        ("1:24-cv-08760 court_id:nysd", "d", None),
    ]
    assert root.stated.docket_number == "1:24-cv-08760"
    assert root.stated_fields_reparsed_by_model is True
    assert root.docket_citation_reextracted_by_model is True
    assert root.judgement(Question.DOCKET_CITATION_REEXTRACTION).outcome == "corrected"
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.authority_id == "courtlistener:docket:44"
    assert len(root.corrections) == 1

    rerun = asyncio.run(reextract_unresolved_docket_root_citations(resolved))
    assert rerun == resolved
    assert review_calls == 1
