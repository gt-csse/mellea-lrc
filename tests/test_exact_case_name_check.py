"""Tests for the exact case-name field check."""

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import ExtractedCitation
from mellea_lrc.validation.field_checks.exact_case_name_check import run_exact_case_name_check
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationOutcome,
    CandidateEvaluationSource,
    CitationValidation,
    FieldCheckOutcome,
    ValidationNodeStatus,
)


def _validation_with_citation(citation: FullCaseCitation) -> CitationValidation:
    extracted = ExtractedCitation(
        citation_id="cite-0001",
        span=Span(0, 10),
        locator_span=Span(0, 10),
        matched_text="347 U.S. 483",
        citation=citation,
    )
    return CitationValidation(citation=extracted)


def _candidate(case_name: str | None) -> CandidateEvaluationNode:
    return CandidateEvaluationNode(
        node_id="cite-0001:locator_candidate_evaluation:1",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=CandidateEvaluationOutcome.READY,
        source=CandidateEvaluationSource.LOCATOR_LOOKUP,
        candidate_index=1,
        cluster_id="123",
        case_name=case_name,
        date_filed=None,
        court_id=None,
        docket_id=None,
        record={},
        depends_on=(),
    )


def test_single_party_defendant_only_uses_that_party_directly() -> None:
    """An 'In re' style caption has only one party - use it, don't call the check unavailable."""
    citation = FullCaseCitation(plaintiff="", defendant="Soundview Elite Ltd.", volume="503", reporter="B.R.", page="571")
    validation = _validation_with_citation(citation)
    candidate = _candidate("In re Soundview Elite, Ltd.")

    node = run_exact_case_name_check(validation, candidate=candidate)

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.extracted_case_name == "Soundview Elite Ltd."
    assert node.outcome is FieldCheckOutcome.MISMATCH


def test_single_party_plaintiff_only_uses_that_party_directly() -> None:
    """The same holds when only the plaintiff field is populated."""
    citation = FullCaseCitation(plaintiff="Ex Parte Young", defendant="", volume="209", reporter="U.S.", page="123")
    validation = _validation_with_citation(citation)
    candidate = _candidate("Ex Parte Young")

    node = run_exact_case_name_check(validation, candidate=candidate)

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.extracted_case_name == "Ex Parte Young"
    assert node.outcome is FieldCheckOutcome.MATCH


def test_no_party_extracted_is_still_unavailable() -> None:
    """A citation with neither party extracted has nothing to compare, unlike single-party captions."""
    citation = FullCaseCitation(volume="347", reporter="U.S.", page="483")
    validation = _validation_with_citation(citation)
    candidate = _candidate("Brown v. Board of Education")

    node = run_exact_case_name_check(validation, candidate=candidate)

    assert node.status is ValidationNodeStatus.SKIPPED
    assert node.extracted_case_name is None
    assert node.outcome is FieldCheckOutcome.UNAVAILABLE
