import pytest

from mellea_lrc.assessment import (
    CaseNameAssessmentStatus,
    CitationAssessment,
    CitationAssessmentStatus,
    ModifiedExtractedCitation,
    YearAssessmentStatus,
    assess_case_name_exact_match,
    assess_year_exact_match,
    build_extracted_case_name,
    get_extended_span_text,
)
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span


def test_get_extended_span_text_includes_context_around_full_span() -> None:
    text = "Before context. Brown v. Board, 347 U.S. 483 (1954). After context."
    full_span = Span(start=text.index("Brown"), end=text.index(". After"))

    extended = get_extended_span_text(text, full_span, before_chars=9, after_chars=7)

    assert extended == "context. Brown v. Board, 347 U.S. 483 (1954). After"


def test_get_extended_span_text_clamps_to_document_boundaries() -> None:
    text = "Brown v. Board, 347 U.S. 483 (1954)."
    full_span = Span(start=0, end=len(text))

    extended = get_extended_span_text(text, full_span, before_chars=50, after_chars=50)

    assert extended == text


def test_get_extended_span_text_rejects_span_past_text_end() -> None:
    text = "Short text"
    full_span = Span(start=0, end=len(text) + 1)

    with pytest.raises(ValueError, match="exceeds text length"):
        get_extended_span_text(text, full_span)


def test_get_extended_span_text_rejects_negative_context_windows() -> None:
    text = "Brown v. Board"
    full_span = Span(start=0, end=len(text))

    with pytest.raises(ValueError, match="must be non-negative"):
        get_extended_span_text(text, full_span, before_chars=-1)


def test_build_extracted_case_name_from_parties() -> None:
    citation = FullCaseCitation(plaintiff="Brown", defendant="Board")

    assert build_extracted_case_name(citation) == "Brown v. Board"


def test_assess_case_name_exact_match_passes_without_semantic_check() -> None:
    result = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board",
    )

    assert result.status == CaseNameAssessmentStatus.EXACT_MATCH


def test_assess_case_name_mismatch_needs_semantic_check() -> None:
    result = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board of Education",
    )

    assert result.status == CaseNameAssessmentStatus.NEEDS_SEMANTIC_ASSESSMENT


def test_assess_case_name_missing_name_is_single_extraction_error_type() -> None:
    result = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name=None,
        courtlistener_case_name="Brown v. Board",
    )

    assert result.status == CaseNameAssessmentStatus.EXTRACTION_ERROR


def test_assess_year_exact_match_uses_string_equality() -> None:
    result = assess_year_exact_match(
        citation_id="cite-1",
        extracted_year="1954",
        courtlistener_year="1954",
    )

    assert result.status == YearAssessmentStatus.EXACT_MATCH


def test_assess_year_mismatch_is_deterministic_error() -> None:
    result = assess_year_exact_match(
        citation_id="cite-1",
        extracted_year="1953",
        courtlistener_year="1954",
    )

    assert result.status == YearAssessmentStatus.MISMATCH


def test_citation_assessment_status_is_derived_from_sub_assessments() -> None:
    case_assess = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board",
    )
    year_assess = assess_year_exact_match(
        citation_id="cite-1",
        extracted_year="1953",
        courtlistener_year="1954",
    )

    result = CitationAssessment(
        citation_id="cite-1",
        case_assess=case_assess,
        year_assess=year_assess,
    )

    assert result.status == CitationAssessmentStatus.EXTRACTION_ERROR
    assert result.message == "Extracted year does not match CourtListener."


def test_modified_extracted_citation_valid_requires_grounded_fields() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    modified = ModifiedExtractedCitation(
        plaintiff="World Wide Ass'n of Specialty Programs",
        defendant="Pure, Inc.",
    )

    assert modified.valid(context)
    assert modified.extracted_case_name == "World Wide Ass'n of Specialty Programs v. Pure, Inc."


def test_modified_extracted_citation_valid_rejects_ungrounded_fields() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    modified = ModifiedExtractedCitation(
        plaintiff="World Wide Association of Specialty Programs",
        defendant="Pure, Inc.",
    )

    assert not modified.valid(context)
