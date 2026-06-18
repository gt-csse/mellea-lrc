import pytest

from mellea_lrc.assessment import (
    CaseNameAssessmentStatus,
    assess_case_name_exact_match,
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
