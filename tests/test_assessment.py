import pytest

from mellea_lrc.assessment import (
    AssessmentSkipReason,
    AssessedCitationAssessment,
    CaseNameAssessmentStatus,
    FailedCitationAssessment,
    ModifiedExtractedCitationProposal,
    SkippedCitationAssessment,
    WaitingCitationAssessment,
    YearAssessmentStatus,
    assess_case_name_exact_match,
    assess_year_exact_match,
    build_extracted_case_name,
    find_text_span_near_full_span,
    get_extended_span_text,
    initialize_assessment,
    run_assessment,
)
from mellea_lrc.assessment import ReextractionStatus, validate_proposal
from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import ExtractedCitation
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessedDocumentMetadata,
    PreprocessingBackend,
    SourceFormat,
)
from mellea_lrc.validation.types import CitationValidation, ValidatedDocument, ValidationStatus


def test_get_extended_span_text_includes_context_around_full_span() -> None:
    text = "Before context. Brown v. Board, 347 U.S. 483 (1954). After context."
    full_span = Span(start=text.index("Brown"), end=text.index(". After"))

    extended = get_extended_span_text(text, full_span, before_chars=9, after_chars=7)

    assert extended == "context. Brown v. Board, 347 U.S. 483 (1954). After"


def test_get_extended_span_text_defaults_to_two_hundred_chars_each_side() -> None:
    before = "b" * 210
    citation = "Brown v. Board, 347 U.S. 483 (1954)"
    after = "a" * 210
    text = before + citation + after
    full_span = Span(start=len(before), end=len(before) + len(citation))

    extended = get_extended_span_text(text, full_span)

    assert extended == before[-200:] + citation + after[:200]


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


def test_find_text_span_near_full_span_disambiguates_repeated_case_name() -> None:
    text = (
        "Brown v. Board appears in background. "
        "Later, See Brown v. Board, 347 U.S. 483 (1954)."
    )
    full_span = Span(start=text.index("Brown v. Board, 347"), end=text.index(" (1954)") + len(" (1954)"))

    span = find_text_span_near_full_span(text, "Brown v. Board", full_span)

    assert span == Span(start=text.index("Brown v. Board, 347"), end=text.index(", 347"))


def test_find_text_span_near_full_span_matches_flexible_whitespace() -> None:
    text = "See World Wide Ass'n\nof Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    full_span = Span(start=text.index("World"), end=text.index("."))

    span = find_text_span_near_full_span(
        text,
        "World Wide Ass'n of Specialty Programs v. Pure, Inc.",
        full_span,
    )

    assert span == Span(start=text.index("World"), end=text.index(", 450"))


def test_find_text_span_near_full_span_returns_none_outside_local_context() -> None:
    text = "Brown v. Board is in background. See 347 U.S. 483 (1954)."
    full_span = Span(start=text.index("347"), end=len(text) - 1)

    assert find_text_span_near_full_span(text, "Brown v. Board", full_span, before_chars=20) is None


def test_build_extracted_case_name_from_parties() -> None:
    citation = FullCaseCitation(plaintiff="Brown", defendant="Board")

    assert build_extracted_case_name(citation) == "Brown v. Board"


def test_build_extracted_case_name_is_missing_without_parties() -> None:
    citation = FullCaseCitation(volume="347", reporter="U.S.", page="483")

    assert build_extracted_case_name(citation) is None


def test_assess_case_name_exact_match_passes_without_semantic_check() -> None:
    result = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board",
    )

    assert result.status == CaseNameAssessmentStatus.EXACT_MATCH


def test_assess_case_name_mismatch_needs_assessment() -> None:
    result = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board of Education",
    )

    assert result.status == CaseNameAssessmentStatus.NEEDS_ASSESSMENT


def test_assess_case_name_missing_name_needs_assessment() -> None:
    result = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name=None,
        courtlistener_case_name="Brown v. Board",
    )

    assert result.status == CaseNameAssessmentStatus.NEEDS_ASSESSMENT


def test_assess_case_name_exact_match_ignores_typographic_apostrophe() -> None:
    result = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name="World Wide Ass’n of Specialty Programs v. Pure, Inc.",
        courtlistener_case_name="World Wide Ass'n of Specialty Programs v. Pure, Inc.",
    )

    assert result.status == CaseNameAssessmentStatus.EXACT_MATCH


def test_assess_case_name_exact_match_ignores_collapsible_whitespace() -> None:
    result = assess_case_name_exact_match(
        citation_id="cite-1",
        extracted_case_name="Brown v.\n\n Board",
        courtlistener_case_name="Brown v. Board",
    )

    assert result.status == CaseNameAssessmentStatus.EXACT_MATCH


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


def test_assess_year_missing_is_field_level_third_status() -> None:
    result = assess_year_exact_match(
        citation_id="cite-1",
        extracted_year=None,
        courtlistener_year="1954",
    )

    assert result.status == YearAssessmentStatus.MISSING


def test_modified_extracted_citation_proposal_valid_requires_grounded_fields() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    modified = ModifiedExtractedCitationProposal(
        case_name="World Wide Ass'n of Specialty Programs v. Pure, Inc.",
    )

    assert modified.valid(context)
    assert modified.extracted_case_name == "World Wide Ass'n of Specialty Programs v. Pure, Inc."


def test_modified_extracted_citation_proposal_valid_rejects_ungrounded_fields() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    modified = ModifiedExtractedCitationProposal(
        case_name="World Wide Association of Specialty Programs v. Pure, Inc.",
    )

    assert not modified.valid(context)


def test_reextraction_validation_accepts_grounded_proposal() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    proposal = ModifiedExtractedCitationProposal(
        case_name="World Wide Ass'n of Specialty Programs v. Pure, Inc.",
    )

    status, error = validate_proposal(proposal, context)

    assert status == ReextractionStatus.ACCEPTED
    assert error is None


def test_reextraction_validation_reports_ungrounded_fields() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    proposal = ModifiedExtractedCitationProposal(
        case_name="World Wide Association of Specialty Programs v. Pure, Inc.",
    )

    status, error = validate_proposal(proposal, context)

    assert status == ReextractionStatus.INVALID
    assert error is not None
    assert "case_name" in error


def test_reextraction_validation_distinguishes_empty_proposal() -> None:
    status, error = validate_proposal(ModifiedExtractedCitationProposal(), "Context")

    assert status == ReextractionStatus.EMPTY
    assert error is None


def test_run_assessment_progresses_document_validation_to_document_assessment() -> None:
    preprocessed = PreprocessedDocument(
        text="Brown v. Board, 347 U.S. 483 (1954).",
        metadata=PreprocessedDocumentMetadata(
            source_path="test.txt",
            source_format=SourceFormat.TEXT,
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
    )
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            year="1954",
        ),
    )
    validation = ValidatedDocument(
        metadata=preprocessed.metadata,
        text=preprocessed.text,
        citations=(citation,),
        validations=(
            CitationValidation(
                citation_id="cite-1",
                locator="347 U.S. 483",
                status=ValidationStatus.FOUND,
                source="test",
                message="found",
                case_names=("Brown v. Board",),
                clusters=({"case_name": "Brown v. Board", "date_filed": "1954-05-17"},),
            ),
        ),
    )

    assessment = run_assessment(validation)

    assert assessment.text == preprocessed.text
    assert assessment.metadata == preprocessed.metadata
    assert assessment.citations == (citation,)
    assert assessment.validations == validation.validations
    assert assessment.assessment_complete is True
    assert len(assessment.assessments) == 1
    record = assessment.assessments[0]
    assert isinstance(record, AssessedCitationAssessment)
    assert record.result.case_assess.status == CaseNameAssessmentStatus.EXACT_MATCH
    assert record.result.year_assess.status == YearAssessmentStatus.EXACT_MATCH


def test_initialize_assessment_marks_eligible_citation_waiting() -> None:
    preprocessed = PreprocessedDocument(
        text="Brown v. Board, 347 U.S. 483 (1954).",
        metadata=PreprocessedDocumentMetadata(),
    )
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            year="1954",
        ),
    )
    validation = ValidatedDocument(
        metadata=preprocessed.metadata,
        text=preprocessed.text,
        citations=(citation,),
        validations=(
            CitationValidation(
                citation_id="cite-1",
                locator="347 U.S. 483",
                status=ValidationStatus.FOUND,
                source="test",
                message="found",
                clusters=({"case_name": "Different v. Case", "date_filed": "1954-05-17"},),
            ),
        ),
    )

    assessment = initialize_assessment(validation)

    assert assessment.assessment_complete is False
    assert assessment.assessments == (WaitingCitationAssessment(citation_id="cite-1"),)


def test_initialize_assessment_marks_unsupported_citation_skipped() -> None:
    text = "See 28 U.S.C. § 636."
    preprocessed = PreprocessedDocument(text=text, metadata=PreprocessedDocumentMetadata())
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(4, 20),
        matched_text="28 U.S.C. § 636",
        citation=FullLawCitation(volume="28", reporter="U.S.C.", page="636"),
    )
    validation = ValidatedDocument(
        metadata=preprocessed.metadata,
        text=preprocessed.text,
        citations=(citation,),
        validations=(
            CitationValidation(
                citation_id="cite-1",
                locator=None,
                status=ValidationStatus.SKIPPED,
                source="test",
                message="unsupported",
            ),
        ),
    )

    assessment = initialize_assessment(validation)

    assert assessment.assessment_complete is True
    assert assessment.assessments == (
        SkippedCitationAssessment(
            citation_id="cite-1",
            reason=AssessmentSkipReason.UNSUPPORTED_CITATION_KIND,
            message="Citation kind FullLawCitation is not assessed.",
        ),
    )


def test_run_assessment_records_per_citation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_assessment(**_kwargs):
        msg = "assessment service unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr("mellea_lrc.assessment.pipeline.assess_found_citation", fail_assessment)
    text = "Brown v. Board, 347 U.S. 483 (1954)."
    preprocessed = PreprocessedDocument(text=text, metadata=PreprocessedDocumentMetadata())
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            year="1954",
        ),
    )
    validation = ValidatedDocument(
        metadata=preprocessed.metadata,
        text=preprocessed.text,
        citations=(citation,),
        validations=(
            CitationValidation(
                citation_id="cite-1",
                locator="347 U.S. 483",
                status=ValidationStatus.FOUND,
                source="test",
                message="found",
                clusters=({"case_name": "Brown v. Board", "date_filed": "1954-05-17"},),
            ),
        ),
    )

    assessment = run_assessment(validation)

    assert assessment.assessment_complete is True
    assert assessment.assessments == (
        FailedCitationAssessment(
            citation_id="cite-1",
            error="RuntimeError: assessment service unavailable",
        ),
    )
