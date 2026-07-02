import asyncio

import pytest

from mellea_lrc.assessment import (
    AmbiguousCitationAssessment,
    AssessmentSkipReason,
    AssessedCitationAssessment,
    CandidateAssessment,
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameProposal,
    CaseNameReassessed,
    CaseNameReassessmentFailed,
    CaseNameReassessmentNotRequired,
    CaseNameReextractionFailed,
    CaseNameAssessmentStatus,
    CourtAssessment,
    CourtAssessmentRun,
    CourtAssessmentStatus,
    CourtFollowupNotRequired,
    CourtInferredFromReporter,
    FailedCitationAssessment,
    DocumentTextWindow,
    ReextractedCaseName,
    SkippedCitationAssessment,
    WaitingCitationAssessment,
    YearAssessmentStatus,
    assess_case_name_exact_match,
    assess_case_name_with_mellea,
    assess_court,
    assess_court_exact_match,
    assess_found_citation,
    assess_year_exact_match,
    build_extracted_case_name,
    find_text_span_near_full_span,
    get_extended_span_text,
    initialize_assessment,
    run_assessment,
)
from mellea_lrc.assessment import ReextractionStatus, validate_proposal
from mellea_lrc.assessment.fields.case_name.reextract import ReextractionResult
from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.types import CitationMatch
from mellea_lrc.extraction.types import ExtractedCitation, ExtractionMetadata
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
)
from mellea_lrc.validation.types import (
    AmbiguousCitationValidation,
    CitationValidation,
    CourtResolutionSource,
    CourtResolutionTrace,
    FoundCitationValidation,
    SkippedCitationValidation,
    ValidatedDocument,
    ValidationMetadata,
    ValidationStatus,
)


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
    text = "Brown v. Board appears in background. Later, See Brown v. Board, 347 U.S. 483 (1954)."
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
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board",
    )

    assert result.status == CaseNameAssessmentStatus.EXACT_MATCH


def test_assess_case_name_mismatch_requires_mellea() -> None:
    result = assess_case_name_exact_match(
        extracted_case_name="Brown v. Board",
        courtlistener_case_name="Brown v. Board of Education",
    )

    assert result is None


def test_assess_case_name_missing_extracted_name_requires_mellea() -> None:
    result = assess_case_name_exact_match(
        extracted_case_name=None,
        courtlistener_case_name="Brown v. Board",
    )

    assert result is None


def test_assess_case_name_without_courtlistener_name_is_unassessable() -> None:
    result = assess_case_name_exact_match(
        extracted_case_name="Brown v. Board",
        courtlistener_case_name=None,
    )

    assert result is not None
    assert result.status == CaseNameAssessmentStatus.UNASSESSABLE


def test_assess_case_name_exact_match_ignores_typographic_apostrophe() -> None:
    result = assess_case_name_exact_match(
        extracted_case_name="World Wide Ass’n of Specialty Programs v. Pure, Inc.",
        courtlistener_case_name="World Wide Ass'n of Specialty Programs v. Pure, Inc.",
    )

    assert result.status == CaseNameAssessmentStatus.EXACT_MATCH


def test_assess_case_name_exact_match_ignores_collapsible_whitespace() -> None:
    result = assess_case_name_exact_match(
        extracted_case_name="Brown v.\n\n Board",
        courtlistener_case_name="Brown v. Board",
    )

    assert result.status == CaseNameAssessmentStatus.EXACT_MATCH


def test_assess_year_exact_match_uses_string_equality() -> None:
    result = assess_year_exact_match(
        extracted_year="1954",
        courtlistener_year="1954",
    )

    assert result.status == YearAssessmentStatus.EXACT_MATCH


def test_assess_year_mismatch_is_deterministic_error() -> None:
    result = assess_year_exact_match(
        extracted_year="1953",
        courtlistener_year="1954",
    )

    assert result.status == YearAssessmentStatus.MISMATCH


def test_assess_year_missing_is_field_level_third_status() -> None:
    result = assess_year_exact_match(
        extracted_year=None,
        courtlistener_year="1954",
    )

    assert result.status == YearAssessmentStatus.MISSING


def test_assess_court_exact_match_uses_string_equality() -> None:
    result = assess_court_exact_match(
        extracted_court="scotus",
        courtlistener_court_id="scotus",
    )

    assert result.status == CourtAssessmentStatus.EXACT_MATCH


def test_assess_court_mismatch_is_deterministic_error() -> None:
    result = assess_court_exact_match(
        extracted_court="ca9",
        courtlistener_court_id="ca8",
    )

    assert result.status == CourtAssessmentStatus.MISMATCH


def test_assess_court_missing_is_field_level_third_status() -> None:
    result = assess_court_exact_match(
        extracted_court=None,
        courtlistener_court_id="scotus",
    )

    assert result.status == CourtAssessmentStatus.MISSING


def test_assess_court_applies_reporter_inference_in_followup() -> None:
    result = assess_court(
        extracted_court=None,
        courtlistener_court_id="scotus",
        reporter="L. Ed. 2d",
    )

    assert result.initial.status == CourtAssessmentStatus.MISSING
    assert isinstance(result.followup, CourtInferredFromReporter)
    assert result.followup.reporter == "L. Ed. 2d"
    assert result.followup.result.status == CourtAssessmentStatus.EXACT_MATCH
    # The terminal verdict is the reporter-inference reassessment, not the
    # initial missing comparison.
    assert result.final is result.followup.result
    assert result.final.status == CourtAssessmentStatus.EXACT_MATCH


def test_assess_court_skips_inference_when_extracted_court_is_present() -> None:
    result = assess_court(
        extracted_court="scotus",
        courtlistener_court_id="scotus",
        reporter="U.S.",
    )

    assert result.initial.status == CourtAssessmentStatus.EXACT_MATCH
    assert isinstance(result.followup, CourtFollowupNotRequired)
    # With no follow-up, the initial comparison is the terminal verdict.
    assert result.final is result.initial


def test_case_name_proposal_valid_requires_grounding() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    modified = CaseNameProposal(
        case_name="World Wide Ass'n of Specialty Programs v. Pure, Inc.",
    )

    assert modified.valid(context)
    assert modified.case_name == "World Wide Ass'n of Specialty Programs v. Pure, Inc."


def test_case_name_proposal_rejects_ungrounded_value() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    modified = CaseNameProposal(
        case_name="World Wide Association of Specialty Programs v. Pure, Inc.",
    )

    assert not modified.valid(context)


def test_reextraction_validation_accepts_grounded_proposal() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    proposal = CaseNameProposal(
        case_name="World Wide Ass'n of Specialty Programs v. Pure, Inc.",
    )

    status, error = validate_proposal(proposal, context)

    assert status == ReextractionStatus.ACCEPTED
    assert error is None


def test_reextraction_validation_reports_ungrounded_fields() -> None:
    context = "See World Wide Ass'n of Specialty Programs v. Pure, Inc., 450 F.3d 1132."
    proposal = CaseNameProposal(
        case_name="World Wide Association of Specialty Programs v. Pure, Inc.",
    )

    status, error = validate_proposal(proposal, context)

    assert status == ReextractionStatus.INVALID
    assert error is not None
    assert "case_name" in error


def test_case_name_proposal_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CaseNameProposal(case_name="")


def test_case_name_followup_preserves_reextracted_value_when_reassessment_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reextracted = ReextractedCaseName(
        case_name="Brown v. Board of Education",
        case_name_span=Span(0, 27),
    )

    async def fail_after_reextraction(*_args, **_kwargs):
        return CaseNameAssessmentRun(
            initial=CaseNameAssessment(
                status=CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
                extracted_case_name="Brown v. Board",
                courtlistener_case_name="Brown v. Board of Education",
                message="re-extraction attempted",
            ),
            followup=CaseNameReassessmentFailed(
                reextracted_case_name=reextracted,
                error="RuntimeError: reassessment unavailable",
            ),
        )

    monkeypatch.setattr(
        "mellea_lrc.assessment.citation.assess.assess_case_name_with_mellea",
        fail_after_reextraction,
    )
    result = asyncio.run(
        assess_found_citation(
            document_text="Brown v. Board of Education, 347 U.S. 483 (1954).",
            span=Span(0, 49),
            extracted_case_name="Brown v. Board",
            courtlistener_case_name="Brown v. Board of Education",
            extracted_year="1954",
            courtlistener_year="1954",
            session=object(),
        )
    )

    followup = result.case_name.followup
    assert isinstance(followup, CaseNameReassessmentFailed)
    assert followup.reextracted_case_name.case_name == "Brown v. Board of Education"
    assert followup.error == "RuntimeError: reassessment unavailable"


def test_case_name_followup_records_reextraction_failure_without_grounded_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_reextraction(*_args, **_kwargs):
        return CaseNameAssessmentRun(
            initial=CaseNameAssessment(
                status=CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
                extracted_case_name="Brown v. Board",
                courtlistener_case_name="Brown v. Board of Education",
                message="re-extraction failed",
            ),
            followup=CaseNameReextractionFailed(error="invalid proposal"),
        )

    monkeypatch.setattr(
        "mellea_lrc.assessment.citation.assess.assess_case_name_with_mellea",
        fail_reextraction,
    )
    result = asyncio.run(
        assess_found_citation(
            document_text="Brown v. Board, 347 U.S. 483 (1954).",
            span=Span(0, 36),
            extracted_case_name="Brown v. Board",
            courtlistener_case_name="Brown v. Board of Education",
            extracted_year="1954",
            courtlistener_year="1954",
            session=object(),
        )
    )

    assert result.case_name.followup == CaseNameReextractionFailed(error="invalid proposal")


def test_case_name_run_captures_reassessment_failure_after_accepted_reextraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.structured_model_options",
        lambda **_kwargs: {},
    )

    async def no_semantic_match(*_args, **_kwargs):
        return "not_semantic_match"

    async def accepted_reextraction(*_args, **_kwargs):
        return ReextractionResult(
            status=ReextractionStatus.ACCEPTED,
            proposal=CaseNameProposal(case_name="Brown v. Board of Education"),
        )

    async def fail_reassessment(*_args, **_kwargs):
        msg = "classifier unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.semantic_match_case_name",
        no_semantic_match,
    )
    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.reextract_case_name",
        accepted_reextraction,
    )
    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess._assess_reextracted_case_name",
        fail_reassessment,
    )

    run = asyncio.run(
        assess_case_name_with_mellea(
            object(),
            extracted_case_name="Brown v. Board",
            courtlistener_case_name="Brown v. Board of Education",
            document_context=DocumentTextWindow.around(
                "Brown v. Board of Education",
                Span(0, 27),
            ),
        )
    )

    assert isinstance(run.followup, CaseNameReassessmentFailed)
    assert run.followup.reextracted_case_name.case_name == "Brown v. Board of Education"
    assert run.followup.error == "RuntimeError: classifier unavailable"


def test_case_name_run_uses_native_semantic_classifier_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.structured_model_options",
        lambda **_kwargs: {},
    )

    async def semantic_match(*_args, **kwargs):
        calls.append(kwargs)
        return "semantic_match"

    async def unexpected_reextraction(*_args, **_kwargs):
        msg = "re-extraction should not run after a semantic match"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.semantic_match_case_name",
        semantic_match,
    )
    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.reextract_case_name",
        unexpected_reextraction,
    )

    run = asyncio.run(
        assess_case_name_with_mellea(
            object(),
            extracted_case_name="Brown v. Board",
            courtlistener_case_name="Brown v. Board of Education",
            document_context=DocumentTextWindow.around(
                "Brown v. Board, 347 U.S. 483",
                Span(0, 14),
            ),
        )
    )

    assert run.initial.status == CaseNameAssessmentStatus.SEMANTIC_MATCH
    assert isinstance(run.followup, CaseNameReassessmentNotRequired)
    assert len(calls) == 1
    assert "model_options" in calls[0]
    assert "strategy" not in calls[0]


def test_case_name_run_uses_native_post_reextraction_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_calls = 0
    non_semantic_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.structured_model_options",
        lambda **_kwargs: {},
    )

    async def no_semantic_match(*_args, **_kwargs):
        nonlocal semantic_calls
        semantic_calls += 1
        return "not_semantic_match"

    async def accepted_reextraction(*_args, **_kwargs):
        return ReextractionResult(
            status=ReextractionStatus.ACCEPTED,
            proposal=CaseNameProposal(case_name="Brown Board"),
        )

    async def irregular_form(*_args, **kwargs):
        non_semantic_calls.append(kwargs)
        return "irregular_form"

    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.semantic_match_case_name",
        no_semantic_match,
    )
    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.reextract_case_name",
        accepted_reextraction,
    )
    monkeypatch.setattr(
        "mellea_lrc.assessment.fields.case_name.assess.classify_non_semantic_case_name",
        irregular_form,
    )

    run = asyncio.run(
        assess_case_name_with_mellea(
            object(),
            extracted_case_name="Brown v. Board",
            courtlistener_case_name="Brown v. Board of Education",
            document_context=DocumentTextWindow.around(
                "Brown Board, 347 U.S. 483",
                Span(0, 11),
            ),
        )
    )

    assert isinstance(run.followup, CaseNameReassessed)
    assert run.followup.result.status == CaseNameAssessmentStatus.IRREGULAR_FORM
    assert semantic_calls == 2
    assert len(non_semantic_calls) == 1
    assert "model_options" in non_semantic_calls[0]
    assert "strategy" not in non_semantic_calls[0]


def test_run_assessment_progresses_document_validation_to_document_assessment() -> None:
    preprocessed = PreprocessedDocument(
        source_metadata=SourceMetadata(path="test.txt", format=SourceFormat.TEXT),
        text="Brown v. Board, 347 U.S. 483 (1954).",
        preprocessing_metadata=PreprocessingMetadata(
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
            court="scotus",
        ),
    )
    validation = ValidatedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(citation,),
        extraction_metadata=ExtractionMetadata(),
        validations=(
            FoundCitationValidation(
                citation_id="cite-1",
                locator="347 U.S. 483",
                source="test",
                lookup_status=200,
                lookup_cache=None,
                lookup_key=None,
                matches=(
                    CitationMatch(
                        case_name="Brown v. Board",
                        date_filed="1954-05-17",
                        court_id="scotus",
                    ),
                ),
                court_resolution=CourtResolutionTrace(
                    courtlistener_court_id="scotus",
                    resolved_via=CourtResolutionSource.CLUSTER_PROVIDED,
                    docket_id=None,
                    docket_url=None,
                    cached=False,
                    error_message=None,
                ),
                extra_data=ExtraData(),
            ),
        ),
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
    )

    assessment = run_assessment(validation)

    assert assessment.text == preprocessed.text
    assert assessment.source_metadata == preprocessed.source_metadata
    assert assessment.preprocessing_metadata == preprocessed.preprocessing_metadata
    assert assessment.citations == (citation,)
    assert assessment.validations == validation.validations
    assert assessment.assessment_complete is True
    assert len(assessment.assessments) == 1
    record = assessment.assessments[0]
    assert isinstance(record, AssessedCitationAssessment)
    assert record.result.case_name.initial.status == CaseNameAssessmentStatus.EXACT_MATCH
    assert isinstance(record.result.case_name.followup, CaseNameReassessmentNotRequired)
    assert record.result.court.initial.status == CourtAssessmentStatus.EXACT_MATCH
    assert isinstance(record.result.court.followup, CourtFollowupNotRequired)
    assert record.result.year.status == YearAssessmentStatus.EXACT_MATCH


def _ambiguous_validation(
    citation: ExtractedCitation,
    matches: tuple[CitationMatch, ...],
) -> ValidatedDocument:
    preprocessed = PreprocessedDocument(
        source_metadata=SourceMetadata(),
        text="Doe v. Roe, 1 F.3d 2 (2001).",
        preprocessing_metadata=PreprocessingMetadata(backend=PreprocessingBackend.PLAIN_TEXT),
    )
    return ValidatedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(citation,),
        extraction_metadata=ExtractionMetadata(),
        validations=(
            AmbiguousCitationValidation(
                citation_id=citation.citation_id,
                locator="1 F.3d 2",
                source="test",
                lookup_status=300,
                lookup_cache=None,
                lookup_key=None,
                matches=matches,
            ),
        ),
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
    )


def test_run_assessment_assesses_each_ambiguous_candidate() -> None:
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 10),
        matched_text="1 F.3d 2",
        citation=FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="1", reporter="F.3d", page="2"),
    )
    # Both candidates take the deterministic (no-Mellea) case-name path: one exact,
    # one with no CourtListener name (unassessable).
    validation = _ambiguous_validation(
        citation,
        (
            CitationMatch(case_name="Doe v. Roe", date_filed="2001-01-01", docket_id="11"),
            CitationMatch(case_name=None, date_filed="2001-01-01", docket_id="22"),
        ),
    )

    assessment = run_assessment(validation)

    record = assessment.assessments[0]
    assert isinstance(record, AmbiguousCitationAssessment)
    assert record.gated is False
    assert len(record.candidates) == 2
    assert all(isinstance(c, CandidateAssessment) for c in record.candidates)
    assert record.candidates[0].match.docket_id == "11"
    assert record.candidates[0].result.case_name.initial.status == CaseNameAssessmentStatus.EXACT_MATCH
    assert record.candidates[1].match.docket_id == "22"
    assert record.candidates[1].result.case_name.initial.status == CaseNameAssessmentStatus.UNASSESSABLE
    assert assessment.assessment_complete is True


def test_run_assessment_gates_ambiguous_beyond_candidate_limit() -> None:
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 10),
        matched_text="1 F.3d 2",
        citation=FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="1", reporter="F.3d", page="2"),
    )
    matches = tuple(CitationMatch(case_name=f"Case {i}", docket_id=str(i)) for i in range(6))
    validation = _ambiguous_validation(citation, matches)

    record = run_assessment(validation).assessments[0]

    assert isinstance(record, AmbiguousCitationAssessment)
    assert record.gated is True
    assert record.candidates == ()
    assert "6 candidates" in record.message


def test_initialize_assessment_marks_eligible_citation_waiting() -> None:
    preprocessed = PreprocessedDocument(
        source_metadata=SourceMetadata(),
        text="Brown v. Board, 347 U.S. 483 (1954).",
        preprocessing_metadata=PreprocessingMetadata(),
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
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(citation,),
        extraction_metadata=ExtractionMetadata(),
        validations=(
            FoundCitationValidation(
                citation_id="cite-1",
                locator="347 U.S. 483",
                source="test",
                lookup_status=200,
                lookup_cache=None,
                lookup_key=None,
                matches=(
                    CitationMatch(
                        case_name="Different v. Case",
                        date_filed="1954-05-17",
                    ),
                ),
                court_resolution=CourtResolutionTrace(
                    courtlistener_court_id=None,
                    resolved_via=CourtResolutionSource.NOT_ATTEMPTED,
                    docket_id=None,
                    docket_url=None,
                    cached=False,
                    error_message=None,
                ),
                extra_data=ExtraData(),
            ),
        ),
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
    )

    assessment = initialize_assessment(validation)

    assert assessment.assessment_complete is False
    assert assessment.assessments == (WaitingCitationAssessment(citation_id="cite-1"),)


def test_initialize_assessment_marks_unsupported_citation_skipped() -> None:
    text = "See 28 U.S.C. § 636."
    preprocessed = PreprocessedDocument(
        source_metadata=SourceMetadata(),
        text=text,
        preprocessing_metadata=PreprocessingMetadata(),
    )
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(4, 20),
        matched_text="28 U.S.C. § 636",
        citation=FullLawCitation(volume="28", reporter="U.S.C.", page="636"),
    )
    validation = ValidatedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(citation,),
        extraction_metadata=ExtractionMetadata(),
        validations=(
            SkippedCitationValidation(
                citation_id="cite-1",
                source="test",
            ),
        ),
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
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

    monkeypatch.setattr(
        "mellea_lrc.assessment.document.pipeline.assess_found_citation",
        fail_assessment,
    )
    text = "Brown v. Board, 347 U.S. 483 (1954)."
    preprocessed = PreprocessedDocument(
        source_metadata=SourceMetadata(),
        text=text,
        preprocessing_metadata=PreprocessingMetadata(),
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
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(citation,),
        extraction_metadata=ExtractionMetadata(),
        validations=(
            FoundCitationValidation(
                citation_id="cite-1",
                locator="347 U.S. 483",
                source="test",
                lookup_status=200,
                lookup_cache=None,
                lookup_key=None,
                matches=(
                    CitationMatch(
                        case_name="Brown v. Board",
                        date_filed="1954-05-17",
                    ),
                ),
                court_resolution=CourtResolutionTrace(
                    courtlistener_court_id=None,
                    resolved_via=CourtResolutionSource.NOT_ATTEMPTED,
                    docket_id=None,
                    docket_url=None,
                    cached=False,
                    error_message=None,
                ),
                extra_data=ExtraData(),
            ),
        ),
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
    )

    assessment = run_assessment(validation)

    assert assessment.assessment_complete is True
    assert assessment.assessments == (
        FailedCitationAssessment(
            citation_id="cite-1",
            error="RuntimeError: assessment service unavailable",
        ),
    )
