"""Tests for the standalone E2E backend pipeline helpers."""

import asyncio
from mellea_lrc.assessment import (
    AssessmentMetadata,
    AssessedCitationAssessment,
    AssessedDocument,
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    CaseNameReassessed,
    CaseNameReassessmentNotRequired,
    CitationAssessment,
    CitationAssessmentResult,
    CourtAssessment,
    CourtAssessmentStatus,
    ReextractedCaseName,
    WaitingCitationAssessment,
    YearAssessment,
    YearAssessmentStatus,
)
from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.types import CitationMatch, CourtListenerCitationLookup
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import PreprocessedDocument, preprocess_plain_text_from_string
from mellea_lrc.serialization import (
    serialize_assessed_document,
    serialize_extracted_document,
    serialize_validated_document,
    serialize_preprocessed_document,
)
from mellea_lrc.llm import llm_api_config_from_env
from mellea_lrc.validation.types import (
    CitationValidation,
    ValidatedDocument,
    ValidationMetadata,
    ValidationStatus,
)
from scripts.e2e_backend.api import _review_snapshot_payload
from scripts.e2e_backend.pipeline import (
    E2EBackend,
    assess_review_payload,
    review_document_assessment,
    review_preprocessed,
    validate_review_citation_payload,
    validate_review_payload,
)


class FakeClient:
    def lookup_citation(self, volume: str, reporter: str, page: str):
        assert (volume, reporter, page) == ("347", "U.S.", "483")
        return CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            matches=(
                CitationMatch(
                    case_name="Brown v. Board of Education",
                    date_filed="1954-05-17",
                    court="scotus",
                ),
            ),
            cache="miss",
            key="lookup-key",
        )


def _court_assessment() -> CourtAssessment:
    return CourtAssessment(
        status=CourtAssessmentStatus.EXACT_MATCH,
        extracted_court="scotus",
        courtlistener_court_id="scotus",
        message="match",
    )


def _extracted_document(
    *,
    preprocessed: PreprocessedDocument,
    citations: tuple[ExtractedCitation, ...],
) -> ExtractedDocument:
    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
    )


def _validated_document(
    *,
    preprocessed: PreprocessedDocument,
    citations: tuple[ExtractedCitation, ...],
    validations: tuple[CitationValidation, ...],
) -> ValidatedDocument:
    return ValidatedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
        validations=validations,
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
    )


def _assessed_document(
    *,
    preprocessed: PreprocessedDocument,
    citations: tuple[ExtractedCitation, ...],
    validations: tuple[CitationValidation, ...],
    assessments: tuple[CitationAssessment, ...],
) -> AssessedDocument:
    return AssessedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
        validations=validations,
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
        assessments=assessments,
        assessment_metadata=AssessmentMetadata(),
    )


def test_review_preprocessed_returns_frontend_span_payload() -> None:
    output = review_preprocessed(
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        client=FakeClient(),
    )

    citation = output["citations"][0]
    assert output["document"]["text"] == "Brown v. Board, 347 U.S. 483."
    assert citation["start"] == 0
    assert citation["end"] == 28
    assert citation["matched_text"] == "347 U.S. 483"
    assert citation["kind"] == "FullCaseCitation"
    assert citation["fields"]["volume"] == "347"
    assert citation["fields"]["reporter"] == "U.S."
    assert citation["fields"]["page"] == "483"
    assert citation["fields"]["plaintiff"] == "Brown"
    assert citation["validation"]["status"] == "found"
    assert citation["validation"]["case_names"] == ["Brown v. Board of Education"]
    assert citation["validation"]["lookup_status"] == 200
    assert citation["validation"]["lookup_cache"] == "miss"
    assert citation["validation"]["lookup_key"] == "lookup-key"
    assert citation["validation"]["matches"][0]["date_filed"] == "1954-05-17"
    assert output["stats"]["found"] == 1


def test_validate_review_payload_reuses_existing_extraction_payload() -> None:
    extracted = review_preprocessed(
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        validate=False,
    )

    output = validate_review_payload(extracted, client=FakeClient())

    citation = output["citations"][0]
    assert citation["start"] == extracted["citations"][0]["start"]
    assert citation["end"] == extracted["citations"][0]["end"]
    assert citation["validation"]["status"] == "found"
    assert output["validation"]["counts"]["found"] == 1


def test_validate_review_citation_payload_returns_single_validation() -> None:
    extracted = review_preprocessed(
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        validate=False,
    )

    validation = validate_review_citation_payload(
        {"citation": extracted["citations"][0]},
        client=FakeClient(),
    )

    assert validation["citation_id"] == extracted["citations"][0]["id"]
    assert validation["status"] == "found"
    assert validation["lookup_key"] == "lookup-key"


def test_assess_review_payload_adds_exact_case_name_assessment_without_llm() -> None:
    extracted = review_preprocessed(
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954)."),
        validate=False,
    )
    extracted["citations"][0]["validation"] = {
        "citation_id": extracted["citations"][0]["id"],
        "locator": "347 U.S. 483",
        "status": "found",
        "source": "test",
        "message": "found",
        "case_names": ["Brown v. Board"],
        "lookup_status": 200,
        "lookup_cache": "miss",
        "lookup_key": "key",
        "error_message": None,
        "failure_detail": None,
        "matches": [
            {
                "case_name": "Brown v. Board",
                "date_filed": "1954-05-17",
                "court": None,
                "extra_data": {},
            }
        ],
        "extra_data": {},
    }

    output = asyncio.run(assess_review_payload(extracted))

    assessment = output["citations"][0]["assessment"]
    assert assessment["status"] == "assessed"
    assert assessment["result"]["case_name"]["initial"]["status"] == "exact_match"
    assert assessment["result"]["case_name"]["followup"]["status"] == "not_required"
    assert assessment["result"]["year"]["status"] == "exact_match"
    assert assessment["result"]["year"]["extracted_year"] == "1954"
    assert assessment["result"]["year"]["courtlistener_year"] == "1954"
    assert output["assessment"]["case_name_counts"]["exact_match"] == 1
    assert output["assessment"]["year_counts"]["exact_match"] == 1


def test_review_document_assessment_renders_cached_assessment_payload() -> None:
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
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
    validation = CitationValidation(
        citation_id="cite-1",
        locator="347 U.S. 483",
        status=ValidationStatus.FOUND,
        source="test",
        message="found",
        matches=(
            CitationMatch(
                case_name="Brown v. Board",
                date_filed="1954-05-17",
            ),
        ),
    )
    assessment_result = CitationAssessmentResult(
        case_name=CaseNameAssessmentRun(
            initial=CaseNameAssessment(
                status=CaseNameAssessmentStatus.EXACT_MATCH,
                extracted_case_name="Brown v. Board",
                courtlistener_case_name="Brown v. Board",
                message="match",
            ),
            followup=CaseNameReassessmentNotRequired(),
        ),
        court=_court_assessment(),
        year=YearAssessment(
            status=YearAssessmentStatus.EXACT_MATCH,
            extracted_year="1954",
            courtlistener_year="1954",
            message="match",
        ),
    )

    output = review_document_assessment(
        _assessed_document(
            preprocessed=preprocessed,
            citations=(citation,),
            validations=(validation,),
            assessments=(
                AssessedCitationAssessment(
                    citation_id="cite-1",
                    result=assessment_result,
                ),
            ),
        )
    )

    assert output["document"]["text"] == preprocessed.text
    assert output["citations"][0]["validation"]["status"] == "found"
    assert output["citations"][0]["assessment"]["result"]["case_name"]["initial"]["status"] == "exact_match"
    assert output["assessment"]["case_name_counts"]["exact_match"] == 1
    assert output["assessment"]["year_counts"]["exact_match"] == 1
    assert output["stats"]["assessed"] == 1


def test_review_document_assessment_preserves_waiting_citation() -> None:
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
    )

    output = review_document_assessment(
        _assessed_document(
            preprocessed=preprocessed,
            citations=(citation,),
            validations=(
                CitationValidation(
                    citation_id="cite-1",
                    locator="347 U.S. 483",
                    status=ValidationStatus.FOUND,
                    source="test",
                    message="found",
                ),
            ),
            assessments=(WaitingCitationAssessment(citation_id="cite-1"),),
        )
    )

    assert output["assessment"]["assessment_complete"] is False
    assert output["assessment"]["assessments"][0]["status"] == "waiting"


def test_review_document_assessment_allows_resolved_reextraction_handoff() -> None:
    """A non-semantic primary conclusion may have a reassessment verdict."""
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
    )
    year = YearAssessment(
        status=YearAssessmentStatus.EXACT_MATCH,
        extracted_year="1954",
        courtlistener_year="1954",
        message="match",
    )
    primary = CitationAssessmentResult(
        case_name=CaseNameAssessmentRun(
            initial=CaseNameAssessment(
                status=CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
                extracted_case_name="Brown v. Board",
                courtlistener_case_name="Brown v. Board of Education",
                message="re-extraction attempted",
            ),
            followup=CaseNameReassessed(
                reextracted_case_name=ReextractedCaseName(
                    case_name="Brown v. Board",
                    case_name_span=Span(0, 14),
                ),
                result=CaseNameAssessment(
                    status=CaseNameAssessmentStatus.SEMANTIC_MATCH,
                    extracted_case_name="Brown v. Board of Education",
                    courtlistener_case_name="Brown v. Board of Education",
                    message="semantic match after re-extraction",
                ),
            ),
        ),
        court=_court_assessment(),
        year=year,
    )
    output = review_document_assessment(
        _assessed_document(
            preprocessed=preprocessed,
            citations=(citation,),
            validations=(
                CitationValidation(
                    citation_id="cite-1",
                    locator="347 U.S. 483",
                    status=ValidationStatus.FOUND,
                    source="test",
                    message="found",
                ),
            ),
            assessments=(AssessedCitationAssessment(citation_id="cite-1", result=primary),),
        )
    )
    followup = output["assessment"]["assessments"][0]["result"]["case_name"]["followup"]
    assert followup["result"]["status"] == "semantic_match"


def test_review_snapshot_payload_detects_serialized_interface_boundaries() -> None:
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
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
    extraction = _extracted_document(preprocessed=preprocessed, citations=(citation,))
    validation = _validated_document(
        preprocessed=preprocessed,
        citations=(citation,),
        validations=(
            CitationValidation(
                citation_id="cite-1",
                locator="347 U.S. 483",
                status=ValidationStatus.FOUND,
                source="test",
                message="found",
            ),
        ),
    )
    assessment = _assessed_document(
        preprocessed=preprocessed,
        citations=(citation,),
        validations=validation.validations,
        assessments=(
            AssessedCitationAssessment(
                citation_id="cite-1",
                result=CitationAssessmentResult(
                    case_name=CaseNameAssessmentRun(
                        initial=CaseNameAssessment(
                            status=CaseNameAssessmentStatus.EXACT_MATCH,
                            extracted_case_name="Brown v. Board",
                            courtlistener_case_name="Brown v. Board",
                            message="match",
                        ),
                        followup=CaseNameReassessmentNotRequired(),
                    ),
                    court=_court_assessment(),
                    year=YearAssessment(
                        status=YearAssessmentStatus.EXACT_MATCH,
                        extracted_year="1954",
                        courtlistener_year="1954",
                        message="match",
                    ),
                ),
            ),
        ),
    )
    backend = E2EBackend()

    assert (
        _review_snapshot_payload(serialize_preprocessed_document(preprocessed), backend)["stage"]
        == "preprocessed"
    )
    assert _review_snapshot_payload(serialize_extracted_document(extraction), backend)["stage"] == "extracted"
    assert _review_snapshot_payload(serialize_validated_document(validation), backend)["stage"] == "validated"
    assert _review_snapshot_payload(serialize_assessed_document(assessment), backend)["stage"] == "assessed"


def test_llm_api_config_binds_an_explicit_openai_compatible_endpoint() -> None:
    config = llm_api_config_from_env(
        {
            "MELLEA_LRC_LLM_MODEL": "model-id",
            "MELLEA_LRC_LLM_TEMPERATURE": "0",
            "MELLEA_LRC_LLM_API_BASE": "https://llm.example/v1",
            "MELLEA_LRC_LLM_API_KEY": "api-key",
        }
    )

    assert config.model == "model-id"
    assert config.api_base == "https://llm.example/v1"
    assert config.api_key == "api-key"
