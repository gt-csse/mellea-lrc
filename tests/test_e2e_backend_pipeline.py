"""Tests for the standalone E2E backend pipeline helpers."""

import asyncio
from mellea_lrc.assessment import (
    AssessmentMetadata,
    AssessedCitationAssessment,
    AssessedDocument,
    CaseNameAssessment,
    CaseNameAssessmentStatus,
    CitationAssessment,
    CitationAssessmentResult,
    CitationReassessment,
    ModifiedExtractedCitation,
    ReassessedCitationReassessment,
    ReassessmentSkipReason,
    SkippedCitationReassessment,
    WaitingCitationReassessment,
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
from mellea_lrc.llm import (
    LlmProvider,
    chat_completions_base_url,
    llm_provider_config_from_env,
)
from mellea_lrc.validation.types import (
    CitationValidation,
    ValidatedDocument,
    ValidationMetadata,
    ValidationStatus,
)
from scripts.e2e_backend.api import _review_snapshot_payload
from scripts.label_studio.label_studio import to_label_studio_prediction
from scripts.e2e_backend.pipeline import (
    E2EBackend,
    add_validation_notes,
    assess_review_payload,
    predict_preprocessed,
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
    reassessments: tuple[CitationReassessment, ...] | None = None,
) -> AssessedDocument:
    if reassessments is None:
        reassessments = tuple(
            SkippedCitationReassessment(
                citation_id=citation.citation_id,
                reason=ReassessmentSkipReason.REEXTRACTION_NOT_REQUIRED,
                message="Primary assessment completed without re-extraction.",
            )
            for citation in citations
        )
    return AssessedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
        validations=validations,
        validation_metadata=ValidationMetadata(client_mode="custom", source="test"),
        assessments=assessments,
        reassessments=reassessments,
        assessment_metadata=AssessmentMetadata(),
    )


def test_predict_preprocessed_adds_validation_notes() -> None:
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483.")

    output = predict_preprocessed(preprocessed, client=FakeClient())

    assert output["text"] == preprocessed.text
    assert output["stats"]["citation_spans"] == 1
    assert output["stats"]["validated"] == 1
    notes = [item for item in output["prediction"]["result"] if item.get("from_name") == "notes"]
    assert notes
    assert notes[0]["value"]["text"][0].startswith("CourtListener: found 347 U.S. 483")


def test_e2e_backend_predict_text_exposes_pipeline_api() -> None:
    output = E2EBackend().predict_text("Brown v. Board, 347 U.S. 483.", validate=False)

    assert output["text"] == "Brown v. Board, 347 U.S. 483."
    assert output["validation"] is None
    assert output["stats"]["citation_spans"] == 1
    assert output["prediction"]["result"]


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
    assert assessment["result"]["case_assess"]["status"] == "exact_match"
    assert assessment["result"]["year_assess"]["status"] == "exact_match"
    assert assessment["result"]["year_assess"]["extracted_year"] == "1954"
    assert assessment["result"]["year_assess"]["courtlistener_year"] == "1954"
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
        citation_id="cite-1",
        case_assess=CaseNameAssessment(
            citation_id="cite-1",
            status=CaseNameAssessmentStatus.EXACT_MATCH,
            extracted_case_name="Brown v. Board",
            courtlistener_case_name="Brown v. Board",
            message="match",
        ),
        year_assess=YearAssessment(
            citation_id="cite-1",
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
                    citation_id=assessment_result.citation_id,
                    result=assessment_result,
                ),
            ),
        )
    )

    assert output["document"]["text"] == preprocessed.text
    assert output["citations"][0]["validation"]["status"] == "found"
    assert output["citations"][0]["assessment"]["result"]["case_assess"]["status"] == "exact_match"
    assert output["assessment"]["case_name_counts"]["exact_match"] == 1
    assert output["assessment"]["year_counts"]["exact_match"] == 1
    assert output["stats"]["assessed"] == 1


def test_review_document_assessment_rejects_unresolved_assessment_handoff() -> None:
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
    )

    try:
        review_document_assessment(
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
                assessments=(
                    AssessedCitationAssessment(
                        citation_id="cite-1",
                        result=CitationAssessmentResult(
                            citation_id="cite-1",
                            case_assess=CaseNameAssessment(
                                citation_id="cite-1",
                                status=CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
                                extracted_case_name="Brown v. Board",
                                courtlistener_case_name="Brown v. Board of Education",
                                message="needs assessment",
                            ),
                            year_assess=YearAssessment(
                                citation_id="cite-1",
                                status=YearAssessmentStatus.EXACT_MATCH,
                                extracted_year="1954",
                                courtlistener_year="1954",
                                message="match",
                            ),
                        ),
                    ),
                ),
                reassessments=(WaitingCitationReassessment(citation_id="cite-1"),),
            )
        )
    except ValueError as exc:
        assert "unresolved case-name assessment" in str(exc)
    else:
        raise AssertionError("Expected unresolved assessment handoff to be rejected")


def test_review_document_assessment_allows_resolved_reextraction_handoff() -> None:
    """NEEDS_ASSESSMENT in assessments is allowed when reassessments has the verdict."""
    preprocessed = preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483 (1954).")
    citation = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 35),
        matched_text="347 U.S. 483",
        citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
    )
    year_assess = YearAssessment(
        citation_id="cite-1",
        status=YearAssessmentStatus.EXACT_MATCH,
        extracted_year="1954",
        courtlistener_year="1954",
        message="match",
    )
    primary = CitationAssessmentResult(
        citation_id="cite-1",
        case_assess=CaseNameAssessment(
            citation_id="cite-1",
            status=CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
            extracted_case_name="Brown v. Board",
            courtlistener_case_name="Brown v. Board of Education",
            message="re-extraction attempted",
        ),
        year_assess=year_assess,
    )
    reassessment = CitationAssessmentResult(
        citation_id="cite-1",
        case_assess=CaseNameAssessment(
            citation_id="cite-1",
            status=CaseNameAssessmentStatus.SEMANTIC_MATCH,
            extracted_case_name="Brown v. Board of Education",
            courtlistener_case_name="Brown v. Board of Education",
            message="semantic match after re-extraction",
        ),
        year_assess=year_assess,
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
            reassessments=(
                ReassessedCitationReassessment(
                    citation_id="cite-1",
                    modified_citation=ModifiedExtractedCitation(
                        citation_id="cite-1",
                        span=Span(0, 14),
                        matched_text="Brown v. Board",
                        case_name="Brown v. Board",
                    ),
                    result=reassessment,
                ),
            ),
        )
    )
    assert output["assessment"]["reassessments"][0]["result"]["case_assess"]["status"] == (
        "semantic_match"
    )


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
                    citation_id="cite-1",
                    case_assess=CaseNameAssessment(
                        citation_id="cite-1",
                        status=CaseNameAssessmentStatus.EXACT_MATCH,
                        extracted_case_name="Brown v. Board",
                        courtlistener_case_name="Brown v. Board",
                        message="match",
                    ),
                    year_assess=YearAssessment(
                        citation_id="cite-1",
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


def test_add_validation_notes_skips_non_case_citations() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("See 28 U.S.C. § 636."),
        citations=(
            ExtractedCitation(
                citation_id="law",
                span=Span(4, 20),
                matched_text="28 U.S.C. § 636",
                citation=FullLawCitation(volume="28", reporter="U.S.C.", page="636"),
            ),
        ),
    )
    prediction = to_label_studio_prediction(extraction)

    enriched = add_validation_notes(
        prediction,
        _validated_document(
            preprocessed=extraction,
            citations=extraction.citations,
            validations=(
                CitationValidation(
                    citation_id="law",
                    locator=None,
                    status=ValidationStatus.SKIPPED,
                    source="cl-access",
                    message="Only FullCaseCitation is validated.",
                ),
            ),
        ),
    )

    assert not [item for item in enriched["result"] if item.get("from_name") == "notes"]


def test_llm_provider_config_supports_explicit_digitalocean_inference() -> None:
    config = llm_provider_config_from_env(
        {
            "MELLEA_LRC_LLM_PROVIDER": "digitalocean",
            "MELLEA_LRC_LLM_TEMPERATURE": "0",
            "MELLEA_LRC_LLM_MODEL": "openai-gpt-oss-20b",
            "MELLEA_LRC_LLM_API_BASE": "https://inference.do-ai.run",
            "MELLEA_LRC_LLM_API_KEY": "do-key",
        }
    )

    assert config.provider == LlmProvider.DIGITALOCEAN
    assert config.backend == "openai"
    assert config.model == "openai-gpt-oss-20b"
    assert config.api_base == "https://inference.do-ai.run"
    assert config.api_key == "do-key"
    assert config.temperature == 0
    assert chat_completions_base_url(config.api_base) == "https://inference.do-ai.run/v1"


def test_llm_provider_config_supports_explicit_openrouter() -> None:
    config = llm_provider_config_from_env(
        {
            "MELLEA_LRC_LLM_PROVIDER": "openrouter",
            "MELLEA_LRC_LLM_MODEL": "openai/gpt-4.1-mini",
            "MELLEA_LRC_LLM_API_BASE": "https://openrouter.ai/api",
            "MELLEA_LRC_LLM_API_KEY": "openrouter-key",
            "MELLEA_LRC_LLM_TEMPERATURE": "0",
            "MELLEA_LRC_LLM_OPENROUTER_REQUIRE_PARAMETERS": "1",
        }
    )

    assert config.provider == LlmProvider.OPENROUTER
    assert config.backend == "openai"
    assert config.model == "openai/gpt-4.1-mini"
    assert config.api_base == "https://openrouter.ai/api"
    assert config.api_key == "openrouter-key"
    assert config.temperature == 0
    assert chat_completions_base_url(config.api_base) == "https://openrouter.ai/api/v1"


def test_llm_provider_config_supports_official_deepseek_v4_pro() -> None:
    config = llm_provider_config_from_env(
        {
            "MELLEA_LRC_LLM_PROVIDER": "deepseek",
            "MELLEA_LRC_LLM_API_KEY": "deepseek-key",
            "MELLEA_LRC_LLM_TEMPERATURE": "0",
        }
    )

    assert config.provider == LlmProvider.DEEPSEEK
    assert config.backend == "openai"
    assert config.model == "deepseek-v4-pro"
    assert config.api_base == "https://api.deepseek.com"
    assert config.api_key == "deepseek-key"
    assert config.temperature == 0
    assert config.chat_completions_base_url() == "https://api.deepseek.com"


def test_llm_provider_config_requires_explicit_provider() -> None:
    try:
        llm_provider_config_from_env(
            {
                "MELLEA_LRC_LLM_MODEL": "openai/gpt-4.1-mini",
                "MELLEA_LRC_LLM_API_BASE": "https://openrouter.ai/api/v1",
                "MELLEA_LRC_LLM_API_KEY": "openrouter-key",
            }
        )
    except RuntimeError as exc:
        assert "MELLEA_LRC_LLM_PROVIDER" in str(exc)
    else:
        raise AssertionError("Expected explicit LLM provider to be required")


def test_llm_provider_config_uses_openrouter_hook_when_selected() -> None:
    config = llm_provider_config_from_env(
        {
            "MELLEA_LRC_LLM_PROVIDER": "openrouter",
            "MELLEA_LRC_LLM_MODEL": "openai/gpt-4.1-mini",
            "MELLEA_LRC_LLM_API_BASE": "https://openrouter.ai/api/v1",
            "MELLEA_LRC_LLM_API_KEY": "openrouter-key",
            "MELLEA_LRC_LLM_TEMPERATURE": "0",
            "MELLEA_LRC_LLM_OPENROUTER_REQUIRE_PARAMETERS": "1",
        }
    )

    assert config.provider == LlmProvider.OPENROUTER
    assert config.api_base == "https://openrouter.ai/api/v1"
    assert config.openrouter_require_parameters is True
    assert config.mellea_call_options(max_tokens=16) == {
        "temperature": 0,
        "max_tokens": 16,
        "extra_body": {"provider": {"require_parameters": True}},
    }


def test_llm_provider_config_uses_digitalocean_hook_when_selected() -> None:
    config = llm_provider_config_from_env(
        {
            "MELLEA_LRC_LLM_PROVIDER": "digitalocean",
            "MELLEA_LRC_LLM_MODEL": "openai-gpt-oss-20b",
            "MELLEA_LRC_LLM_TEMPERATURE": "0",
            "MELLEA_LRC_LLM_API_BASE": "https://inference.do-ai.run",
            "MELLEA_LRC_LLM_API_KEY": "do-key",
        }
    )

    assert config.model == "openai-gpt-oss-20b"
    assert config.api_base == "https://inference.do-ai.run"
    assert config.api_key == "do-key"
