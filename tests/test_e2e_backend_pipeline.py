"""Tests for the standalone E2E backend pipeline helpers."""

from mellea_lrc.assessment import (
    CaseNameAssessment,
    CaseNameAssessmentStatus,
    CitationAssessment,
    DocumentAssessment,
    YearAssessment,
    YearAssessmentStatus,
)
from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import DocumentExtraction, ExtractedCitation
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.serialization import (
    serialize_document_assessment,
    serialize_document_extraction,
    serialize_document_validation,
    serialize_preprocessed_document,
)
from mellea_lrc.validation.types import CitationValidation, DocumentValidation, ValidationStatus
from scripts.e2e_backend.api import _review_snapshot_payload
from scripts.label_studio.label_studio import to_label_studio_prediction
from scripts.e2e_backend.pipeline import (
    E2EBackend,
    _assessment_provider_config_from_env,
    _chat_completions_base_url,
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
        return type(
            "Lookup",
            (),
            {
                "citation": "347 U.S. 483",
                "status": 200,
                "clusters": (
                    {
                        "case_name": "Brown v. Board of Education",
                        "date_filed": "1954-05-17",
                        "court": "scotus",
                    },
                ),
                "cache": "miss",
                "key": "lookup-key",
                "error_message": None,
                "limit_detail": None,
            },
        )()


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
    assert citation["validation"]["clusters"][0]["date_filed"] == "1954-05-17"
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
        "limit_detail": None,
        "clusters": [{"case_name": "Brown v. Board", "date_filed": "1954-05-17"}],
    }

    output = assess_review_payload(extracted)

    assessment = output["citations"][0]["assessment"]
    assert assessment["status"] == "exact_match"
    assert assessment["case_assess"]["status"] == "exact_match"
    assert assessment["year_assess"]["status"] == "exact_match"
    assert assessment["year_assess"]["extracted_year"] == "1954"
    assert assessment["year_assess"]["courtlistener_year"] == "1954"
    assert output["assessment"]["counts"]["exact_match"] == 1


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
        case_names=("Brown v. Board",),
        clusters=({"case_name": "Brown v. Board", "date_filed": "1954-05-17"},),
    )
    assessment = CitationAssessment(
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
        DocumentAssessment(
            preprocessed=preprocessed,
            citations=(citation,),
            validations=(validation,),
            assessments=(assessment,),
        )
    )

    assert output["document"]["text"] == preprocessed.text
    assert output["citations"][0]["validation"]["status"] == "found"
    assert output["citations"][0]["assessment"]["status"] == "exact_match"
    assert output["assessment"]["counts"]["exact_match"] == 1
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
            DocumentAssessment(
                preprocessed=preprocessed,
                citations=(citation,),
                validations=(),
                assessments=(
                    CitationAssessment(
                        citation_id="cite-1",
                        case_assess=CaseNameAssessment(
                            citation_id="cite-1",
                            status=CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
                            extracted_case_name="Brown v. Board",
                            courtlistener_case_name="Brown v. Board of Education",
                            message="needs assessment",
                        ),
                    ),
                ),
            )
        )
    except ValueError as exc:
        assert "unresolved case-name assessment" in str(exc)
    else:
        raise AssertionError("Expected unresolved assessment handoff to be rejected")


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
    extraction = DocumentExtraction(preprocessed=preprocessed, citations=(citation,))
    validation = DocumentValidation(
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
    assessment = DocumentAssessment(
        preprocessed=preprocessed,
        citations=(citation,),
        validations=validation.validations,
        assessments=(
            CitationAssessment(
                citation_id="cite-1",
                case_assess=CaseNameAssessment(
                    citation_id="cite-1",
                    status=CaseNameAssessmentStatus.EXACT_MATCH,
                    extracted_case_name="Brown v. Board",
                    courtlistener_case_name="Brown v. Board",
                    message="match",
                ),
            ),
        ),
    )
    backend = E2EBackend()

    assert _review_snapshot_payload(serialize_preprocessed_document(preprocessed), backend)["stage"] == "preprocessed"
    assert _review_snapshot_payload(serialize_document_extraction(extraction), backend)["stage"] == "extracted"
    assert _review_snapshot_payload(serialize_document_validation(validation), backend)["stage"] == "validated"
    assert _review_snapshot_payload(serialize_document_assessment(assessment), backend)["stage"] == "assessed"


def test_add_validation_notes_skips_non_case_citations() -> None:
    extraction = DocumentExtraction(
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
        DocumentValidation(
            preprocessed=extraction.preprocessed,
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


def test_assessment_provider_config_supports_digitalocean_inference_defaults() -> None:
    config = _assessment_provider_config_from_env(
        {
            "MELLEA_LRC_ASSESSMENT_PROVIDER": "digitalocean",
            "MELLEA_LRC_ASSESSMENT_MODEL": "openai/gpt-4.1-mini",
            "DIGITALOCEAN_INFERENCE_MODEL": "openai-gpt-oss-20b",
            "DIGITALOCEAN_INFERENCE_API_KEY": "do-key",
        }
    )

    assert config["provider"] == "digitalocean"
    assert config["backend"] == "openai"
    assert config["model"] == "openai-gpt-oss-20b"
    assert config["api_base"] == "https://inference.do-ai.run"
    assert config["api_key"] == "do-key"
    assert _chat_completions_base_url(config["api_base"]) == "https://inference.do-ai.run/v1"


def test_assessment_provider_config_supports_openrouter_defaults() -> None:
    config = _assessment_provider_config_from_env(
        {
            "MELLEA_LRC_ASSESSMENT_PROVIDER": "openrouter",
            "MELLEA_LRC_ASSESSMENT_MODEL": "openai/gpt-4.1-mini",
            "MELLEA_LRC_ASSESSMENT_API_KEY": "openrouter-key",
        }
    )

    assert config["provider"] == "openrouter"
    assert config["backend"] == "openai"
    assert config["model"] == "openai/gpt-4.1-mini"
    assert config["api_base"] == "https://openrouter.ai/api"
    assert config["api_key"] == "openrouter-key"
    assert _chat_completions_base_url(config["api_base"]) == "https://openrouter.ai/api/v1"


def test_assessment_provider_config_infers_openrouter_from_generic_base() -> None:
    config = _assessment_provider_config_from_env(
        {
            "MELLEA_LRC_ASSESSMENT_MODEL": "openai/gpt-4.1-mini",
            "MELLEA_LRC_ASSESSMENT_API_BASE": "https://openrouter.ai/api/v1",
            "MELLEA_LRC_ASSESSMENT_API_KEY": "openrouter-key",
        }
    )

    assert config["provider"] == "openrouter"
    assert config["api_base"] == "https://openrouter.ai/api/v1"
    assert config["api_key"] == "openrouter-key"


def test_assessment_provider_config_uses_digitalocean_hook_when_selected() -> None:
    config = _assessment_provider_config_from_env(
        {
            "MELLEA_LRC_ASSESSMENT_PROVIDER": "digitalocean",
            "MELLEA_LRC_ASSESSMENT_MODEL": "openai/gpt-4.1-mini",
            "DIGITALOCEAN_INFERENCE_MODEL": "openai-gpt-oss-20b",
            "MELLEA_LRC_ASSESSMENT_API_BASE": "https://openrouter.ai/api/v1",
            "MELLEA_LRC_ASSESSMENT_API_KEY": "openrouter-key",
            "DIGITALOCEAN_INFERENCE_API_BASE": "https://inference.do-ai.run",
            "DIGITALOCEAN_INFERENCE_API_KEY": "do-key",
        }
    )

    assert config["model"] == "openai-gpt-oss-20b"
    assert config["api_base"] == "https://inference.do-ai.run"
    assert config["api_key"] == "do-key"
