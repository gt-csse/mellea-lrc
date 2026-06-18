"""Tests for the standalone E2E backend pipeline helpers."""

from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import DocumentExtraction, ExtractedCitation
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.validation.types import CitationValidation, DocumentValidation, ValidationStatus
from scripts.label_studio.label_studio import to_label_studio_prediction
from scripts.e2e_backend.pipeline import (
    E2EBackend,
    add_validation_notes,
    assess_review_payload,
    predict_preprocessed,
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
        preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
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
        "clusters": [{"case_name": "Brown v. Board"}],
    }

    output = assess_review_payload(extracted)

    assessment = output["citations"][0]["assessment"]
    assert assessment["status"] == "exact_match"
    assert output["assessment"]["counts"]["exact_match"] == 1


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
            validations=(
                CitationValidation(
                    citation_id="law",
                    locator=None,
                    status=ValidationStatus.SKIPPED,
                    source="cl-access",
                    message="Only FullCaseCitation is validated.",
                ),
            )
        ),
    )

    assert not [item for item in enriched["result"] if item.get("from_name") == "notes"]
