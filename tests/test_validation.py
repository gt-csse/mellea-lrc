"""Tests for first-layer citation validation."""

import pytest

from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.remote import (
    CourtListenerAccessClient,
    CourtListenerAccessConfig,
)
from mellea_lrc.extraction.types import DocumentExtraction, ExtractedCitation
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.validation.pipeline import validate_extraction
from mellea_lrc.validation.types import ValidationStatus


def _client(response: dict[str, object]) -> CourtListenerAccessClient:
    return CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: response,
    )


def test_validate_full_case_found() -> None:
    extraction = DocumentExtraction(
        preprocessed=preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        citations=(
            ExtractedCitation(
                citation_id="abc123",
                span=Span(0, 28),
                matched_text="347 U.S. 483",
                citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
            ),
        ),
    )

    result = validate_extraction(
        extraction,
        client_mode="custom",
        client=_client(
            {
                "cache": "miss",
                "response": {
                    "citation": "347 U.S. 483",
                    "status": 200,
                    "clusters": [{"case_name": "Brown v. Board of Education"}],
                },
            }
        ),
    )

    validation = result.validations[0]
    assert validation.status == ValidationStatus.FOUND
    assert validation.locator == "347 U.S. 483"
    assert validation.case_names == ("Brown v. Board of Education",)
    assert result.found == (validation,)


def test_validate_non_case_citation_is_skipped() -> None:
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

    result = validate_extraction(extraction, client_mode="custom", client=_client({}))

    assert result.validations[0].status == ValidationStatus.SKIPPED


def test_validate_missing_locator_is_invalid_without_service_call() -> None:
    calls = 0

    def post_json(_url: str, _data: object) -> object:
        nonlocal calls
        calls += 1
        return {}

    extraction = DocumentExtraction(
        preprocessed=preprocess_plain_text_from_string("Bad citation."),
        citations=(
            ExtractedCitation(
                citation_id="bad",
                span=Span(0, 12),
                matched_text="Bad citation",
                citation=FullCaseCitation(volume="1", reporter=None, page="2"),
            ),
        ),
    )
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=post_json,
    )

    result = validate_extraction(extraction, client_mode="custom", client=client)

    assert result.validations[0].status == ValidationStatus.INVALID
    assert calls == 0


def test_validate_rejects_custom_mode_without_client() -> None:
    extraction = DocumentExtraction(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )

    with pytest.raises(ValueError, match="client is required"):
        validate_extraction(extraction, client_mode="custom")


def test_validate_rejects_client_override_for_managed_modes() -> None:
    extraction = DocumentExtraction(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )
    client = _client({})

    with pytest.raises(ValueError, match="client must be None"):
        validate_extraction(extraction, client_mode="deployed", client=client)

    with pytest.raises(ValueError, match="client must be None"):
        validate_extraction(extraction, client_mode="sdk", client=client)


def test_validate_rejects_unknown_client_mode() -> None:
    extraction = DocumentExtraction(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )

    with pytest.raises(ValueError, match="client_mode must be one of"):
        validate_extraction(extraction, client_mode="other")  # type: ignore[arg-type]
