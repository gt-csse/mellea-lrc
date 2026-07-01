"""Tests for first-layer citation validation."""

import pytest
from pydantic import ValidationError

from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.remote import (
    CourtListenerAccessClient,
    CourtListenerAccessConfig,
)
from mellea_lrc.courtlistener.lookup import (
    citation_lookup_envelope_dict,
    normalize_citation_lookup_payload,
)
from mellea_lrc.courtlistener.types import CitationMatch, ValidationFailureDetail
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import PreprocessedDocument, preprocess_plain_text_from_string
from mellea_lrc.validation.pipeline import run_validation
from mellea_lrc.validation.types import (
    CourtResolutionSource,
    ValidationStatus,
)


def _client(
    response: dict[str, object],
    *,
    docket_response: dict[str, object] | None = None,
) -> CourtListenerAccessClient:
    return CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: response,
        get_json=lambda _url: docket_response or {},
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


def test_validate_full_case_found() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        citations=(
            ExtractedCitation(
                citation_id="abc123",
                span=Span(0, 28),
                matched_text="347 U.S. 483",
                citation=FullCaseCitation(volume="347", reporter="U.S.", page="483", court="scotus"),
            ),
        ),
    )

    result = run_validation(
        extraction,
        client_mode="custom",
        client=_client(
            {
                "cache": "miss",
                "request_id": "request-1",
                "response": {
                    "citation": "347 U.S. 483",
                    "status": 200,
                    "clusters": [
                        {
                            "case_name": "Brown v. Board of Education",
                            "date_filed": "1954-05-17",
                            "docket_id": 191796,
                            "absolute_url": "/opinion/1/",
                        }
                    ],
                    "query_time_ms": 12,
                },
            },
            docket_response={"id": 191796, "court_id": "scotus"},
        ),
    )

    validation = result.validations[0]
    assert result.text == extraction.text
    assert result.source_metadata == extraction.source_metadata
    assert result.preprocessing_metadata == extraction.preprocessing_metadata
    assert result.extraction_metadata == extraction.extraction_metadata
    assert result.citations == extraction.citations
    assert validation.court_resolution.courtlistener_court_id == "scotus"
    assert validation.status == ValidationStatus.FOUND
    assert validation.locator == "347 U.S. 483"
    assert validation.case_names == ("Brown v. Board of Education",)
    assert validation.lookup_status == 200
    assert validation.lookup_cache == "miss"
    assert validation.matches == (
        CitationMatch(
            case_name="Brown v. Board of Education",
            date_filed="1954-05-17",
            court_id=None,
            extra_data=ExtraData({"docket_id": 191796, "absolute_url": "/opinion/1/"}),
        ),
    )
    assert validation.court_resolution.courtlistener_court_id == "scotus"
    assert validation.court_resolution.resolved_via == CourtResolutionSource.DOCKET_LOOKUP
    assert result.found == (validation,)
    assert validation.extra_data.to_dict() == {
        "response": {"query_time_ms": 12},
        "envelope": {"request_id": "request-1"},
    }


def test_validate_found_docket_lookup_is_best_effort_and_deduplicated() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Example, 1 F.3d 2."),
        citations=(
            ExtractedCitation(
                citation_id="cite-1",
                span=Span(0, 18),
                matched_text="1 F.3d 2",
                citation=FullCaseCitation(volume="1", reporter="F.3d", page="2"),
            ),
        ),
    )
    get_urls: list[str] = []
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {
                "citation": "1 F.3d 2",
                "status": 200,
                "clusters": [
                    {"case_name": "Example A", "docket_id": 42},
                    {"case_name": "Example B", "docket_id": 42},
                ],
            }
        },
        get_json=lambda url: get_urls.append(url) or {"detail": "temporarily unavailable"},
    )

    result = run_validation(extraction, client_mode="custom", client=client)

    assert result.validations[0].status == ValidationStatus.FOUND
    assert [match.court_id for match in result.validations[0].matches] == [None, None]
    assert get_urls == ["https://cl-access.example.test/dockets/42"]


def test_validate_non_case_citation_is_skipped() -> None:
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

    result = run_validation(extraction, client_mode="custom", client=_client({}))

    assert result.validations[0].status == ValidationStatus.SKIPPED


def test_validate_surfaces_typed_courtlistener_failure_detail() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        citations=(
            ExtractedCitation(
                citation_id="limited",
                span=Span(0, 28),
                matched_text="347 U.S. 483",
                citation=FullCaseCitation(volume="347", reporter="U.S.", page="483"),
            ),
        ),
    )

    result = run_validation(
        extraction,
        client_mode="custom",
        client=_client(
            {
                "response": {
                    "citation": "347 U.S. 483",
                    "status": 429,
                    "error_message": "CourtListener POST failed with 429",
                    "limit_detail": {
                        "failure_type": "api_limit",
                        "message": "CourtListener POST failed with 429",
                        "retryable": True,
                        "upstream_status_code": 429,
                    },
                    "clusters": [],
                },
            }
        ),
    )

    validation = result.validations[0]
    assert validation.status == ValidationStatus.THROTTLED
    assert validation.error_message == "CourtListener POST failed with 429"
    assert validation.failure_detail == ValidationFailureDetail(
        failure_type="api_limit",
        message="CourtListener POST failed with 429",
        retryable=True,
        upstream_status_code=429,
    )


def test_courtlistener_transport_rejects_type_coercion() -> None:
    with pytest.raises(ValidationError):
        normalize_citation_lookup_payload(
            {"response": {"citation": "347 U.S. 483", "status": "200"}},
            "347",
            "U.S.",
            "483",
        )


def test_direct_courtlistener_response_does_not_duplicate_fields_as_envelope_extras() -> None:
    lookup = normalize_citation_lookup_payload(
        {
            "citation": "347 U.S. 483",
            "status": 200,
            "clusters": [],
            "query_time_ms": 12,
        },
        "347",
        "U.S.",
        "483",
    )

    assert lookup.extra_data.to_dict() == {"response": {"query_time_ms": 12}}


def test_courtlistener_service_round_trip_preserves_explicit_extra_data() -> None:
    original = normalize_citation_lookup_payload(
        {
            "request_id": "request-1",
            "response": {
                "citation": "347 U.S. 483",
                "status": 200,
                "query_time_ms": 12,
                "clusters": [{"case_name": "Brown", "absolute_url": "/opinion/1/"}],
            },
        },
        "347",
        "U.S.",
        "483",
    )

    restored = normalize_citation_lookup_payload(
        citation_lookup_envelope_dict(original),
        "347",
        "U.S.",
        "483",
    )

    assert restored == original


def test_validate_missing_locator_is_invalid_without_service_call() -> None:
    calls = 0

    def post_json(_url: str, _data: object) -> object:
        nonlocal calls
        calls += 1
        return {}

    extraction = _extracted_document(
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

    result = run_validation(extraction, client_mode="custom", client=client)

    assert result.validations[0].status == ValidationStatus.INVALID
    assert calls == 0


def test_validate_rejects_custom_mode_without_client() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )

    with pytest.raises(ValueError, match="client is required"):
        run_validation(extraction, client_mode="custom")


def test_validate_rejects_client_override_for_managed_modes() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )
    client = _client({})

    with pytest.raises(ValueError, match="client must be None"):
        run_validation(extraction, client_mode="deployed", client=client)

    with pytest.raises(ValueError, match="client must be None"):
        run_validation(extraction, client_mode="sdk", client=client)


def test_validate_rejects_unknown_client_mode() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )

    with pytest.raises(ValueError, match="client_mode must be one of"):
        run_validation(extraction, client_mode="other")  # type: ignore[arg-type]
