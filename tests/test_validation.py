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
from mellea_lrc.courtlistener.types import CourtListenerCitationRecord, ValidationFailureDetail
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import PreprocessedDocument, preprocess_plain_text_from_string
from mellea_lrc.validation.pipeline import run_validation
from mellea_lrc.validation.types import (
    AmbiguousCitationValidation,
    CaseNameSearchStatus,
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
    assert validation.candidate.court_resolution.courtlistener_court_id == "scotus"
    assert validation.status == ValidationStatus.FOUND
    assert validation.locator == "347 U.S. 483"
    assert validation.case_names == ("Brown v. Board of Education",)
    assert validation.lookup_status == 200
    assert validation.lookup_cache == "miss"
    assert validation.candidate.record == CourtListenerCitationRecord(
        case_name="Brown v. Board of Education",
        date_filed="1954-05-17",
        court_id=None,
        docket_id="191796",
        extra_data=ExtraData({"absolute_url": "/opinion/1/"}),
    )
    assert validation.candidate.court_resolution.courtlistener_court_id == "scotus"
    assert validation.candidate.court_resolution.resolved_via == CourtResolutionSource.DOCKET_LOOKUP
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
    assert result.validations[0].candidate.record.court_id is None
    assert get_urls == ["https://cl-access.example.test/dockets/42"]


def test_validate_ambiguous_resolves_court_for_each_candidate() -> None:
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

    def get_json(url: str) -> dict[str, object]:
        get_urls.append(url)
        return {"court_id": "ca1" if url.endswith("/11") else "ca2"}

    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {
                "citation": "1 F.3d 2",
                "status": 300,
                "clusters": [
                    {"case_name": "Example A", "docket_id": 11},
                    {"case_name": "Example B", "docket_id": 22},
                ],
            }
        },
        get_json=get_json,
    )

    result = run_validation(extraction, client_mode="custom", client=client)

    validation = result.validations[0]
    assert isinstance(validation, AmbiguousCitationValidation)
    assert [candidate.record.case_name for candidate in validation.candidates] == [
        "Example A",
        "Example B",
    ]
    assert [candidate.court_resolution.courtlistener_court_id for candidate in validation.candidates] == [
        "ca1",
        "ca2",
    ]
    assert get_urls == [
        "https://cl-access.example.test/dockets/11",
        "https://cl-access.example.test/dockets/22",
    ]


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


def _not_found_extraction(citation: FullCaseCitation) -> ExtractedDocument:
    return _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Doe v. Roe, 999 U.S. 999."),
        citations=(
            ExtractedCitation(
                citation_id="nf",
                span=Span(0, 24),
                matched_text="999 U.S. 999",
                citation=citation,
            ),
        ),
    )


def test_not_found_reports_case_name_search_count() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {"count": 7, "results": []},
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter="U.S.", page="999"),
    )

    validation = run_validation(extraction, client_mode="custom", client=client).validations[0]

    assert validation.status == ValidationStatus.NOT_FOUND
    assert validation.candidate_search.status == CaseNameSearchStatus.SEARCHED
    assert validation.candidate_search.query == 'caseName:"Doe v. Roe"'
    assert validation.candidate_search.http_status == 200
    assert validation.candidate_search.case_count == 7


def test_not_found_reports_zero_case_name_search_results() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {"count": 0, "results": []},
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter="U.S.", page="999"),
    )

    validation = run_validation(extraction, client_mode="custom", client=client).validations[0]

    assert validation.candidate_search.status == CaseNameSearchStatus.SEARCHED
    assert validation.candidate_search.http_status == 200
    assert validation.candidate_search.case_count == 0


def test_not_found_preserves_failed_search_http_status() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {"http_status": 503, "detail": "upstream search unavailable"},
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter="U.S.", page="999"),
    )

    validation = run_validation(extraction, client_mode="custom", client=client).validations[0]

    assert validation.candidate_search.status == CaseNameSearchStatus.SEARCH_FAILED
    assert validation.candidate_search.http_status == 503
    assert validation.candidate_search.case_count is None
    assert validation.candidate_search.error_message == "upstream search unavailable"


def test_not_found_reads_count_from_deployed_service_raw_response() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {
            "results": [],
            "raw": {"count": 0, "next": None, "previous": None, "results": []},
        },
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter="U.S.", page="999"),
    )

    validation = run_validation(extraction, client_mode="custom", client=client).validations[0]

    assert validation.candidate_search.status == CaseNameSearchStatus.SEARCHED
    assert validation.candidate_search.http_status == 200
    assert validation.candidate_search.case_count == 0
    assert validation.candidate_search.error_message is None


def test_not_found_skips_search_without_both_parties() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: pytest.fail("search must not run without both parties"),
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", volume="999", reporter="U.S.", page="999"),
    )

    validation = run_validation(extraction, client_mode="custom", client=client).validations[0]

    assert validation.status == ValidationStatus.NOT_FOUND
    assert validation.candidate_search.status == CaseNameSearchStatus.SKIPPED_PARTIAL_CASE_NAME
    assert validation.candidate_search.case_count is None


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
