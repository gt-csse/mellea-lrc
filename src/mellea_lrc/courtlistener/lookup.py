"""Citation lookup validation and domain conversion."""

from mellea_lrc.courtlistener.transport import CitationLookupEnvelopePayload
from mellea_lrc.courtlistener.types import (
    CitationMatch,
    CourtListenerCitationLookup,
    ValidationFailureDetail,
)

HTTP_NOT_FOUND = 404


def normalize_citation_lookup_payload(
    payload: object,
    volume: str,
    reporter: str,
    page: str,
) -> CourtListenerCitationLookup:
    """Validate an external lookup payload and convert it to domain records."""
    envelope = payload if isinstance(payload, dict) else {}
    has_envelope = "response" in envelope
    raw_response = envelope.get("response") if has_envelope else envelope
    response = _first_lookup_response(raw_response)
    envelope_fields = (
        {
            "cache": envelope.get("cache"),
            "key": envelope.get("key"),
            "extra_data": envelope.get("extra_data", {}),
            **{
                key: value
                for key, value in envelope.items()
                if key not in {"response", "cache", "key", "extra_data"}
            },
        }
        if has_envelope
        else {}
    )
    validated = CitationLookupEnvelopePayload.model_validate(
        {
            "response": response,
            **envelope_fields,
        }
    )
    return validated.to_domain(
        fallback_citation=f"{volume} {reporter} {page}",
        fallback_status=HTTP_NOT_FOUND,
    )


def citation_lookup_response_dict(lookup: CourtListenerCitationLookup) -> dict[str, object]:
    """Convert a domain lookup into a JSON-ready service response."""
    response: dict[str, object] = {
        "citation": lookup.citation,
        "status": lookup.status,
        "clusters": [_citation_match_dict(item) for item in lookup.matches],
    }
    if lookup.error_message is not None:
        response["error_message"] = lookup.error_message
    if lookup.failure_detail is not None:
        response["limit_detail"] = _failure_detail_dict(lookup.failure_detail)
    response_extra = lookup.extra_data.to_dict().get("response")
    if isinstance(response_extra, dict) and response_extra:
        response["extra_data"] = response_extra
    return response


def citation_lookup_envelope_dict(lookup: CourtListenerCitationLookup) -> dict[str, object]:
    """Convert a domain lookup into the cl-access service envelope."""
    envelope: dict[str, object] = {"response": citation_lookup_response_dict(lookup)}
    if lookup.cache is not None:
        envelope["cache"] = lookup.cache
    if lookup.key is not None:
        envelope["key"] = lookup.key
    envelope_extra = lookup.extra_data.to_dict().get("envelope")
    if isinstance(envelope_extra, dict) and envelope_extra:
        envelope["extra_data"] = envelope_extra
    return envelope


def _citation_match_dict(item: CitationMatch) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_name": item.case_name,
        "date_filed": item.date_filed,
        "court": item.court,
        "court_id": item.court_id,
    }
    if item.extra_data:
        payload["extra_data"] = item.extra_data.to_dict()
    return payload


def _failure_detail_dict(item: ValidationFailureDetail) -> dict[str, object]:
    payload: dict[str, object] = {
        "failure_type": item.failure_type,
        "message": item.message,
        "retryable": item.retryable,
        "upstream_status_code": item.upstream_status_code,
        "key": item.key,
        "url": item.url,
        "retry_after_seconds": item.retry_after_seconds,
    }
    if item.extra_data:
        payload["extra_data"] = item.extra_data.to_dict()
    return payload


def _first_lookup_response(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}
