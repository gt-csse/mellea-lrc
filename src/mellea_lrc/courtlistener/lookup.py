"""Citation lookup normalization shared by direct and remote clients."""

from mellea_lrc.courtlistener.types import CourtListenerCitationLookup

HTTP_NOT_FOUND = 404


def normalize_citation_lookup_payload(
    payload: object,
    volume: str,
    reporter: str,
    page: str,
) -> CourtListenerCitationLookup:
    """Normalize CourtListener citation lookup payloads into one typed shape."""
    envelope = payload if isinstance(payload, dict) else {}
    raw_response = envelope.get("response", envelope)
    response = _first_lookup_response(raw_response)

    citation = _string_value(response.get("citation")) or f"{volume} {reporter} {page}"
    status = _int_value(response.get("status")) or HTTP_NOT_FOUND
    clusters = tuple(item for item in response.get("clusters", []) if isinstance(item, dict))
    error_message = _string_value(response.get("error_message"))

    return CourtListenerCitationLookup(
        citation=citation,
        status=status,
        clusters=clusters,
        cache=_string_value(envelope.get("cache")),
        key=_string_value(envelope.get("key")),
        error_message=error_message,
    )


def citation_lookup_response_dict(lookup: CourtListenerCitationLookup) -> dict[str, object]:
    """Convert a normalized citation lookup into a JSON-ready response dict."""
    response: dict[str, object] = {
        "citation": lookup.citation,
        "status": lookup.status,
        "clusters": list(lookup.clusters),
    }
    if lookup.error_message is not None:
        response["error_message"] = lookup.error_message
    return response


def citation_lookup_envelope_dict(lookup: CourtListenerCitationLookup) -> dict[str, object]:
    """Convert a normalized lookup into the cl-access service response envelope."""
    envelope: dict[str, object] = {"response": citation_lookup_response_dict(lookup)}
    if lookup.cache is not None:
        envelope["cache"] = lookup.cache
    if lookup.key is not None:
        envelope["key"] = lookup.key
    return envelope


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_value(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _first_lookup_response(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return first
    return {}
