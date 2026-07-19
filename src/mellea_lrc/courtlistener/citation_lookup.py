"""Citation lookup retrieval and domain conversion."""

from mellea_lrc.courtlistener.citation_lookup_models import (
    CourtListenerCitationRecord,
    CourtListenerCitationLookup,
)
from mellea_lrc.courtlistener.citation_lookup_transport import (
    CourtListenerCitationLookupResponsePayload,
    CourtListenerCitationLookupResultPayload,
)


def normalize_citation_lookup_response(
    payload: object,
    *,
    cache: str | None = None,
    key: str | None = None,
) -> CourtListenerCitationLookup:
    """Validate CourtListener's outer response list and convert its one result."""
    validated = CourtListenerCitationLookupResponsePayload.model_validate(payload)
    return validated.to_domain(cache=cache, key=key)


def normalize_citation_lookup_result(payload: object) -> CourtListenerCitationLookup:
    """Validate a flat citation result returned by the access service."""
    validated = CourtListenerCitationLookupResultPayload.model_validate(payload)
    return validated.to_domain()


def citation_lookup_result_dict(lookup: CourtListenerCitationLookup) -> dict[str, object]:
    """Convert a domain lookup into a JSON-ready service response."""
    response: dict[str, object] = {
        "citation": lookup.citation,
        "status": lookup.status,
        "clusters": [_citation_record_dict(item) for item in lookup.records],
    }
    if lookup.cache is not None:
        response["cache"] = lookup.cache
    if lookup.key is not None:
        response["key"] = lookup.key
    if lookup.extra_data:
        response["extra_data"] = lookup.extra_data.to_dict()
    return response


def _citation_record_dict(item: CourtListenerCitationRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_name": item.case_name,
        "date_filed": item.date_filed,
        "court": item.court,
        "court_id": item.court_id,
        "docket_id": item.docket_id,
    }
    if item.extra_data:
        payload["extra_data"] = item.extra_data.to_dict()
    return payload
