"""Citation lookup retrieval and domain conversion."""

from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationLookup
from mellea_lrc.courtlistener.citation_lookup_transport import (
    CourtListenerCitationLookupResponsePayload,
)


def normalize_citation_lookup_payload(
    payload: object,
) -> CourtListenerCitationLookup:
    """Validate an external lookup payload and convert it to domain records."""
    validated = CourtListenerCitationLookupResponsePayload.model_validate(payload)
    return validated.to_domain()
