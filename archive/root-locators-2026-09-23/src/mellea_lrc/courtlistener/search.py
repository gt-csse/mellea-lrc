"""CourtListener search retrieval and domain conversion."""

from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult
from mellea_lrc.courtlistener.search_transport import CourtListenerSearchResponsePayload


def normalize_search_payload(
    payload: object,
    *,
    query: str,
    search_type: str,
    semantic: bool,
) -> CourtListenerSearchResult:
    """Validate an external search payload and convert it to a domain result."""
    return CourtListenerSearchResponsePayload.model_validate(payload).to_domain(
        query=query,
        search_type=search_type,
        semantic=semantic,
    )
