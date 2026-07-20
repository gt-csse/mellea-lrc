"""Reusable CourtListener API client and retrieval capabilities."""

from mellea_lrc.courtlistener.citation_lookup import normalize_citation_lookup_payload
from mellea_lrc.courtlistener.citation_lookup_models import (
    CourtListenerCitationLookup,
    CourtListenerCitationRecord,
)
from mellea_lrc.courtlistener.client import (
    CourtListenerClient,
    CourtListenerConfig,
    CourtListenerError,
)
from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
from mellea_lrc.courtlistener.search import normalize_search_payload
from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult

__all__ = [
    "CourtListenerCitationLookup",
    "CourtListenerCitationRecord",
    "CourtListenerClient",
    "CourtListenerConfig",
    "CourtListenerError",
    "CourtListenerSearchResult",
    "CourtListenerServiceClient",
    "normalize_citation_lookup_payload",
    "normalize_search_payload",
]
