"""Reusable CourtListener API client and retrieval capabilities."""

from mellea_lrc.courtlistener.citation_lookup import normalize_citation_lookup_payload
from mellea_lrc.courtlistener.citation_lookup_models import (
    CourtListenerCitationLookup,
    CourtListenerCitationRecord,
)
from mellea_lrc.courtlistener.docket_models import CourtListenerDocket
from mellea_lrc.courtlistener.docket import normalize_docket_payload
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
    "CourtListenerDocket",
    "CourtListenerError",
    "CourtListenerSearchResult",
    "CourtListenerServiceClient",
    "normalize_citation_lookup_payload",
    "normalize_docket_payload",
    "normalize_search_payload",
]
