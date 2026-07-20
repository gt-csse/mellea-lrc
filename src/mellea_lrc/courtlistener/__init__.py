"""Reusable CourtListener API client and citation-lookup capability."""

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
from mellea_lrc.courtlistener.protocols import CitationLookupClient

__all__ = [
    "CitationLookupClient",
    "CourtListenerCitationLookup",
    "CourtListenerCitationRecord",
    "CourtListenerClient",
    "CourtListenerConfig",
    "CourtListenerError",
    "normalize_citation_lookup_payload",
]
