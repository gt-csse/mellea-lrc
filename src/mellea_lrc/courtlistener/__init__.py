"""Reusable clients for direct and service-backed CourtListener access."""

from mellea_lrc.courtlistener.cache import CacheEntry, CacheStore, NullCache, R2Cache
from mellea_lrc.courtlistener.client import (
    CourtListenerClient,
    CourtListenerConfig,
    CourtListenerError,
    CourtListenerRateLimitConfig,
    CourtListenerRateLimiter,
)
from mellea_lrc.courtlistener.lookup import normalize_citation_lookup_payload
from mellea_lrc.courtlistener.remote import CourtListenerAccessClient, CourtListenerAccessConfig
from mellea_lrc.courtlistener.types import (
    CitationLookupClient,
    CitationRetrievalClient,
    CourtListenerCitationLookup,
    CourtListenerCitationRecord,
    RetrievalFailureDetail,
)

__all__ = [
    "CacheEntry",
    "CacheStore",
    "CitationLookupClient",
    "CitationRetrievalClient",
    "CourtListenerAccessClient",
    "CourtListenerAccessConfig",
    "CourtListenerCitationLookup",
    "CourtListenerCitationRecord",
    "CourtListenerClient",
    "CourtListenerConfig",
    "CourtListenerError",
    "CourtListenerRateLimitConfig",
    "CourtListenerRateLimiter",
    "NullCache",
    "R2Cache",
    "RetrievalFailureDetail",
    "normalize_citation_lookup_payload",
]
