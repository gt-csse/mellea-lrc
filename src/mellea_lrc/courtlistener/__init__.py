"""Reusable CourtListener access service components."""

from mellea_lrc.courtlistener.api import create_api
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
    CourtListenerCitationRecord,
    CourtListenerCitationLookup,
    RetrievalFailureDetail,
)
from mellea_lrc.courtlistener.taxonomy import (
    CourtsDBClassification,
    get_courts_db_classification,
    is_recognized_court,
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
    "CourtsDBClassification",
    "NullCache",
    "R2Cache",
    "RetrievalFailureDetail",
    "create_api",
    "get_courts_db_classification",
    "is_recognized_court",
    "normalize_citation_lookup_payload",
]
