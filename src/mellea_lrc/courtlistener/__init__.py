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
    CitationValidationClient,
    CourtListenerCitationRecord,
    CourtListenerCitationLookup,
    ValidationFailureDetail,
)

__all__ = [
    "CacheEntry",
    "CacheStore",
    "CitationLookupClient",
    "CitationValidationClient",
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
    "ValidationFailureDetail",
    "create_api",
    "normalize_citation_lookup_payload",
]
