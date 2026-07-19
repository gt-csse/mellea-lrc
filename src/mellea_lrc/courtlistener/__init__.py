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
from mellea_lrc.courtlistener.citation_lookup import (
    normalize_citation_lookup_response,
    normalize_citation_lookup_result,
)
from mellea_lrc.courtlistener.citation_lookup_models import (
    CourtListenerCitationRecord,
    CourtListenerCitationLookup,
)
from mellea_lrc.courtlistener.protocols import CitationLookupClient, CitationRetrievalClient
from mellea_lrc.courtlistener.remote import CourtListenerAccessClient, CourtListenerAccessConfig
from mellea_lrc.courtlistener.search_models import (
    CourtListenerRecapDocumentRecord,
    CourtListenerSearchRecord,
    CourtListenerSearchResult,
)
from mellea_lrc.courtlistener.search import normalize_search_payload, search_result_dict
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
    "CourtListenerRecapDocumentRecord",
    "CourtListenerSearchRecord",
    "CourtListenerSearchResult",
    "CourtsDBClassification",
    "NullCache",
    "R2Cache",
    "create_api",
    "get_courts_db_classification",
    "is_recognized_court",
    "normalize_citation_lookup_response",
    "normalize_citation_lookup_result",
    "normalize_search_payload",
    "search_result_dict",
]
