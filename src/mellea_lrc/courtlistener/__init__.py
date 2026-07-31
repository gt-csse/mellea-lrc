"""Reusable CourtListener API client and retrieval capabilities."""

from mellea_lrc.courtlistener.citation_lookup import normalize_citation_lookup_payload
from mellea_lrc.courtlistener.citation_lookup_models import (
    CourtListenerCitationLookup,
)
from mellea_lrc.courtlistener.client import (
    CourtListenerClient,
    CourtListenerConfig,
    CourtListenerError,
)
from mellea_lrc.courtlistener.docket import normalize_docket_payload
from mellea_lrc.courtlistener.docket_models import CourtListenerDocket, courtlistener_docket_url
from mellea_lrc.courtlistener.opinion import normalize_opinion_payload
from mellea_lrc.courtlistener.opinion_models import (
    CourtListenerOpinion,
    CourtListenerOpinionCluster,
    CourtListenerOpinionClusterCitation,
)
from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
from mellea_lrc.courtlistener.search import normalize_search_payload
from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult

__all__ = [
    "CourtListenerCitationLookup",
    "CourtListenerClient",
    "CourtListenerConfig",
    "CourtListenerDocket",
    "CourtListenerError",
    "CourtListenerOpinion",
    "CourtListenerOpinionCluster",
    "CourtListenerOpinionClusterCitation",
    "CourtListenerSearchResult",
    "CourtListenerServiceClient",
    "courtlistener_docket_url",
    "normalize_citation_lookup_payload",
    "normalize_docket_payload",
    "normalize_opinion_payload",
    "normalize_search_payload",
]
