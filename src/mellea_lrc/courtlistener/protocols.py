"""Shared interface for CourtListener client implementations."""

from typing import Literal, Protocol

from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationLookup
from mellea_lrc.courtlistener.docket_models import CourtListenerDocket
from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult


class CourtListenerServiceClient(Protocol):
    """Client interface shared by direct and remote CourtListener access."""

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Retrieve one exact reporter citation from CourtListener."""

    def search(
        self,
        query: str,
        search_type: Literal["r", "rd", "d", "o"],
        cursor: str | None = None,
        *,
        semantic: bool = False,
    ) -> CourtListenerSearchResult:
        """Search a CourtListener corpus."""

    def get_docket(self, docket_id: str) -> CourtListenerDocket:
        """Retrieve one docket by its CourtListener identifier."""
