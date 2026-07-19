"""Shared protocols for CourtListener client implementations."""

from collections.abc import Mapping
from typing import Protocol

from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationLookup
from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult


class CitationLookupClient(Protocol):
    """Protocol for clients that can validate reporter citations."""

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Look up one reporter citation."""


class CitationRetrievalClient(CitationLookupClient, Protocol):
    """Protocol for citation lookup plus case-level docket enrichment."""

    def get_docket(self, cl_docket_id: int | str) -> Mapping[str, object]:
        """Retrieve one canonical docket record."""

    def search_docket_entries(
        self,
        cl_docket_id: int | str,
        entry_number: int | str | None = None,
        cursor: str | None = None,
        order_by: str | None = None,
    ) -> Mapping[str, object]:
        """Retrieve one page of docket entries with nested RECAP documents."""

    def search_opinions(self, q: str) -> CourtListenerSearchResult:
        """Run a CourtListener opinion (``type=o``) search."""

    def search_recap(self, q: str) -> CourtListenerSearchResult:
        """Run a CourtListener RECAP (``type=r``) search."""
