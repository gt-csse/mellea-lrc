"""Protocols for CourtListener client implementations."""

from typing import Protocol

from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationLookup


class CitationLookupClient(Protocol):
    """Retrieve CourtListener citations through the citation-lookup endpoint.

    This capability boundary allows direct and remote-service clients to be
    wired into retrieval without coupling callers to either transport.
    """

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Retrieve one exact reporter citation from CourtListener."""
