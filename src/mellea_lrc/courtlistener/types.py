"""Shared CourtListener service types."""

from dataclasses import dataclass
from typing import Protocol, TypeAlias

JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class CourtListenerCitationLookup:
    """Normalized citation lookup response from CourtListener access paths."""

    citation: str
    status: int
    clusters: tuple[JsonObject, ...]
    cache: str | None = None
    key: str | None = None
    error_message: str | None = None


class CitationLookupClient(Protocol):
    """Protocol for clients that can validate reporter citations."""

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Look up one reporter citation."""
