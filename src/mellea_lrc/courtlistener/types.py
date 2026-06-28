"""Shared CourtListener service types."""

from dataclasses import dataclass
from typing import Protocol

from mellea_lrc.core.immutable import FrozenJsonObject, freeze_json_object


@dataclass(frozen=True, slots=True)
class CourtListenerCitationLookup:
    """Normalized citation lookup response from CourtListener access paths."""

    citation: str
    status: int
    clusters: tuple[FrozenJsonObject, ...]
    cache: str | None = None
    key: str | None = None
    error_message: str | None = None
    limit_detail: FrozenJsonObject | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clusters",
            tuple(freeze_json_object(item) for item in self.clusters),
        )
        if self.limit_detail is not None:
            object.__setattr__(self, "limit_detail", freeze_json_object(self.limit_detail))


class CitationLookupClient(Protocol):
    """Protocol for clients that can validate reporter citations."""

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Look up one reporter citation."""
