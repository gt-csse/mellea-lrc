"""Typed CourtListener domain records and lookup protocol."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Protocol

from mellea_lrc.core.immutable import ExtraData


@dataclass(frozen=True, slots=True)
class CourtListenerCitationRecord:
    """One CourtListener record retrieved by a reporter citation lookup."""

    case_name: str | None = None
    date_filed: str | None = None
    court: str | None = None
    court_id: str | None = None
    docket_id: str | None = None
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def year(self) -> str | None:
        """Return the filing year when the upstream date begins with one."""
        return self.date_filed[:4] if self.date_filed else None


@dataclass(frozen=True, slots=True)
class RetrievalFailureDetail:
    """Typed diagnostic detail for a failed or throttled lookup."""

    failure_type: str | None = None
    message: str | None = None
    retryable: bool | None = None
    upstream_status_code: int | None = None
    key: str | None = None
    url: str | None = None
    retry_after_seconds: float | None = None
    extra_data: ExtraData = field(default_factory=ExtraData)


@dataclass(frozen=True, slots=True)
class CourtListenerCitationLookup:
    """Normalized citation lookup response from CourtListener access paths."""

    citation: str
    status: int
    records: tuple[CourtListenerCitationRecord, ...]
    cache: str | None = None
    key: str | None = None
    error_message: str | None = None
    failure_detail: RetrievalFailureDetail | None = None
    extra_data: ExtraData = field(default_factory=ExtraData)


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

    def search_opinions(self, q: str) -> Mapping[str, object]:
        """Run a CourtListener opinion (``type=o``) search; response carries ``count``."""

    def search_recap(self, q: str) -> Mapping[str, object]:
        """Run a CourtListener RECAP (``type=r``) search; response carries ``count``."""
