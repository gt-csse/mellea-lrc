"""Domain models for CourtListener citation lookup."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CourtListenerCitationRecord:
    """One CourtListener record retrieved by a reporter citation lookup."""

    case_name: str | None = None
    date_filed: str | None = None
    court: str | None = None
    court_id: str | None = None
    docket_id: str | None = None

    @property
    def year(self) -> str | None:
        """Return the filing year when the upstream date begins with one."""
        return self.date_filed[:4] if self.date_filed else None


@dataclass(frozen=True, slots=True)
class CourtListenerCitationLookup:
    """Normalized citation lookup response from CourtListener."""

    citation: str
    status: int
    records: tuple[CourtListenerCitationRecord, ...]
    error_message: str | None = None
