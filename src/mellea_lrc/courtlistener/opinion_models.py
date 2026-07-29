"""Domain models for CourtListener opinion clusters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CourtListenerOpinionCluster:
    """One CourtListener opinion cluster returned by a retrieval route."""

    cluster_id: str | None = None
    case_name: str | None = None
    date_filed: str | None = None
    court: str | None = None
    court_id: str | None = None
    docket_id: str | None = None

    @property
    def year(self) -> str | None:
        """Return the filing-year prefix when the upstream date provides one."""
        return self.date_filed[:4] if self.date_filed else None
