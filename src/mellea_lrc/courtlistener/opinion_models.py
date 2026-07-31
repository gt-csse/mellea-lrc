"""Domain models for CourtListener opinion clusters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CourtListenerOpinionClusterCitation:
    """One reporter citation attached to an opinion cluster."""

    volume: str
    reporter: str
    page: str


@dataclass(frozen=True, slots=True)
class CourtListenerOpinionCluster:
    """One CourtListener opinion cluster returned by a retrieval route."""

    cluster_id: str | None = None
    case_name: str | None = None
    date_filed: str | None = None
    court: str | None = None
    court_id: str | None = None
    docket_id: str | None = None
    citations: tuple[CourtListenerOpinionClusterCitation, ...] = ()
    sub_opinion_ids: tuple[str, ...] = ()

    @property
    def year(self) -> str | None:
        """Return the filing-year prefix when the upstream date provides one."""
        return self.date_filed[:4] if self.date_filed else None


@dataclass(frozen=True, slots=True)
class CourtListenerOpinion:
    """One CourtListener sub-opinion with citation-aware HTML."""

    opinion_id: str
    cluster_id: str | None
    opinion_type: str
    html_with_citations: str
    ordering_key: int | None = None
