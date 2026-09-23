"""Domain model for CourtListener citation-lookup responses."""

from dataclasses import dataclass

from mellea_lrc.courtlistener.opinion_models import CourtListenerOpinionCluster


@dataclass(frozen=True, slots=True)
class CourtListenerCitationLookup:
    """Normalized citation lookup response from CourtListener."""

    citation: str
    status: int
    clusters: tuple[CourtListenerOpinionCluster, ...]
    error_message: str | None = None
