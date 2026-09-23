"""Inbound boundary layer for untrusted CourtListener citation JSON."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationLookup
from mellea_lrc.courtlistener.opinion_transport import CourtListenerOpinionClusterPayload


class _CitationLookupPayload(BaseModel):
    # CourtListener may add response fields independently of this package. The
    # first client boundary validates fields it understands and ignores the rest.
    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")


class CourtListenerCitationLookupResultPayload(_CitationLookupPayload):
    """One citation result inside the external CourtListener response."""

    citation: str
    status: int
    clusters: list[CourtListenerOpinionClusterPayload] = Field(default_factory=list)
    error_message: str = ""

    def to_domain(self) -> CourtListenerCitationLookup:
        """Convert the validated result into a domain lookup."""
        return CourtListenerCitationLookup(
            citation=self.citation,
            status=self.status,
            clusters=tuple(item.to_domain() for item in self.clusters),
            error_message=self.error_message or None,
        )


class CourtListenerCitationLookupResponsePayload(RootModel[list[CourtListenerCitationLookupResultPayload]]):
    """External response for one explicit reporter-citation lookup."""

    model_config = ConfigDict(strict=True, frozen=True)

    @model_validator(mode="after")
    def require_one_result(self) -> CourtListenerCitationLookupResponsePayload:
        """Require the cardinality promised by the explicit-locator request."""
        if len(self.root) != 1:
            message = "Citation lookup response must contain exactly one result"
            raise ValueError(message)
        return self

    def to_domain(self) -> CourtListenerCitationLookup:
        """Convert the response's single result into a domain lookup."""
        return self.root[0].to_domain()
