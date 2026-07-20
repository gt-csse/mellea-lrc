"""Inbound boundary layer for untrusted CourtListener citation JSON."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, RootModel, model_validator

from mellea_lrc.courtlistener.citation_lookup_models import (
    CourtListenerCitationRecord,
    CourtListenerCitationLookup,
)


class _CitationLookupPayload(BaseModel):
    # CourtListener may add response fields independently of this package. The
    # first client boundary validates fields it understands and ignores the rest.
    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")


class CourtListenerCitationLookupRecordPayload(_CitationLookupPayload):
    """External CourtListener cluster payload."""

    case_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("case_name", "caseName"),
    )
    date_filed: str | None = Field(
        default=None,
        validation_alias=AliasChoices("date_filed", "dateFiled"),
    )
    court: str | None = None
    court_id: str | None = None
    # CourtListener returns docket_id as an int; keep it structured (str) for all
    # lookups — a stable case-identity key and, for ambiguous 300s, a per-candidate
    # discriminator worth analyzing.
    docket_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("docket_id", "docketId"),
    )

    def to_domain(self) -> CourtListenerCitationRecord:
        """Convert validated transport data into a domain record."""
        return CourtListenerCitationRecord(
            case_name=self.case_name,
            date_filed=self.date_filed,
            court=self.court,
            court_id=self.court_id,
            docket_id=str(self.docket_id) if self.docket_id is not None else None,
        )


class CourtListenerCitationLookupResultPayload(_CitationLookupPayload):
    """One citation result inside the external CourtListener response."""

    citation: str
    status: int
    clusters: list[CourtListenerCitationLookupRecordPayload] = Field(default_factory=list)
    error_message: str = ""

    def to_domain(self) -> CourtListenerCitationLookup:
        """Convert the validated result into a domain lookup."""
        return CourtListenerCitationLookup(
            citation=self.citation,
            status=self.status,
            records=tuple(item.to_domain() for item in self.clusters),
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
