"""Inbound boundary layer for untrusted CourtListener citation JSON.

Payloads are validated with Pydantic before conversion to immutable domain types.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue

from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.courtlistener.types import (
    CitationMatch,
    CourtListenerCitationLookup,
    ValidationFailureDetail,
)


class _ExternalPayload(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="allow")

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    extra_data: dict[str, JsonValue] = Field(default_factory=dict)

    def collected_extra_data(self) -> ExtraData:
        """Combine explicit and previously unknown external fields."""
        values: dict[str, object] = dict(self.extra_data)
        values.update(self.__pydantic_extra__ or {})
        return ExtraData(values)


class CitationMatchPayload(_ExternalPayload):
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

    def to_domain(self) -> CitationMatch:
        """Convert validated transport data into an immutable domain record."""
        return CitationMatch(
            case_name=self.case_name,
            date_filed=self.date_filed,
            court=self.court,
            court_id=self.court_id,
            extra_data=self.collected_extra_data(),
        )


class ValidationFailureDetailPayload(_ExternalPayload):
    """External lookup failure-detail payload."""

    failure_type: str | None = None
    message: str | None = None
    retryable: bool | None = None
    upstream_status_code: int | None = None
    key: str | None = None
    url: str | None = None
    retry_after_seconds: float | None = None

    def to_domain(self) -> ValidationFailureDetail:
        """Convert validated transport data into an immutable domain record."""
        return ValidationFailureDetail(
            failure_type=self.failure_type,
            message=self.message,
            retryable=self.retryable,
            upstream_status_code=self.upstream_status_code,
            key=self.key,
            url=self.url,
            retry_after_seconds=self.retry_after_seconds,
            extra_data=self.collected_extra_data(),
        )


class CitationLookupResponsePayload(_ExternalPayload):
    """External CourtListener response body."""

    citation: str | None = None
    status: int | None = None
    clusters: list[CitationMatchPayload] = Field(default_factory=list)
    error_message: str | None = None
    limit_detail: ValidationFailureDetailPayload | None = None


class CitationLookupEnvelopePayload(_ExternalPayload):
    """Cache envelope around a CourtListener response."""

    response: CitationLookupResponsePayload = Field(default_factory=CitationLookupResponsePayload)
    cache: str | None = None
    key: str | None = None

    def to_domain(self, *, fallback_citation: str, fallback_status: int) -> CourtListenerCitationLookup:
        """Convert the validated envelope into an immutable domain lookup."""
        response_extra = self.response.collected_extra_data().to_dict()
        envelope_extra = self.collected_extra_data().to_dict()
        extra_data: dict[str, object] = {}
        if response_extra:
            extra_data["response"] = response_extra
        if envelope_extra:
            extra_data["envelope"] = envelope_extra
        return CourtListenerCitationLookup(
            citation=self.response.citation or fallback_citation,
            status=self.response.status or fallback_status,
            matches=tuple(item.to_domain() for item in self.response.clusters),
            cache=self.cache,
            key=self.key,
            error_message=self.response.error_message,
            failure_detail=(
                self.response.limit_detail.to_domain() if self.response.limit_detail is not None else None
            ),
            extra_data=ExtraData(extra_data),
        )
