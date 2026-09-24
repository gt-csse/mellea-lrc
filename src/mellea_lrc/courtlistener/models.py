"""Values returned by CourtListener's exact reporter citation lookup."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class _CourtListenerPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", populate_by_name=True)

    # Keep the upstream object for fields a later validation stage may need.
    raw_json: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def preserve_raw_json(cls, value: Any) -> Any:
        if isinstance(value, dict) and "raw_json" not in value:
            return {**value, "raw_json": dict(value)}
        return value


class CourtListenerClusterCitation(BaseModel):
    """A reporter citation listed on one opinion cluster."""

    model_config = ConfigDict(strict=True, extra="ignore")

    volume: str
    reporter: str
    page: str

    @field_validator("volume", "page", mode="before")
    @classmethod
    def stringify_number(cls, value: Any) -> Any:
        # CourtListener serializes citation volume as an integer in its API
        # examples; keep the caller-facing locator components as strings.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value


class CourtListenerCluster(_CourtListenerPayload):
    """One candidate opinion cluster for a reporter citation."""

    id: str | None = Field(default=None, validation_alias=AliasChoices("id", "clusterId", "cluster_id"))
    case_name: str | None = Field(default=None, validation_alias=AliasChoices("case_name", "caseName"))
    case_name_full: str | None = Field(
        default=None, validation_alias=AliasChoices("case_name_full", "caseNameFull")
    )
    date_filed: str | None = Field(default=None, validation_alias=AliasChoices("date_filed", "dateFiled"))
    court: str | None = None
    court_id: str | None = Field(default=None, validation_alias=AliasChoices("court_id", "courtId"))
    docket_id: str | None = Field(default=None, validation_alias=AliasChoices("docket_id", "docketId"))
    citations: list[CourtListenerClusterCitation] = Field(default_factory=list)

    @field_validator("id", "docket_id", mode="before")
    @classmethod
    def stringify_identifier(cls, value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value


class CourtListenerCitationLookup(_CourtListenerPayload):
    """The single result returned for one exact reporter citation request."""

    citation: str
    status: int
    clusters: list[CourtListenerCluster] = Field(default_factory=list)
    error_message: str | None = ""
