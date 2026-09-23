"""Inbound boundary for one CourtListener docket response."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from mellea_lrc.courtlistener.docket_models import CourtListenerDocket


class CourtListenerDocketResponsePayload(BaseModel):
    """External CourtListener docket payload."""

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    id: int | str
    court_id: str | None = Field(default=None, validation_alias=AliasChoices("court_id", "courtId"))
    court: str | None = None
    case_name: str | None = Field(default=None, validation_alias=AliasChoices("case_name", "caseName"))

    def to_domain(self) -> CourtListenerDocket:
        """Convert the external docket representation into the domain model."""
        court_id = self.court_id or _court_id_from_url(self.court)
        return CourtListenerDocket(docket_id=str(self.id), court_id=court_id, case_name=self.case_name)


def _court_id_from_url(value: str | None) -> str | None:
    if not value:
        return None
    return value.rstrip("/").rsplit("/", maxsplit=1)[-1] or None
