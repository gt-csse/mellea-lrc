"""Durable evidence from one exact reporter-locator lookup."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.courtlistener.models import CourtListenerCitationLookup, CourtListenerDocket


class ReporterExactLookupOutcome(str, Enum):
    """Candidate count, not a final case-identity judgment."""

    UNNORMALIZABLE = "unnormalizable"
    NOT_FOUND = "not_found"
    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"


class ReporterExactLookupQuery(BaseModel):
    """Normalized locator parts sent to the exact lookup endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    volume: int
    edition: str
    page: str


class ReporterExactLookup(BaseModel):
    """One citation-local query and its complete CourtListener response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    outcome: ReporterExactLookupOutcome
    query: ReporterExactLookupQuery | None = None
    response: CourtListenerCitationLookup | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if self.outcome is ReporterExactLookupOutcome.UNNORMALIZABLE:
            if self.query is not None or self.response is not None:
                raise ValueError("An unnormalizable locator cannot have lookup results")
            return self
        if self.query is None or self.response is None:
            raise ValueError("A reporter lookup requires its query and response")
        count = len(self.response.clusters)
        expected = (
            ReporterExactLookupOutcome.NOT_FOUND
            if count == 0
            else ReporterExactLookupOutcome.UNIQUE
            if count == 1
            else ReporterExactLookupOutcome.AMBIGUOUS
        )
        if self.outcome is not expected:
            raise ValueError("Lookup outcome must reflect the returned candidate count")
        return self


class ReporterExactDocket(BaseModel):
    """Saved docket response used to check the court of one unique cluster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    docket_id: str
    response: CourtListenerDocket | None

    @model_validator(mode="after")
    def _validate_docket(self) -> Self:
        if not self.docket_id or (self.response is not None and self.response.id != self.docket_id):
            raise ValueError("Reporter docket evidence must match the requested docket ID")
        return self
