"""Durable evidence from one exact reporter-locator lookup."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.courtlistener.models import CourtListenerCitationLookup


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


class ReporterExactCandidateCheck(BaseModel):
    """Deterministic checks on one returned cluster, in response order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_index: int
    name_source: str | None = None
    plaintiff_present: bool | None = None
    defendant_present: bool | None = None
    subject_present: bool | None = None
    name_rule_passed: bool | None = None
    locator_present: bool | None = None
    qualifies: bool = False

    @model_validator(mode="after")
    def _validate_qualification(self) -> Self:
        if self.qualifies != (self.name_rule_passed is True and self.locator_present is not False):
            raise ValueError("Candidate qualification must follow the recorded rule checks")
        return self


class ReporterExactLookup(BaseModel):
    """A citation-local, checkpointable lookup and all its candidate evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    outcome: ReporterExactLookupOutcome
    query: ReporterExactLookupQuery | None = None
    response: CourtListenerCitationLookup | None = None
    candidate_checks: tuple[ReporterExactCandidateCheck, ...] = ()

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if self.outcome is ReporterExactLookupOutcome.UNNORMALIZABLE:
            if self.query is not None or self.response is not None or self.candidate_checks:
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
        if tuple(check.candidate_index for check in self.candidate_checks) != tuple(range(count)):
            raise ValueError("Every returned candidate must have one ordered check")
        return self
