"""Durable evidence from one exact reporter-locator lookup."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mellea_lrc.courtlistener.models import CourtListenerCitationLookup, CourtListenerDocket
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.ivr import IvrRun


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


class ReporterExactCandidateDocket(ReporterExactDocket):
    """Linked court evidence for one candidate of an ambiguous response."""

    candidate_index: int

    @model_validator(mode="after")
    def _validate_index(self) -> Self:
        if self.candidate_index < 0:
            raise ValueError("Candidate index must be nonnegative")
        return self


class ReporterExactAmbiguityOutcome(str, Enum):
    """Facts about a saved multi-candidate rule check, separate from routing."""

    UNIQUE_RULE_MATCH = "unique_rule_match"
    NO_UNIQUE_RULE_MATCH = "no_unique_rule_match"
    CANDIDATE_LIMIT_EXCEEDED = "candidate_limit_exceeded"


class ReporterExactAmbiguityResolution(BaseModel):
    """Explicit result of assessing every bounded exact-lookup candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    outcome: ReporterExactAmbiguityOutcome
    passing_candidate_indices: tuple[int, ...] = ()
    selected_candidate_index: int | None = None

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        if tuple(sorted(set(self.passing_candidate_indices))) != self.passing_candidate_indices:
            raise ValueError("Passing candidate indices must be distinct and ordered")
        if any(index < 0 for index in self.passing_candidate_indices):
            raise ValueError("Passing candidate indices must be nonnegative")
        if self.outcome is ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH:
            if (
                len(self.passing_candidate_indices) != 1
                or self.selected_candidate_index != self.passing_candidate_indices[0]
            ):
                raise ValueError("A unique rule match must select its sole passing candidate")
        elif self.selected_candidate_index is not None:
            raise ValueError("Only a unique rule match selects a candidate")
        if (
            self.outcome is ReporterExactAmbiguityOutcome.NO_UNIQUE_RULE_MATCH
            and len(self.passing_candidate_indices) == 1
        ):
            raise ValueError("One passing candidate must be admitted as a unique rule match")
        if (
            self.outcome is ReporterExactAmbiguityOutcome.CANDIDATE_LIMIT_EXCEEDED
            and self.passing_candidate_indices
        ):
            raise ValueError("An unassessed large candidate set cannot report passing candidates")
        return self


class ReporterUniqueFieldAssessment(BaseModel):
    """One field comparison with an explicit, source-grounded replacement intent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    propose_replacement: bool
    quote: str | None
    result: MatchResult
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_replacement_intent(self) -> Self:
        if self.propose_replacement:
            if self.quote is None or not self.quote.strip():
                raise ValueError("A proposed replacement requires a nonempty source quote")
        elif self.quote is not None:
            raise ValueError("A field without a proposed replacement must have a null quote")
        return self


class ReporterUniqueReviewDecision(BaseModel):
    """One combined re-reading and comparison against a unique lookup record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_name: ReporterUniqueFieldAssessment
    court: ReporterUniqueFieldAssessment
    date: ReporterUniqueFieldAssessment
    reason: str = Field(min_length=1)


class ReporterUniqueReview(BaseModel):
    """Durable model decision and full IVR trace at one citation node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    decision: ReporterUniqueReviewDecision | None = None
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("Review must contain either a decision or a failure reason")
        # A successful IVR answer can still fail the stage's final source
        # grounding check. Keep that run and the boundary failure together.
        if self.ivr is not None and not self.ivr.success and self.decision is not None:
            raise ValueError("A failed IVR run cannot supply an accepted decision")
        return self
