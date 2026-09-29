"""Citation-local searches and reviews that identify a possibly intended case.

These records deliberately do not issue an identity judgment about the cited
locator. A matching name in another document can suggest an authority without
establishing that the source filing's reporter or docket identifier is right.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from mellea_lrc.model.citations.body_evidence import (
    BodyEvidence,
    BodyEvidenceFailure,
    BodySearchAttempt,
    BodySource,
)
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span


class FieldBodySearch(BaseModel):
    """One provider's bounded case-name search and fetched body excerpts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    source: BodySource
    retrospective_date: date | None
    query_name: str | None
    attempts: tuple[BodySearchAttempt, ...] = ()
    discovery_pages: tuple[dict[str, JsonValue], ...] = ()
    evidence: tuple[BodyEvidence, ...] = ()
    failures: tuple[BodyEvidenceFailure, ...] = ()

    @model_validator(mode="after")
    def _validate_search(self) -> Self:
        if self.query_name is not None and not self.query_name.strip():
            raise ValueError("A field search query cannot be blank")
        if any(item.anchor_kind != "case_name" for item in self.evidence):
            raise ValueError("Field discovery needs case-name-anchored evidence")
        if self.retrospective_date is not None and any(
            item.issued_on is None or item.issued_on > self.retrospective_date for item in self.evidence
        ):
            raise ValueError("Field evidence must be dated on or before the retrospective cutoff")
        return self


class IntendedCaseConfidence(str, Enum):
    """How strongly an external citation suggests the case the filer meant."""

    POSSIBLE = "possible"
    LIKELY = "likely"


class IntendedCaseDecision(BaseModel):
    """One grounded external citation, without a verdict on the filing's locator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: BodySource | None
    evidence_index: int | None = Field(ge=0)
    citation_quote: str | None
    case_name: str | None
    locator: str | None
    court: str | None
    date: str | None
    confidence: IntendedCaseConfidence | None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_decision(self) -> Self:
        selected = self.source is not None
        required = (self.evidence_index, self.citation_quote, self.case_name, self.confidence)
        if selected != all(value is not None for value in required):
            raise ValueError(
                "A selected intended-case candidate needs provenance, quote, name and confidence"
            )
        if not selected and any(
            value is not None
            for value in (
                self.evidence_index,
                self.citation_quote,
                self.case_name,
                self.locator,
                self.court,
                self.date,
                self.confidence,
            )
        ):
            raise ValueError("A declined candidate cannot carry extracted fields")
        for value in (self.citation_quote, self.case_name, self.locator, self.court, self.date):
            if value is not None and not value.strip():
                raise ValueError("Candidate field quotes cannot be blank")
        return self


class IntendedCaseReview(BaseModel):
    """A saved model decision and its exact anchor in a fetched document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    decision: IntendedCaseDecision | None = None
    grounded_quote: str | None = None
    quote_span: Span | None = None
    quote_similarity: float | None = Field(default=None, ge=0, le=100)
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _validate_review(self) -> Self:
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("Intended-case review needs either a decision or a failure")
        selected = self.decision is not None and self.decision.source is not None
        if selected != all(
            value is not None for value in (self.grounded_quote, self.quote_span, self.quote_similarity)
        ):
            raise ValueError("A selected intended case needs a grounded quote and span")
        if self.ivr is not None and not self.ivr.success and self.decision is not None:
            raise ValueError("A failed model run cannot provide a decision")
        return self
