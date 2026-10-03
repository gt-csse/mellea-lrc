"""Occurrence-level proposition, opinion evidence, and pinpoint judgments.

Opinion bodies live on the root. These records retain source spans and stable
history indexes; they do not copy bodies or change the root's identity verdict.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, StrictBool, model_validator

from mellea_lrc.model.citations.fields.base import require_all_json_properties
from mellea_lrc.model.citations.fields.pin_cite import PinCiteTarget
from mellea_lrc.model.citations.reporter_page_resolution import OpinionPageReference
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span


class GroundedPassage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    quote: str
    span: Span

    @model_validator(mode="after")
    def _valid_quote(self) -> Self:
        if not self.quote.strip() or len(self.quote) != self.span.end - self.span.start:
            raise ValueError("A grounded passage needs a nonempty exact source quote")
        return self

    def validate_source(self, source: str) -> None:
        if source[self.span.start : self.span.end] != self.quote:
            raise ValueError("Grounded passage differs from its source span")


class PropositionDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)
    quotes: tuple[str, ...]
    reason: str

    @model_validator(mode="after")
    def _valid_decision(self) -> Self:
        if not self.reason.strip() or any(not quote.strip() for quote in self.quotes):
            raise ValueError("Proposition quotes and the reason must be nonempty")
        return self


class ReporterCitationProposition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str
    resolution_index: int
    decision: PropositionDecision | None
    passages: tuple[GroundedPassage, ...] = ()
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _valid_record(self) -> Self:
        if self.resolution_index < 0:
            raise ValueError("Proposition must reference an existing page resolution")
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("Proposition reading needs a decision or explicit failure")
        if self.decision is None and self.passages:
            raise ValueError("Failed proposition reading cannot retain accepted passages")
        if self.decision is not None and len(self.passages) != len(self.decision.quotes):
            raise ValueError("Every proposed proposition quote must have one grounded passage")
        return self


class PinpointEvidenceOutcome(StrEnum):
    READY = "ready"
    MISSING_PAGES = "missing_pages"
    NO_PROPOSITION = "no_proposition"
    NO_PINCITE = "no_pincite"
    UNNORMALIZABLE = "unnormalizable"
    READING_FAILED = "reading_failed"


class ReporterCitationPinpointEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str
    root_id: str
    resolution_index: int
    proposition_index: int | None
    pages: tuple[OpinionPageReference, ...]
    outcome: PinpointEvidenceOutcome
    reason: str

    @model_validator(mode="after")
    def _valid_evidence(self) -> Self:
        if self.resolution_index < 0 or (self.proposition_index is not None and self.proposition_index < 0):
            raise ValueError("Pinpoint evidence needs valid absolute history indexes")
        if not self.reason.strip():
            raise ValueError("Pinpoint evidence needs a reason")
        if self.outcome is PinpointEvidenceOutcome.READY and (
            self.proposition_index is None or not self.pages
        ):
            raise ValueError("Ready evidence requires proposition and page references")
        return self


class OpinionSupportResult(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


class OpinionReviewScope(StrEnum):
    CITED_PAGES = "cited_pages"
    FULL_OPINION = "full_opinion"


class OpinionEvidenceQuote(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)
    opinion_id: str
    quote: str

    @model_validator(mode="after")
    def _valid_quote(self) -> Self:
        if not self.opinion_id.isdecimal() or not self.quote.strip():
            raise ValueError("Opinion evidence needs a valid opinion ID and nonempty quote")
        return self


class PinpointPageAssessment(BaseModel):
    """Placement in the cited pagination, independently of content support.

    Availability describes the source used for this assessment. Recovered
    pages are known locations, not an exhaustive assertion of every location.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)
    pagination_available: StrictBool
    correct_page: StrictBool | None
    found_pages: tuple[PinCiteTarget, ...]

    @model_validator(mode="after")
    def _valid_pagination(self) -> Self:
        if not self.pagination_available and (self.correct_page is not None or self.found_pages):
            raise ValueError("Unavailable pagination requires correct_page=null and found_pages=[]")
        return self


class ReporterSupportDecision(PinpointPageAssessment):
    """The model judges the attributed use; code grounds every evidence quote."""

    result: OpinionSupportResult
    evidence: tuple[OpinionEvidenceQuote, ...]
    reason: str

    @model_validator(mode="after")
    def _valid_decision(self) -> Self:
        if not self.reason.strip():
            raise ValueError("Opinion support review requires a prose reason")
        if (
            self.result in {OpinionSupportResult.SUPPORTED, OpinionSupportResult.CONTRADICTED}
            and not self.evidence
        ):
            raise ValueError("Support or contradiction needs quoted opinion evidence")
        if any(not evidence.quote.strip() for evidence in self.evidence):
            raise ValueError("Evidence quotes must be nonempty")
        return self


class ReporterOpinionEvidence(GroundedPassage):
    """A source-grounded excerpt in a root-owned indexed opinion."""

    node_id: str
    root_id: str
    opinion_id: str


class ReporterCitationSupportReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str
    evidence_index: int
    scope: OpinionReviewScope
    decision: ReporterSupportDecision | None
    opinion_evidence_indices: tuple[int, ...] = ()
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _valid_review(self) -> Self:
        if self.evidence_index < 0 or any(index < 0 for index in self.opinion_evidence_indices):
            raise ValueError("Support review needs valid absolute history indexes")
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("Support review requires a decision or explicit failure")
        if self.decision is None and self.opinion_evidence_indices:
            raise ValueError("Failed support review cannot retain accepted opinion evidence references")
        if self.decision is not None and len(self.decision.evidence) != len(self.opinion_evidence_indices):
            raise ValueError("Every opinion quote needs one accepted evidence reference")
        return self


class ReporterPinpointVerdict(StrEnum):
    CORRECT_PINCITE = "CORRECT_PINCITE"
    WRONG_PINCITE = "WRONG_PINCITE"
    UNDETERMINED = "UNDETERMINED"


class ReporterPinpointJudgment(PinpointPageAssessment):
    """Citation support and the precision of the written target stay explicit.

    CORRECT_PINCITE follows the dataset's support convention: evidence elsewhere
    can support the attribution while correct_page records a wrong or unknown page.
    A failure to retrieve or locate text never alone becomes WRONG_PINCITE.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str
    evidence_index: int
    review_index: int | None
    verdict: ReporterPinpointVerdict
    reason: str

    @model_validator(mode="after")
    def _valid_judgment(self) -> Self:
        if self.evidence_index < 0 or (self.review_index is not None and self.review_index < 0):
            raise ValueError("Pinpoint judgment needs valid absolute history indexes")
        if not self.reason.strip():
            raise ValueError("Pinpoint judgment needs a reason")
        return self
