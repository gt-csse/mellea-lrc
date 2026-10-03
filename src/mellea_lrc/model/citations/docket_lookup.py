"""Citation-local evidence from a CourtListener docket search and review."""

from __future__ import annotations

from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from mellea_lrc.model.citations.fields.base import require_all_json_properties
from mellea_lrc.model.citations.fields.case_name import CaseName
from mellea_lrc.model.citations.judgments import (
    MatchResult,
    validate_replacement_quote,
    validate_reviewer_case_name,
)
from mellea_lrc.model.ivr import IvrRun

DocketSearchSource: TypeAlias = Literal["d", "o"]


def _raw_identifier(result: dict[str, JsonValue], source_type: DocketSearchSource) -> str | None:
    keys = ("docket_id", "docketId", "id") if source_type == "d" else ("cluster_id", "clusterId", "id")
    for key in keys:
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


def _raw_docket_number(result: dict[str, JsonValue]) -> str | None:
    for key in ("docketNumber", "docket_number"):
        value = result.get(key)
        if isinstance(value, str):
            return value
    return None


class DocketLookupFailure(BaseModel):
    """A retained search failure, including the available upstream detail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    upstream_status_code: int | None = None
    url: str | None = None
    upstream_detail: JsonValue = None


class DocketLookupAttempt(BaseModel):
    """One search query, every returned page, and any eventual failure.

    The page dictionaries retain the full CourtListener JSON, including fields
    this pipeline does not currently interpret. A failure after pagination
    leaves preceding pages available for audit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: DocketSearchSource
    query: str = Field(min_length=1)
    pages: tuple[dict[str, JsonValue], ...] = ()
    retry_failures: tuple[DocketLookupFailure, ...] = ()
    failure: DocketLookupFailure | None = None


class DocketLookupCandidate(BaseModel):
    """A lightweight pointer into one saved search page.

    The three indices identify the exact raw result, even when CourtListener
    provides no record ID. The similarity is a 0-100 docket-number score and
    is evidence for the stage's shortlist, not a case-identity verdict.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: DocketSearchSource
    record_id: str | None = None
    attempt_index: int = Field(ge=0)
    page_index: int = Field(ge=0)
    result_index: int = Field(ge=0)
    docket_number: str | None = None
    docket_similarity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _validate_record_id(self) -> Self:
        if self.record_id is not None and not self.record_id.strip():
            raise ValueError("A candidate record ID must be nonempty when present")
        return self


class DocketLookup(BaseModel):
    """One root's complete search trace and its docket-number shortlist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    attempts: tuple[DocketLookupAttempt, ...] = ()
    candidates: tuple[DocketLookupCandidate, ...] = ()
    shortlisted_candidate_indices: tuple[int, ...] = ()
    failure: DocketLookupFailure | None = None

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        if not self.attempts and self.failure is None:
            raise ValueError("A docket lookup needs a query attempt or preparation failure")
        locations: set[tuple[int, int, int]] = set()
        for candidate in self.candidates:
            if candidate.attempt_index >= len(self.attempts):
                raise ValueError("Candidate points to a missing search attempt")
            attempt = self.attempts[candidate.attempt_index]
            if candidate.source_type != attempt.source_type:
                raise ValueError("Candidate source type must match its search attempt")
            if candidate.page_index >= len(attempt.pages):
                raise ValueError("Candidate points to a missing search page")
            results = attempt.pages[candidate.page_index].get("results")
            if not isinstance(results, list) or candidate.result_index >= len(results):
                raise ValueError("Candidate points to a missing search result")
            raw_result = results[candidate.result_index]
            if not isinstance(raw_result, dict):
                raise ValueError("Candidate search result must be an object")
            if candidate.record_id != _raw_identifier(raw_result, candidate.source_type):
                raise ValueError("Candidate record ID must match its raw search result")
            if candidate.docket_number != _raw_docket_number(raw_result):
                raise ValueError("Candidate docket number must match its raw search result")
            location = (candidate.attempt_index, candidate.page_index, candidate.result_index)
            if location in locations:
                raise ValueError("A search result cannot appear twice in the candidate list")
            locations.add(location)
        if tuple(sorted(set(self.shortlisted_candidate_indices))) != self.shortlisted_candidate_indices:
            raise ValueError("Shortlisted candidate indices must be distinct and ordered")
        if any(index < 0 or index >= len(self.candidates) for index in self.shortlisted_candidate_indices):
            raise ValueError("Shortlist points to a missing lookup candidate")
        return self


class DocketLookupFieldAssessment(BaseModel):
    """One comparison and an optional grounded correction to the filing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    propose_replacement: bool
    quote: str | None
    result: MatchResult
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_replacement_intent(self) -> Self:
        validate_replacement_quote(self.propose_replacement, self.quote)
        return self


class DocketLookupCaseNameAssessment(DocketLookupFieldAssessment):
    """A filing name and its model reading, independent of the record caption."""

    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)

    normalized: CaseName | None = None

    @model_validator(mode="after")
    def _validate_grounded_name(self) -> Self:
        validate_reviewer_case_name(self.normalized)
        return self


class DocketLookupReviewDecision(BaseModel):
    """A structured choice of one shortlisted candidate or none."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_candidate_index: int | None
    docket_number: DocketLookupFieldAssessment
    case_name: DocketLookupCaseNameAssessment
    court: DocketLookupFieldAssessment
    date: DocketLookupFieldAssessment
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        if self.selected_candidate_index is not None and self.selected_candidate_index < 0:
            raise ValueError("Selected candidate index must be nonnegative")
        if self.selected_candidate_index is None and any(
            assessment.result in {MatchResult.MATCH, MatchResult.MISMATCH}
            for assessment in (self.docket_number, self.case_name, self.court, self.date)
        ):
            raise ValueError("No selection cannot make candidate-relative field judgments")
        return self


class DocketLookupReview(BaseModel):
    """A later node's model decision and complete instruct/validate/repair trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    decision: DocketLookupReviewDecision | None = None
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("Review must contain either a decision or a failure reason")
        if self.failure_reason is not None and not self.failure_reason.strip():
            raise ValueError("Review failure reason must be nonempty")
        if self.ivr is not None and not self.ivr.success and self.decision is not None:
            raise ValueError("A failed IVR run cannot supply an accepted decision")
        return self
