"""Citation-local evidence from a GovInfo USCOURTS search."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from mellea_lrc.model.citations.docket_lookup import DocketLookupFailure, DocketLookupReviewDecision
from mellea_lrc.model.ivr import IvrRun


class GovInfoLookupAttempt(BaseModel):
    """One query, its raw pages, retry failures, and any terminal failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    pages: tuple[dict[str, JsonValue], ...] = ()
    count: int | None = Field(default=None, ge=0)
    next_offset_marks: tuple[str, ...] = ()
    retry_failures: tuple[DocketLookupFailure, ...] = ()
    failure: DocketLookupFailure | None = None


class GovInfoLookupCandidate(BaseModel):
    """A pointer to one exact result in the saved GovInfo response pages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_index: int = Field(ge=0)
    page_index: int = Field(ge=0)
    result_index: int = Field(ge=0)
    package_id: str | None = None
    granule_id: str | None = None
    court_code: str | None = None
    docket_number: str | None = None
    docket_similarity: float = Field(ge=0, le=100)


class GovInfoDocketLookup(BaseModel):
    """One docket root's saved GovInfo query trace and number shortlist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    attempts: tuple[GovInfoLookupAttempt, ...] = ()
    candidates: tuple[GovInfoLookupCandidate, ...] = ()
    shortlisted_candidate_indices: tuple[int, ...] = ()
    failure: DocketLookupFailure | None = None

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        if not self.attempts and self.failure is None:
            raise ValueError("A GovInfo lookup needs a query attempt or preparation failure")
        locations: set[tuple[int, int, int]] = set()
        for candidate in self.candidates:
            if candidate.attempt_index >= len(self.attempts):
                raise ValueError("Candidate points to a missing GovInfo attempt")
            attempt = self.attempts[candidate.attempt_index]
            if candidate.page_index >= len(attempt.pages):
                raise ValueError("Candidate points to a missing GovInfo page")
            results = attempt.pages[candidate.page_index].get("results")
            if not isinstance(results, list) or candidate.result_index >= len(results):
                raise ValueError("Candidate points to a missing GovInfo search result")
            result = results[candidate.result_index]
            if not isinstance(result, dict):
                raise ValueError("Candidate GovInfo result must be an object")
            if candidate.package_id != _result_string(result, "packageId", "package_id"):
                raise ValueError("Candidate package ID must match its raw result")
            if candidate.granule_id != _result_string(result, "granuleId", "granule_id"):
                raise ValueError("Candidate granule ID must match its raw result")
            location = (candidate.attempt_index, candidate.page_index, candidate.result_index)
            if location in locations:
                raise ValueError("A GovInfo result cannot appear twice in the candidate list")
            locations.add(location)
        if tuple(sorted(set(self.shortlisted_candidate_indices))) != self.shortlisted_candidate_indices:
            raise ValueError("Shortlisted candidate indices must be distinct and ordered")
        if any(index < 0 or index >= len(self.candidates) for index in self.shortlisted_candidate_indices):
            raise ValueError("Shortlist points to a missing GovInfo candidate")
        return self


class GovInfoDocketReview(BaseModel):
    """A later model review of the GovInfo shortlist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    decision: DocketLookupReviewDecision | None = None
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("GovInfo review must contain either a decision or a failure reason")
        if self.failure_reason is not None and not self.failure_reason.strip():
            raise ValueError("Review failure reason must be nonempty")
        if self.ivr is not None and not self.ivr.success and self.decision is not None:
            raise ValueError("A failed IVR run cannot supply an accepted decision")
        return self


def _result_string(result: dict[str, JsonValue], *keys: str) -> str | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None
