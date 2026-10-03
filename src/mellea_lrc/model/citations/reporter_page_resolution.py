"""Occurrence-specific references into a reporter root's saved page index."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import require_all_json_properties
from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind
from mellea_lrc.model.ivr import IvrRun


class ReporterPageResolutionOutcome(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNLOCATED = "unlocated"
    NO_PIN = "no_pin"
    UNNORMALIZABLE = "unnormalizable"


class OpinionPageReference(BaseModel):
    """A page stored once on the root; no duplicate opinion text on leaves."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    opinion_id: str
    page_index: int
    pagination_confirmed: bool

    @model_validator(mode="after")
    def _valid_reference(self) -> Self:
        if not self.opinion_id.isdecimal() or self.page_index < 0:
            raise ValueError("Page references need an opinion ID and nonnegative page index")
        return self


class ReporterPageCandidates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    target_index: int
    label: int
    kind: PinCiteKind
    candidates: tuple[OpinionPageReference, ...]

    @model_validator(mode="after")
    def _valid_candidates(self) -> Self:
        if self.target_index < 0 or self.label < 1:
            raise ValueError("Pinpoint targets and page labels must be valid")
        keys = [(item.opinion_id, item.page_index) for item in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate opinion-page candidate")
        return self


class ReporterCitationPageResolution(BaseModel):
    """One occurrence's locator and pinpoint histories, and its page choices.

    An unqualified Id. can inherit its immediate antecedent's pinpoint. Both
    pointers retain the original reading, rather than copying it onto the Id.
    No identity, source field, or root attachment changes in page resolution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str
    root_id: str
    locator_citation_id: str | None
    locator_reading_index: int | None
    pin_citation_id: str | None
    pin_reading_index: int | None
    outcome: ReporterPageResolutionOutcome
    pages: tuple[ReporterPageCandidates, ...] = ()
    reason: str

    @model_validator(mode="after")
    def _valid_resolution(self) -> Self:
        for identifier, index in (
            (self.locator_citation_id, self.locator_reading_index),
            (self.pin_citation_id, self.pin_reading_index),
        ):
            if (identifier is None) != (index is None) or (index is not None and index < 0):
                raise ValueError("A source reading pointer needs both citation ID and absolute index")
        if not self.reason.strip():
            raise ValueError("Page resolution needs a reason")
        if self.outcome is ReporterPageResolutionOutcome.RESOLVED and (
            not self.pages
            or any(
                len(page.candidates) != 1 or not page.candidates[0].pagination_confirmed
                for page in self.pages
            )
        ):
            raise ValueError("Rule resolution requires one pagination-matched source for every page")
        return self


class ReporterOpinionPageChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)
    page_index: int
    candidate_index: int | None


class ReporterCitationOpinionDecision(BaseModel):
    """Choose a representative for each requested page, or leave it unresolved."""

    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)
    choices: tuple[ReporterOpinionPageChoice, ...]
    reason: str

    @model_validator(mode="after")
    def _valid_choices(self) -> Self:
        if not self.reason.strip():
            raise ValueError("An opinion choice needs a reason")
        if len({choice.page_index for choice in self.choices}) != len(self.choices):
            raise ValueError("A requested page can have only one choice")
        if any(
            choice.page_index < 0 or (choice.candidate_index is not None and choice.candidate_index < 0)
            for choice in self.choices
        ):
            raise ValueError("Opinion choice indexes must be nonnegative")
        return self


class ReporterCitationOpinionReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str
    resolution_index: int
    decision: ReporterCitationOpinionDecision | None
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _valid_review(self) -> Self:
        if self.resolution_index < 0:
            raise ValueError("Opinion review must reference a page resolution")
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("Opinion review needs a decision or an explicit failure")
        if self.failure_reason is not None and not self.failure_reason.strip():
            raise ValueError("Opinion-review failure needs a reason")
        return self
