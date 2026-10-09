"""Source rereading decisions assisted by an attached root's saved validation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mellea_lrc.matching.fuzziness import FuzzinessType
from mellea_lrc.model.citations.fields.base import require_all_json_properties
from mellea_lrc.model.citations.fields.case_name import CaseName
from mellea_lrc.model.citations.judgments import validate_replacement_quote, validate_reviewer_case_name
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span

LeafCorrectionField = Literal["case_name", "pin_cite", "court", "date"]
RootEvidenceHistory = Literal[
    "case_name",
    "court",
    "date",
    "locator",
    "case_name_judgments",
    "court_judgments",
    "date_judgments",
    "identity_judgments",
    "reporter_exact_lookup",
    "reporter_exact_docket",
    "reporter_exact_candidate_dockets",
    "reporter_exact_ambiguity_resolution",
    "reporter_unique_review",
    "reporter_ambiguous_review",
    "docket_lookup",
    "docket_lookup_review",
    "govinfo_docket_lookup",
    "govinfo_docket_review",
    "body_reviews",
    "intended_case_reviews",
]


class RootValidationEvidenceReference(BaseModel):
    """An exact native root-history entry reused without another lookup."""

    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)

    history: RootEvidenceHistory
    record_index: int | None = Field(default=None, ge=0)
    node_id: str
    selected_candidate_index: int | None = Field(default=None, ge=0)


class LeafCorrectionWindow(BaseModel):
    """The filing context in which one field is allowed to ground."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: LeafCorrectionField
    span: Span


class LeafFieldCorrectionProposal(BaseModel):
    """Explicit replacement intent for one source reading; never a numeric override."""

    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)

    field: LeafCorrectionField
    propose_replacement: bool
    quote: str | None
    normalized: CaseName | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_intent(self) -> Self:
        validate_replacement_quote(self.propose_replacement, self.quote)
        validate_reviewer_case_name(self.normalized)
        if self.normalized is not None and (self.field != "case_name" or not self.propose_replacement):
            raise ValueError("Only a grounded case-name replacement can supply model normalization")
        return self


class LeafFieldCorrectionDecision(BaseModel):
    """A review of every available field, including unchanged readings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fields: tuple[LeafFieldCorrectionProposal, ...]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_fields(self) -> Self:
        if len({item.field for item in self.fields}) != len(self.fields):
            raise ValueError("A field can be assessed only once in a correction decision")
        return self


class GroundedLeafFieldCorrection(BaseModel):
    """The exact source bytes resolved by the shared grounding policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: LeafCorrectionField
    quote: str = Field(min_length=1)
    span: Span
    match_type: FuzzinessType
    similarity_percent: float = Field(ge=0, le=100)
    edits: int = Field(ge=0)
    reading_index: int = Field(ge=0)
    applied: bool


class LeafFieldCorrectionReview(BaseModel):
    """A node-bound decision/failure with root evidence and field grounding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    root_id: str
    evidence_refs: tuple[RootValidationEvidenceReference, ...] = Field(min_length=1)
    windows: tuple[LeafCorrectionWindow, ...] = Field(min_length=1)
    decision: LeafFieldCorrectionDecision | None = None
    grounded: tuple[GroundedLeafFieldCorrection, ...] = ()
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _validate_review(self) -> Self:
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("Leaf correction review needs either a decision or a failure")
        if self.ivr is not None and not self.ivr.success and self.decision is not None:
            raise ValueError("A failed model run cannot supply a correction decision")
        if len({item.field for item in self.windows}) != len(self.windows):
            raise ValueError("Correction windows must name distinct fields")
        if self.decision is None:
            if self.grounded:
                raise ValueError("A failed review cannot append grounded corrections")
            return self
        if {item.field for item in self.decision.fields} != {item.field for item in self.windows}:
            raise ValueError("A correction decision must assess every allowed field")
        proposed = {item.field for item in self.decision.fields if item.propose_replacement}
        if proposed != {item.field for item in self.grounded} or len(proposed) != len(self.grounded):
            raise ValueError("Every proposed correction needs exactly one grounded result")
        windows = {item.field: item.span for item in self.windows}
        for item in self.grounded:
            window = windows[item.field]
            if not window.start <= item.span.start < item.span.end <= window.end:
                raise ValueError("A correction must ground inside its own field window")
        return self
