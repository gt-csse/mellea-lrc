"""Source readings and attribution history of short-form citations."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.fields import CaseName, CaseNameField, CaseNameKind, PinCiteField
from mellea_lrc.model.citations.fields.base import require_all_json_properties
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span


class AttributionResult(StrEnum):
    ATTACHED = "attached"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


class LeafAttribution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str
    candidate_root_ids: tuple[str, ...]
    result: AttributionResult
    reason: str

    @model_validator(mode="after")
    def _valid_candidates(self) -> Self:
        if len(set(self.candidate_root_ids)) != len(self.candidate_root_ids):
            raise ValueError("Leaf attribution candidates must be unique")
        if not self.reason.strip():
            raise ValueError("Leaf attribution needs a reason")
        return self


class LeafReviewDecision(BaseModel):
    """The model chooses a candidate; the program writes the root attachment."""

    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)
    is_citation: bool
    root_index: int | None
    reason: str

    @model_validator(mode="after")
    def _consistent_choice(self) -> Self:
        if not self.reason.strip():
            raise ValueError("A leaf review needs a reason")
        if self.root_index is not None and (not self.is_citation or self.root_index < 0):
            raise ValueError("Only a citation may select a nonnegative root index")
        return self


class LeafReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str
    candidate_root_ids: tuple[str, ...]
    decision: LeafReviewDecision | None
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _valid_review(self) -> Self:
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("A leaf review needs a decision or a failure")
        if self.failure_reason is not None and not self.failure_reason.strip():
            raise ValueError("A failed leaf review needs a reason")
        if len(set(self.candidate_root_ids)) != len(self.candidate_root_ids):
            raise ValueError("Leaf review candidates must be unique")
        if self.decision and self.decision.root_index is not None:
            if self.decision.root_index >= len(self.candidate_root_ids):
                raise ValueError("Leaf review selected a missing root candidate")
        return self


class LeafCitation(Citation):
    """Append-only leaf fields; identity remains a property of the attached root."""

    case_name: tuple[CaseNameField, ...] = ()
    pin_cite: tuple[PinCiteField, ...] = ()
    attributions: tuple[LeafAttribution, ...] = ()
    reviews: tuple[LeafReview, ...] = ()

    def get_case_name(self) -> CaseName:
        return (
            self.case_name[-1].get_normalized() if self.case_name else CaseName(kind=CaseNameKind.NOT_STATED)
        )

    def with_case_name(self, source: str, span: Span) -> Self:
        return self._with_log(
            case_name=(
                *self.case_name,
                CaseNameField.from_source(source, span, node_id=self._decision_node_id()),
            )
        )

    def with_pin_cite(self, source: str, span: Span) -> Self:
        return self._with_log(
            pin_cite=(
                *self.pin_cite,
                PinCiteField.from_source(source, span, node_id=self._decision_node_id()),
            )
        )

    def with_attribution(self, candidates: tuple[str, ...], result: AttributionResult, reason: str) -> Self:
        return self._with_log(
            attributions=(
                *self.attributions,
                LeafAttribution(
                    node_id=self._decision_node_id(),
                    candidate_root_ids=candidates,
                    result=result,
                    reason=reason,
                ),
            )
        )

    def with_review(self, review: LeafReview) -> Self:
        if review.node_id != self._decision_node_id():
            raise ValueError("Leaf review must point to its decision node")
        return self._with_log(reviews=(*self.reviews, review))
