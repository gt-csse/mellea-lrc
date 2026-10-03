"""Citation-local judgments with stable references to the evidence they assessed."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameKind


def validate_replacement_quote(propose_replacement: bool, quote: str | None) -> None:
    """Keep field comparison corrections tied to an explicit quoted replacement."""
    if propose_replacement:
        if quote is None or not quote.strip():
            raise ValueError("A proposed replacement requires a nonempty source quote")
    elif quote is not None:
        raise ValueError("A field without a proposed replacement must have a null quote")


def validate_reviewer_case_name(normalized: CaseName | None) -> None:
    """An optional reviewer normalization must describe a written name."""
    if normalized is not None and normalized.kind is CaseNameKind.NOT_STATED:
        raise ValueError("A reviewer case-name normalization must describe a quoted name")


class MatchResult(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


class IdentityVerdict(str, Enum):
    CORRECT_IDENTITY = "correct_identity"
    WRONG_IDENTITY = "wrong_identity"
    PARTIALLY_CORROBORATED = "partially_corroborated"
    UNDETERMINED = "undetermined"


class IdentityBasis(str, Enum):
    THIRD_PARTY = "third_party"


class IdentityJudgment(BaseModel):
    """A durable identity opinion; routing is recorded separately."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    verdict: IdentityVerdict
    basis: IdentityBasis | None = None


class _ReporterExactFieldJudgment(BaseModel):
    """One field comparison against a cluster in the sole exact lookup response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    reading_index: int | None
    candidate_index: int
    result: MatchResult

    @model_validator(mode="after")
    def _validate_indices(self) -> Self:
        if (self.reading_index is not None and self.reading_index < 0) or self.candidate_index < 0:
            raise ValueError("Judgment references must be nonnegative absolute indices")
        if self.reading_index is None and self.result is not MatchResult.UNAVAILABLE:
            raise ValueError("A field without a filing reading is unavailable for comparison")
        return self


class ReporterExactCaseNameJudgment(_ReporterExactFieldJudgment):
    """Compare one case-name reading with one exact-lookup cluster."""


class ReporterExactCourtJudgment(_ReporterExactFieldJudgment):
    """Compare one court reading with one exact-lookup cluster."""


class ReporterExactDateJudgment(_ReporterExactFieldJudgment):
    """Compare one date reading with one exact-lookup cluster."""
