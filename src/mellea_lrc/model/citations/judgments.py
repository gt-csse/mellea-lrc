"""Citation-local judgments with stable references to the evidence they assessed."""

from __future__ import annotations

import re
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class MatchResult(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNDETERMINED = "undetermined"


class IdentityVerdict(str, Enum):
    CORRECT_IDENTITY = "correct_identity"
    WRONG_IDENTITY = "wrong_identity"
    DEFERRED = "deferred"


class IdentityJudgment(BaseModel):
    """The durable overall result and route after one citation decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    verdict: IdentityVerdict
    next_stage: str | None = None

    @model_validator(mode="after")
    def _validate_route(self) -> Self:
        if (self.verdict is IdentityVerdict.DEFERRED) != (self.next_stage is not None):
            raise ValueError("Only a deferred identity judgment has a next stage")
        if self.next_stage is not None and re.fullmatch(r"[a-z][a-z0-9_]*", self.next_stage) is None:
            raise ValueError("A route must name a lowercase stage ID")
        return self


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
        if self.reading_index is None and self.result is not MatchResult.UNDETERMINED:
            raise ValueError("A field without a reading can only be undetermined")
        return self


class ReporterExactCaseNameJudgment(_ReporterExactFieldJudgment):
    """Compare one case-name reading with one exact-lookup cluster."""


class ReporterExactCourtJudgment(_ReporterExactFieldJudgment):
    """Compare one court reading with one exact-lookup cluster."""


class ReporterExactDateJudgment(_ReporterExactFieldJudgment):
    """Compare one date reading with one exact-lookup cluster."""
