"""Citation-local evidence for reviewing docket-root equivalence."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mellea_lrc.model.ivr import IvrRun


class DocketRootPartition(BaseModel):
    """A model's partition of candidate indices into same-case groups.

    Singleton groups explicitly mean that a candidate remains a separate root.
    The stage checks that the partition covers its particular candidate set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    groups: tuple[tuple[int, ...], ...] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_groups(self) -> Self:
        indices = [index for group in self.groups for index in group]
        if any(not group for group in self.groups):
            raise ValueError("Every docket-root group needs at least one candidate")
        if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("Candidate indices must be distinct and nonnegative")
        return self


class DocketRootReview(BaseModel):
    """One component review and its complete IVR trace on its first citation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    candidate_ids: tuple[str, ...] = Field(min_length=2)
    decision: DocketRootPartition | None = None
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Docket review candidate IDs must be distinct")
        if (self.decision is None) == (self.failure_reason is None):
            raise ValueError("Docket review needs either a partition or a failure reason")
        if self.ivr is not None and not self.ivr.success and self.decision is not None:
            raise ValueError("A failed IVR run cannot supply an accepted partition")
        if self.decision is not None:
            actual = {index for group in self.decision.groups for index in group}
            if actual != set(range(len(self.candidate_ids))):
                raise ValueError("Docket review must partition every candidate exactly once")
        return self
