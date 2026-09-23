"""The typed page or paragraph targets of a pinpoint citation."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator


class PinCiteKind(StrEnum):
    PAGE = "page"
    STAR = "star"
    PARAGRAPH = "paragraph"


class PinCiteTarget(BaseModel):
    """One inclusive pinpoint range, independent of its written notation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first: int
    last: int
    kind: PinCiteKind
    footnote: str | None = None

    @model_validator(mode="after")
    def _valid_range(self) -> PinCiteTarget:
        if self.first < 1 or self.last < self.first:
            raise ValueError("Pin cite pages must form a positive, ascending range")
        if self.footnote == "":
            raise ValueError("Pin cite footnote cannot be empty")
        return self


PinCiteValue: TypeAlias = tuple[PinCiteTarget, ...]
