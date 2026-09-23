"""The written date attached to a full citation."""

from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class CitationDate(BaseModel):
    """A decision date, possibly only a year."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    month: int | None = None
    day: int | None = None

    @model_validator(mode="after")
    def _validate_calendar_date(self) -> Self:
        if self.day is not None and self.month is None:
            raise ValueError("Citation date cannot have a day without a month")
        try:
            date(
                self.year,
                self.month if self.month is not None else 1,
                self.day if self.day is not None else 1,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid citation date: {exc}") from exc
        return self
