"""The written date attached to a full citation."""

from pydantic import BaseModel, ConfigDict


class CitationDate(BaseModel):
    """A decision date, possibly only a year."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    month: int | None = None
    day: int | None = None
