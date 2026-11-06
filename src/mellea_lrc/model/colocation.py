"""A parsing group of adjacent citation occurrences."""

from pydantic import BaseModel, ConfigDict


class Colocation(BaseModel):
    """Source sites sharing a parsing window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    citation_ids: tuple[str, ...]
