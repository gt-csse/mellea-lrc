"""A parsing group of adjacent full or short reporter citation occurrences."""

from pydantic import BaseModel, ConfigDict


class Colocation(BaseModel):
    """Source sites sharing a parsing window, without any identity assertion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    citation_ids: tuple[str, ...]
