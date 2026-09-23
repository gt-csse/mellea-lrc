"""A parsing group of adjacent full citation occurrences."""

from pydantic import BaseModel, ConfigDict


class Colocation(BaseModel):
    """Locators sharing a citation site, without any identity assertion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    citation_ids: tuple[str, ...]
