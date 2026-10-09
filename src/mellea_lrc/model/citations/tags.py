"""Occurrence metadata carried from the preprocessed source components."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CitationTagKind(StrEnum):
    TABLE_OF_AUTHORITIES = "table_of_authorities"


class CitationTag(BaseModel):
    """A creation-node tag referencing one source component's range."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    kind: CitationTagKind
    component_index: int = Field(ge=0)
