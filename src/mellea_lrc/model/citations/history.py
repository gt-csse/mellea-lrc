"""Citation-local decisions and current values from append-only field histories."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from mellea_lrc.model.citations.fields.base import CitationField, NotStatedField

T = TypeVar("T")


class RelationshipUpdate(BaseModel, Generic[T]):
    """A non-text relationship assignment at a citation-local decision node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: T
    node_id: str


class Node(BaseModel):
    """A citation-local decision that can update several distinct fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, pattern=r"\S")
    substage: str = Field(pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){0,2}$")


def latest(log: tuple[CitationField[T] | NotStatedField | RelationshipUpdate[T], ...] | None) -> T | None:
    """Read the latest normalized value; a failed latest reading still raises."""
    if not log or isinstance(log[-1], NotStatedField):
        return None
    reading = log[-1]
    return reading.value if isinstance(reading, RelationshipUpdate) else reading.get_normalized()
