"""Citation-local decisions and current values from append-only field histories."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from mellea_lrc.model.citations.fields.base import CitationField, NotStatedField, current_reading

T = TypeVar("T")


class Node(BaseModel):
    """A citation-local decision that can update several distinct fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, pattern=r"\S")
    substage: str = Field(pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){0,2}$")


def latest(log: tuple[CitationField[T] | NotStatedField, ...] | None) -> T | None:
    """Read the latest normalized value; a failed latest reading still raises."""
    reading = current_reading(log)
    return None if reading is None else reading.get_normalized()
