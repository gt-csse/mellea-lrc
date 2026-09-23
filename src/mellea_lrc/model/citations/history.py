"""Citation-local decisions and append-only field values."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from mellea_lrc.model.citations.fields import CitationField

WITHDRAWN_ROOT_ID = "__withdrawn__"

T = TypeVar("T")


class RelationshipUpdate(BaseModel, Generic[T]):
    """A non-text relationship assignment at a decision node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: T
    node_id: str


def latest(log: tuple[CitationField[T] | RelationshipUpdate[T], ...]) -> T | None:
    """Read the newest value; None means an empty log or a null relationship."""
    if not log:
        return None
    entry = log[-1]
    return entry.get_normalized() if isinstance(entry, CitationField) else entry.value


class Node(BaseModel):
    """A citation-local decision that can update any number of fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    stage: str
