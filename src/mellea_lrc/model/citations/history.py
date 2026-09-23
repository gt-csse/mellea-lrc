"""Citation-local decisions and append-only field values."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from mellea_lrc.model.span import Span

WITHDRAWN_ROOT_ID = "__withdrawn__"

T = TypeVar("T")


class FieldUpdate(BaseModel, Generic[T]):
    """One value, its optional source span, and its decision node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: T
    node_id: str
    span: Span | None = None


def latest(log: tuple[FieldUpdate[T], ...]) -> T | None:
    """Read the last value; an empty log has not been read yet."""
    return log[-1].value if log else None


class Node(BaseModel):
    """A citation-local decision that can update any number of fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    stage: str
