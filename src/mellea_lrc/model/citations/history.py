"""Citation-local decisions and append-only field values."""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict


class CitationField(str, Enum):
    LOCATOR_SPAN = "locator_span"
    LOCATOR_TEXT = "locator_text"
    VOLUME = "volume"
    REPORTER = "reporter"
    PAGE = "page"
    DOCKET_NUMBER = "docket_number"
    DOCKET_ENTRY = "docket_entry"
    DOCKET_ENTRY_SPAN = "docket_entry_span"
    CASE_NAME = "case_name"
    CASE_NAME_SPAN = "case_name_span"
    COURT = "court"
    COURT_SPAN = "court_span"
    DATE = "date"
    DATE_SPAN = "date_span"
    PIN_CITE = "pin_cite"
    PIN_CITE_SPAN = "pin_cite_span"
    COLOCATION_ID = "colocation_id"
    ROOT_ID = "root_id"


WITHDRAWN_ROOT_ID = "__withdrawn__"

T = TypeVar("T")


class FieldUpdate(BaseModel, Generic[T]):
    """One value appended to a field, pointing to its decision node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: T
    node_id: str


def latest(log: tuple[FieldUpdate[T], ...]) -> T | None:
    """Read the last value; an empty log has not been read yet."""
    return log[-1].value if log else None


class Node(BaseModel):
    """A citation-local decision that can update any number of fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    stage: str
