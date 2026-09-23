"""Stage-neutral decisions and durable changes to citation state."""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from mellea_lrc.model.citations import CitationDate, FullCitationKind
from mellea_lrc.model.span import Span


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


class OperationKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"


FieldValue = Span | CitationDate | str | None
WITHDRAWN_ROOT_ID = "__withdrawn__"


class Operation(BaseModel):
    """One durable create or field update made by a decision node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    node_id: str
    kind: OperationKind
    citation_id: str
    citation_kind: FullCitationKind | None = None
    field: CitationField | None = None
    value: FieldValue = None


class Node(BaseModel):
    """One stage decision; it may materialize several field operations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    stage: str
    citation_id: str
    operation_ids: tuple[str, ...]
