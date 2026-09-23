"""Fields shared by every occurrence with a full case locator."""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.span import Span


class FullCitationKind(str, Enum):
    REPORTER = "reporter"
    DOCKET = "docket"


class FullCitation(BaseModel):
    """One written full citation occurrence, before or after root formation.

    Create initializes a concrete subtype with only its ID and kind. Later
    operations fill source-grounded fields without changing that subtype.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: FullCitationKind
    locator_span: Span | None = None
    locator_text: str | None = None
    case_name: str | None = None
    case_name_span: Span | None = None
    court: str | None = None
    court_span: Span | None = None
    date: CitationDate | None = None
    date_span: Span | None = None
    pin_cite: str | None = None
    pin_cite_span: Span | None = None
    colocation_id: str | None = None
    root_id: str | None = None
