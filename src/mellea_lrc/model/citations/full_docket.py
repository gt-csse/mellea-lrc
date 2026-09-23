"""Full citations identified by a case docket number."""

from typing import Literal

from mellea_lrc.model.citations.full import FullCitation, FullCitationKind
from mellea_lrc.model.span import Span


class FullDocketCitation(FullCitation):
    """A docket occurrence with an optional adjacent entry reference."""

    kind: Literal[FullCitationKind.DOCKET] = FullCitationKind.DOCKET
    docket_number: str | None = None
    docket_entry: str | None = None
    docket_entry_span: Span | None = None
