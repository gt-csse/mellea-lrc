"""Full citations identified by a case docket number."""

from typing import Literal

from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import FieldUpdate
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullDocketCitation(FullCitation):
    """A docket occurrence with an optional adjacent entry reference."""

    kind: Literal[FullCitationKind.DOCKET] = FullCitationKind.DOCKET
    locator_span: tuple[FieldUpdate[Span | None], ...] = ()
    locator_text: tuple[FieldUpdate[str | None], ...] = ()
    docket_number: tuple[FieldUpdate[str | None], ...] = ()
    docket_entry: tuple[FieldUpdate[str | None], ...] = ()
    docket_entry_span: tuple[FieldUpdate[Span | None], ...] = ()
