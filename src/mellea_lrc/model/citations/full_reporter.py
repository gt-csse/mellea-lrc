"""Full citations identified by a reporter or legal database locator."""

from typing import Literal

from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import FieldUpdate
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullReporterCitation(FullCitation):
    """A volume, reporter, and first-page occurrence."""

    kind: Literal[FullCitationKind.REPORTER] = FullCitationKind.REPORTER
    locator_span: tuple[FieldUpdate[Span | None], ...] = ()
    locator_text: tuple[FieldUpdate[str | None], ...] = ()
    volume: tuple[FieldUpdate[str | None], ...] = ()
    reporter: tuple[FieldUpdate[str | None], ...] = ()
    page: tuple[FieldUpdate[str | None], ...] = ()
