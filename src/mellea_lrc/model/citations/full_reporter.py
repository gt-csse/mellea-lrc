"""Full citations identified by a reporter or legal database locator."""

from typing import Literal

from mellea_lrc.model.citations.full import FullCitation, FullCitationKind


class FullReporterCitation(FullCitation):
    """A volume, reporter, and first-page occurrence."""

    kind: Literal[FullCitationKind.REPORTER] = FullCitationKind.REPORTER
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
