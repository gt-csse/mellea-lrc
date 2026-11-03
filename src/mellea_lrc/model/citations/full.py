"""Common contextual fields of full reporter and docket citations."""

from __future__ import annotations

from typing import Self

from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.fields import CourtField, DateField
from mellea_lrc.model.citations.fields.base import NotStatedField
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullCitation(Citation):
    """A full citation with append-only written or inferred contextual fields."""

    kind: FullCitationKind
    court: tuple[CourtField | NotStatedField, ...] = ()
    date: tuple[DateField | NotStatedField, ...] = ()

    @property
    def locator_span(self) -> Span:
        """The fixed written locator site supplied by each full citation kind."""
        raise NotImplementedError

    @property
    def site_span(self) -> Span:
        return self.locator_span

    def with_court(self, source: str, span: Span) -> Self:
        """Append a written court reading and its typed normalization."""
        reading = CourtField.from_source(source, span, node_id=self._decision_node_id())
        return self._with_log(court=(*self.court, reading))

    def with_inferred_court(self, court_id: str) -> Self:
        """Append an inferred court without inventing a quote or span."""
        reading = CourtField.inferred(court_id, node_id=self._decision_node_id())
        return self._with_log(court=(*self.court, reading))

    def with_court_not_stated(self) -> Self:
        reading = NotStatedField(node_id=self._decision_node_id())
        return self._with_log(court=(*self.court, reading))

    def with_date(self, source: str, span: Span) -> Self:
        """Append a written date while retaining its available precision."""
        reading = DateField.from_source(source, span, node_id=self._decision_node_id())
        return self._with_log(date=(*self.date, reading))

    def with_date_not_stated(self) -> Self:
        reading = NotStatedField(node_id=self._decision_node_id())
        return self._with_log(date=(*self.date, reading))
