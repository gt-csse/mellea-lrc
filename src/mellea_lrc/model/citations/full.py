"""Common contextual fields of full reporter and docket citations."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.fields import CourtField, DateField
from mellea_lrc.model.citations.fields.base import NotStatedField
from mellea_lrc.model.citations.history import RelationshipUpdate
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullCitation(Citation):
    """A full citation with append-only written or inferred contextual fields."""

    kind: FullCitationKind
    court: tuple[CourtField | NotStatedField, ...] = ()
    date: tuple[DateField | NotStatedField, ...] = ()
    colocation_id: tuple[RelationshipUpdate[str | None], ...] = ()
    root_id: tuple[RelationshipUpdate[str | None], ...] = ()

    def with_root(self, root_id: str | None) -> Self:
        """Append a root assignment under the current decision node."""
        return self._with_log(
            root_id=(*self.root_id, RelationshipUpdate(value=root_id, node_id=self._decision_node_id()))
        )

    def with_colocation(self, group_id: str) -> Self:
        """Append a parsing-group assignment under the current decision node."""
        return self._with_log(
            colocation_id=(
                *self.colocation_id,
                RelationshipUpdate(value=group_id, node_id=self._decision_node_id()),
            )
        )

    @model_validator(mode="after")
    def _validate_colocation_ids(self) -> Self:
        if any(update.value is not None and not update.value.strip() for update in self.colocation_id):
            raise ValueError("Colocation ID must not be blank")
        return self

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
