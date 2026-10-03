"""Short reporter citations are leaves, not full locator occurrences."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields import PinCiteField, ShortReporterLocator
from mellea_lrc.model.citations.history import Node, RelationshipUpdate
from mellea_lrc.model.citations.kind import ShortCitationKind
from mellea_lrc.model.citations.leaf import LeafCitation
from mellea_lrc.model.span import Span


class ShortReporterCitation(LeafCitation):
    """An eyecite short case citation awaiting root attachment."""

    kind: Literal[ShortCitationKind.REPORTER] = ShortCitationKind.REPORTER
    short_locator: tuple[ShortReporterLocator, ...]
    colocation_id: tuple[RelationshipUpdate[str | None], ...] = ()

    @property
    def short_locator_span(self) -> Span:
        return self.short_locator[-1].span

    @property
    def site_span(self) -> Span:
        return self.short_locator_span

    @classmethod
    def from_short_locator(
        cls,
        *,
        citation_id: str,
        stage: str,
        source: str,
        span: Span,
        pin_cite_span: Span | None = None,
    ) -> Self:
        """Create one short citation from eyecite's source-grounded site."""
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        return cls(
            id=citation_id,
            nodes=(node,),
            short_locator=(ShortReporterLocator.from_source(source, span, node_id=node.id),),
            pin_cite=(PinCiteField.from_source(source, pin_cite_span, node_id=node.id),)
            if pin_cite_span is not None
            else None,
        )

    def with_colocation(self, group_id: str | None) -> Self:
        """Append a parsing-group assignment, independently of root attachment."""
        return self._with_log(
            colocation_id=(
                *self.colocation_id,
                RelationshipUpdate(value=group_id, node_id=self._decision_node_id()),
            )
        )

    @model_validator(mode="after")
    def _validate_short_locator(self) -> Self:
        if not self.short_locator or self.short_locator[0].node_id != self.nodes[0].id:
            raise ValueError("Short reporter citation needs a source-spanned short locator")
        return self
