"""Full citation occurrences identified by a docket locator."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from mellea_lrc.model.citations.fields import DocketEntryField, FullDocketLocator
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullDocketCitation(FullCitation):
    """A docket occurrence retaining locator and adjacent entry readings."""

    kind: Literal[FullCitationKind.DOCKET] = FullCitationKind.DOCKET
    locator: Annotated[tuple[FullDocketLocator, ...], Field(min_length=1)]
    docket_entry: tuple[DocketEntryField, ...] = ()

    @property
    def locator_span(self) -> Span:
        return self.locator[0].span

    @classmethod
    def from_locator(
        cls,
        *,
        citation_id: str,
        substage: str,
        source: str,
        span: Span,
        number_span: Span,
    ) -> Self:
        """Create one grounded docket locator at the citation's creation node."""
        node = Node(id=f"{citation_id}:node:0", substage=substage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator=(FullDocketLocator.from_source(source, span, number_span, node_id=node.id),),
        )

    def with_docket_entry(self, source: str, span: Span) -> Self:
        """Append an adjacent entry reading under the current decision node."""
        reading = DocketEntryField.from_source(source, span, node_id=self._decision_node_id())
        return self._with_log(docket_entry=(*self.docket_entry, reading))

    def with_docket_number(self, source: str, number_span: Span) -> Self:
        """Reread the number within the original locator, retaining prior readings."""
        reading = FullDocketLocator.from_source(
            source, self.locator_span, number_span, node_id=self._decision_node_id()
        )
        return self._with_log(locator=(*self.locator, reading))

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        initial = self.locator[0]
        if initial.node_id != self.nodes[0].id:
            raise ValueError("Docket citation locator must belong to its creation node")
        if any(reading.span != initial.span or reading.quote != initial.quote for reading in self.locator):
            raise ValueError("Docket locator history must retain its original source site")
        return self
