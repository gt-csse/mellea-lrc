"""Full citation occurrences identified by a reporter locator."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from mellea_lrc.model.citations.fields import FullReporterLocator
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullReporterCitation(FullCitation):
    """A reporter occurrence with an exact locator and contextual field histories."""

    kind: Literal[FullCitationKind.REPORTER] = FullCitationKind.REPORTER
    locator: Annotated[tuple[FullReporterLocator, ...], Field(min_length=1)]

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
    ) -> Self:
        """Create one grounded locator at the citation's creation node."""
        node = Node(id=f"{citation_id}:node:0", substage=substage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator=(FullReporterLocator.from_source(source, span, node_id=node.id),),
        )

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        initial = self.locator[0]
        if initial.node_id != self.nodes[0].id:
            raise ValueError("Reporter citation locator must belong to its creation node")
        if any(reading.span != initial.span or reading.quote != initial.quote for reading in self.locator):
            raise ValueError("Reporter locator history must retain its original source site")
        return self
