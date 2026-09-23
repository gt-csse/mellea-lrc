"""Full citations identified by a case docket number."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields import DocketEntryField, FullDocketLocator
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullDocketCitation(FullCitation):
    """A docket occurrence with an optional adjacent entry reference."""

    kind: Literal[FullCitationKind.DOCKET] = FullCitationKind.DOCKET
    locator: tuple[FullDocketLocator, ...]
    docket_entry: tuple[DocketEntryField, ...] = ()

    @property
    def locator_span(self) -> Span:
        return self.locator[-1].span

    @classmethod
    def from_locator(
        cls,
        *,
        citation_id: str,
        stage: str,
        source: str,
        span: Span,
        number_span: Span,
        docket_entry_span: Span | None = None,
    ) -> Self:
        """Create the citation and grounded identifier readings together."""
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator=(FullDocketLocator.from_source(source, span, number_span, node_id=node.id),),
            docket_entry=(
                (DocketEntryField.from_source(source, docket_entry_span, node_id=node.id),)
                if docket_entry_span is not None
                else ()
            ),
        )

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        if (
            not self.locator
            or self.locator[0].node_id != self.nodes[0].id
            or (self.docket_entry and self.docket_entry[0].node_id != self.nodes[0].id)
        ):
            raise ValueError("Docket citation needs a source-spanned locator")
        return self
