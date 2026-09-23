"""Full citations identified by a case docket number."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields import DocketEntryField, DocketNumberField, LocatorField
from mellea_lrc.model.citations.full import FullCitation, _require_exact, _source_slice
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullDocketCitation(FullCitation):
    """A docket occurrence with an optional adjacent entry reference."""

    kind: Literal[FullCitationKind.DOCKET] = FullCitationKind.DOCKET
    locator_text: tuple[LocatorField, ...]
    docket_number: tuple[DocketNumberField, ...] = ()
    docket_entry: tuple[DocketEntryField, ...] = ()

    @property
    def locator_span(self) -> Span:
        span = self.locator_text[-1].span
        assert span is not None
        return span

    @classmethod
    def from_locator(
        cls,
        *,
        citation_id: str,
        stage: str,
        source: str,
        span: Span,
        docket_number: str,
        docket_number_span: Span,
        docket_entry: str | None = None,
        docket_entry_span: Span | None = None,
    ) -> Self:
        """Create the citation and grounded identifier readings together."""
        written = _source_slice(source, span)
        if not (span.start <= docket_number_span.start < docket_number_span.end <= span.end):
            raise ValueError("Docket number span must be inside the locator")
        _require_exact(source, docket_number_span, docket_number)
        entry_quote = _source_slice(source, docket_entry_span) if docket_entry_span is not None else None
        if docket_entry is not None and (entry_quote is None or docket_entry not in entry_quote):
            raise ValueError("Docket entry does not match its source span")
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator_text=(LocatorField.from_source(source, span, normalized=written, node_id=node.id),),
            docket_number=(
                DocketNumberField.from_source(
                    source,
                    docket_number_span,
                    normalized=docket_number,
                    node_id=node.id,
                ),
            ),
            docket_entry=(
                (
                    DocketEntryField.from_source(
                        source,
                        docket_entry_span,
                        normalized=docket_entry,
                        node_id=node.id,
                    ),
                )
                if docket_entry is not None
                else ()
            ),
        )

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        if (
            not self.locator_text
            or not self.docket_number
            or self.locator_text[0].node_id != self.nodes[0].id
            or self.docket_number[0].node_id != self.nodes[0].id
            or self.docket_number[0].span is None
            or any(update.span is None for update in self.locator_text)
        ):
            raise ValueError("Docket citation needs a source-spanned locator")
        return self
