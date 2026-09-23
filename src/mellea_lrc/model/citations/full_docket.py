"""Full citations identified by a case docket number."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.full import FullCitation, _require_exact, _source_slice
from mellea_lrc.model.citations.history import FieldUpdate, Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullDocketCitation(FullCitation):
    """A docket occurrence with an optional adjacent entry reference."""

    kind: Literal[FullCitationKind.DOCKET] = FullCitationKind.DOCKET
    locator_text: tuple[FieldUpdate[str], ...]
    docket_number: tuple[FieldUpdate[str | None], ...] = ()
    docket_entry: tuple[FieldUpdate[str | None], ...] = ()

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
        if docket_entry is not None and (
            docket_entry_span is None or docket_entry not in _source_slice(source, docket_entry_span)
        ):
            raise ValueError("Docket entry does not match its source span")
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator_text=(FieldUpdate(value=written, span=span, node_id=node.id),),
            docket_number=(FieldUpdate(value=docket_number, span=docket_number_span, node_id=node.id),),
            docket_entry=(
                (FieldUpdate(value=docket_entry, span=docket_entry_span, node_id=node.id),)
                if docket_entry is not None
                else ()
            ),
        )

    def validate_source(self, source: str) -> None:
        super().validate_source(source)
        for update in self.locator_text:
            assert update.span is not None
            _require_exact(source, update.span, update.value)
        for update in self.docket_number:
            if update.value is not None and update.span is not None:
                _require_exact(source, update.span, update.value)
        for update in self.docket_entry:
            if update.value is not None and (
                update.span is None or update.value not in _source_slice(source, update.span)
            ):
                raise ValueError("Docket entry does not match its source span")

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
