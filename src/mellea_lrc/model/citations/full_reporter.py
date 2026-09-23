"""Full citations identified by a reporter or legal database locator."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.full import FullCitation, _require_exact, _source_slice
from mellea_lrc.model.citations.history import FieldUpdate, Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullReporterCitation(FullCitation):
    """A reporter occurrence; its locator text owns the source span."""

    kind: Literal[FullCitationKind.REPORTER] = FullCitationKind.REPORTER
    locator_text: tuple[FieldUpdate[str], ...]
    volume: tuple[FieldUpdate[str | None], ...] = ()
    reporter: tuple[FieldUpdate[str | None], ...] = ()
    page: tuple[FieldUpdate[str | None], ...] = ()

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
        volume: str | None,
        reporter: str | None,
        page: str | None,
    ) -> Self:
        """Create the citation and initial readings in one decision node."""
        written = _source_slice(source, span)
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator_text=(FieldUpdate(value=written, span=span, node_id=node.id),),
            volume=(FieldUpdate(value=volume, node_id=node.id),),
            reporter=(FieldUpdate(value=reporter, node_id=node.id),),
            page=(FieldUpdate(value=page, node_id=node.id),),
        )

    def validate_source(self, source: str) -> None:
        super().validate_source(source)
        for update in self.locator_text:
            assert update.span is not None
            _require_exact(source, update.span, update.value)

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        if (
            not self.locator_text
            or self.locator_text[0].node_id != self.nodes[0].id
            or any(update.span is None for update in self.locator_text)
        ):
            raise ValueError("Reporter citation needs a source-spanned locator")
        return self
