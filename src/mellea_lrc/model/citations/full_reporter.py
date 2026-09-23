"""Full citations identified by a reporter or legal database locator."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields import FullReporterLocator
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullReporterCitation(FullCitation):
    """A reporter occurrence whose one locator field owns all locator parts."""

    kind: Literal[FullCitationKind.REPORTER] = FullCitationKind.REPORTER
    locator: tuple[FullReporterLocator, ...]

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
    ) -> Self:
        """Create one source-grounded locator and its normalized Reporter."""
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator=(FullReporterLocator.from_source(source, span, node_id=node.id),),
        )

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        if not self.locator or self.locator[0].node_id != self.nodes[0].id:
            raise ValueError("Reporter citation needs a source-spanned locator")
        return self
