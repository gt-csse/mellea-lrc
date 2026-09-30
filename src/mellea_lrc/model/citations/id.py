"""An Id. occurrence, kept distinct from a reporter locator."""

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields.short_forms import IdField
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import ShortCitationKind
from mellea_lrc.model.citations.leaf import LeafCitation
from mellea_lrc.model.span import Span


class IdCitation(LeafCitation):
    kind: Literal[ShortCitationKind.ID] = ShortCitationKind.ID
    id_reference: tuple[IdField, ...]

    @property
    def site_span(self) -> Span:
        return self.id_reference[-1].span

    @classmethod
    def from_source(cls, *, source: str, span: Span, stage: str) -> Self:
        identifier = f"id:{span.start}:{span.end}"
        node = Node(id=f"{identifier}:node:0", stage=stage)
        return cls(
            id=identifier, nodes=(node,), id_reference=(IdField.from_source(source, span, node_id=node.id),)
        )

    @model_validator(mode="after")
    def _validate_creation(self) -> Self:
        if not self.id_reference or self.id_reference[0].node_id != self.nodes[0].id:
            raise ValueError("A id citation needs its source reading on the creation node")
        return self
