"""A supra reference and its quoted antecedent."""

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields.short_forms import SupraField
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import ShortCitationKind
from mellea_lrc.model.citations.leaf import LeafCitation
from mellea_lrc.model.span import Span


class SupraCitation(LeafCitation):
    kind: Literal[ShortCitationKind.SUPRA] = ShortCitationKind.SUPRA
    supra_reference: tuple[SupraField, ...]

    @property
    def site_span(self) -> Span:
        return self.supra_reference[-1].span

    @classmethod
    def from_source(cls, *, source: str, span: Span, substage: str) -> Self:
        identifier = f"supra:{span.start}:{span.end}"
        node = Node(id=f"{identifier}:node:0", substage=substage)
        return cls(
            id=identifier,
            nodes=(node,),
            supra_reference=(SupraField.from_source(source, span, node_id=node.id),),
        )

    @model_validator(mode="after")
    def _validate_creation(self) -> Self:
        if not self.supra_reference or self.supra_reference[0].node_id != self.nodes[0].id:
            raise ValueError("A supra citation needs its source reading on the creation node")
        return self
