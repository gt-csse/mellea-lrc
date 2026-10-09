"""A case-name reference; its written name identifies its source site."""

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields import CaseNameField, PinCiteField
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import ShortCitationKind
from mellea_lrc.model.citations.leaf import LeafCitation
from mellea_lrc.model.span import Span


class ReferenceCitation(LeafCitation):
    kind: Literal[ShortCitationKind.REFERENCE] = ShortCitationKind.REFERENCE
    reference_name: tuple[CaseNameField, ...]

    @property
    def site_span(self) -> Span:
        return self.reference_name[-1].span

    @classmethod
    def from_source(cls, *, source: str, span: Span, substage: str, pin_span: Span | None = None) -> Self:
        identifier = f"reference:{span.start}:{span.end}"
        node = Node(id=f"{identifier}:node:0", substage=substage)
        name = CaseNameField.from_source(source, span, node_id=node.id)
        return cls(
            id=identifier,
            nodes=(node,),
            reference_name=(name,),
            case_name=(name,),
            pin_cite=(PinCiteField.from_source(source, pin_span, node_id=node.id),)
            if pin_span is not None
            else None,
        )

    @model_validator(mode="after")
    def _validate_creation(self) -> Self:
        if not self.reference_name or self.reference_name[0].node_id != self.nodes[0].id:
            raise ValueError("A reference citation needs its source reading on the creation node")
        if not self.case_name or self.case_name[0] != self.reference_name[0]:
            raise ValueError("A reference citation must retain its initial case-name reading")
        return self
