"""Shared append-only fields for full citations."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.citations.history import (
    WITHDRAWN_ROOT_ID,
    CitationField,
    FieldUpdate,
    Node,
)
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullCitation(BaseModel):
    """A typed citation whose mutable fields are their own complete histories.

    The first node creates the citation. Later nodes explain field updates;
    several field entries may point to the same decision. No current-value
    cache or second operation log exists.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: FullCitationKind
    nodes: tuple[Node, ...]
    case_name: tuple[FieldUpdate[str | None], ...] = ()
    case_name_span: tuple[FieldUpdate[Span | None], ...] = ()
    court: tuple[FieldUpdate[str | None], ...] = ()
    court_span: tuple[FieldUpdate[Span | None], ...] = ()
    date: tuple[FieldUpdate[CitationDate | None], ...] = ()
    date_span: tuple[FieldUpdate[Span | None], ...] = ()
    pin_cite: tuple[FieldUpdate[str | None], ...] = ()
    pin_cite_span: tuple[FieldUpdate[Span | None], ...] = ()
    colocation_id: tuple[FieldUpdate[str | None], ...] = ()
    root_id: tuple[FieldUpdate[str | None], ...] = ()

    @classmethod
    def create(cls, citation_id: str, stage: str) -> Self:
        """Create only the concrete type; fields can be read later."""
        if cls is FullCitation:
            raise TypeError("Create a concrete full citation subtype")
        return cls(id=citation_id, nodes=(Node(id=f"{citation_id}:node:0", stage=stage),))

    def update_fields(self, stage: str, changes: dict[CitationField, object]) -> Self:
        """Append field values from one decision without copying current state."""
        if not changes:
            return self
        node = Node(id=f"{self.id}:node:{len(self.nodes)}", stage=stage)
        data = self.model_dump(mode="python")
        data["nodes"] = (*self.nodes, node)
        for field, value in changes.items():
            if field.value not in type(self).model_fields:
                raise ValueError(f"{type(self).__name__} has no {field.value} field")
            data[field.value] = (*getattr(self, field.value), FieldUpdate(value=value, node_id=node.id))
        return type(self).model_validate(data)

    def withdraw(self, stage: str) -> Self:
        """Retain the former root assignments and append the dummy head."""
        return self.update_fields(stage, {CitationField.ROOT_ID: WITHDRAWN_ROOT_ID})

    @model_validator(mode="after")
    def _validate_history(self) -> Self:
        """Reject dangling or time-reversed field-to-node references."""
        if not self.nodes:
            raise ValueError("Full citation must have a creation node")
        positions = {node.id: index for index, node in enumerate(self.nodes)}
        if len(positions) != len(self.nodes):
            raise ValueError("Duplicate citation node")
        for field in CitationField:
            name = field.value
            if name not in type(self).model_fields:
                continue
            log = getattr(self, name)
            previous = 0
            for update in log:
                position = positions.get(update.node_id)
                if position is None or position == 0:
                    raise ValueError(f"{name} refers to a missing or creation node")
                if position < previous:
                    raise ValueError(f"{name} updates are out of order")
                previous = position
        return self
