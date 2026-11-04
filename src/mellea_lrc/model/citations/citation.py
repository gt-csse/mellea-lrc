"""Immutable citation occurrences with local decisions and typed field histories."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mellea_lrc.model.citations.fields import (
    CaseName,
    CaseNameField,
    CaseNameKind,
    PinCiteField,
    PinCiteValue,
)
from mellea_lrc.model.citations.fields.base import CitationField, NotStatedField, current_reading
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.span import Span


class Citation(BaseModel):
    """An immutable occurrence whose readings point to its decision nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, pattern=r"\S")
    kind: str
    nodes: tuple[Node, ...]
    case_name: tuple[CaseNameField | NotStatedField, ...] = ()
    pin_cite: Annotated[tuple[PinCiteField | NotStatedField, ...], Field(min_length=1)] | None = None

    @property
    def site_span(self) -> Span:
        """The written source site used to order citation occurrences."""
        raise NotImplementedError

    def record(self, substage: str) -> Self:
        """Append a decision node that named field updates can share."""
        node = Node(id=f"{self.id}:node:{len(self.nodes)}", substage=substage)
        return self._with_log(nodes=(*self.nodes, node))

    def _decision_node_id(self) -> str:
        if len(self.nodes) < 2:
            raise ValueError("Record a decision node before changing citation fields")
        return self.nodes[-1].id

    def _with_log(self, **logs: object) -> Self:
        """Revalidate every immutable update, including ownership and ordering."""
        return type(self).model_validate({**self.model_dump(mode="python"), **logs})

    def _through_node_count(self, count: int) -> Self:
        """Recover the citation and its readings through a decision boundary."""
        if not 1 <= count <= len(self.nodes):
            raise ValueError("Citation cutoff must retain its creation node")
        nodes = self.nodes[:count]
        node_ids = {node.id for node in nodes}
        data = {**self.model_dump(mode="python"), "nodes": nodes}
        for name, field in type(self).model_fields.items():
            if name in {"id", "kind", "nodes"}:
                continue
            retained = tuple(reading for reading in getattr(self, name) or () if reading.node_id in node_ids)
            data[name] = retained if retained else field.get_default()
        return type(self).model_validate(data)

    def with_case_name(self, source: str, span: Span, *, normalized: CaseName | None = None) -> Self:
        """Append an exact source quote with deterministic or model normalization."""
        reading = (
            CaseNameField.from_source(source, span, node_id=self._decision_node_id())
            if normalized is None
            else CaseNameField.from_model(source, span, normalized, node_id=self._decision_node_id())
        )
        return self._with_log(case_name=(*self.case_name, reading))

    def with_case_name_not_stated(self) -> Self:
        """Append absence without removing a previously quoted name."""
        reading = NotStatedField(node_id=self._decision_node_id())
        return self._with_log(case_name=(*self.case_name, reading))

    def get_case_name(self) -> CaseName:
        """Read the current name or its explicit unstated form."""
        reading = current_reading(self.case_name)
        return CaseName(kind=CaseNameKind.NOT_STATED) if reading is None else reading.get_normalized()

    def with_pin_cite(self, source: str, span: Span) -> Self:
        """Append a grounded pinpoint using the existing typed normalizer."""
        reading = PinCiteField.from_source(source, span, node_id=self._decision_node_id())
        return self._with_log(pin_cite=(*(self.pin_cite or ()), reading))

    def with_pin_cite_not_stated(self) -> Self:
        """Append absence without removing earlier pinpoint readings."""
        reading = NotStatedField(node_id=self._decision_node_id())
        return self._with_log(pin_cite=(*(self.pin_cite or ()), reading))

    def get_pin_cite(self) -> PinCiteValue | None:
        """Read current targets; an unnormalizable quoted pinpoint raises."""
        reading = current_reading(self.pin_cite)
        return None if reading is None else reading.get_normalized()

    def validate_source(self, source: str) -> None:
        """Check every historical quote against the owning source after loading."""
        for name in type(self).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            log = getattr(self, name)
            for reading in log or ():
                if isinstance(reading, CitationField):
                    reading.validate_source(source)

    @model_validator(mode="after")
    def _validate_history(self) -> Self:
        if not self.nodes:
            raise ValueError("Citation must have a creation node")
        positions = {node.id: index for index, node in enumerate(self.nodes)}
        if len(positions) != len(self.nodes):
            raise ValueError("Duplicate citation node")
        for index, node in enumerate(self.nodes):
            if node.id != f"{self.id}:node:{index}":
                raise ValueError("Decision node ID must belong to its citation and position")
        for name in type(self).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            previous = -1
            for reading in getattr(self, name) or ():
                position = positions.get(reading.node_id)
                if position is None:
                    raise ValueError(f"{name} refers to a missing node")
                if position <= previous:
                    raise ValueError(f"{name} updates are out of order")
                if position == 0 and name != "locator":
                    raise ValueError(f"{name} requires a later decision node")
                previous = position
        return self
