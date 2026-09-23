"""Shared field histories and explicit changes to full citations."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.citations.fields import (
    CaseNameField,
    CitationField,
    CourtField,
    DateField,
    PinCiteField,
)
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID, Node, RelationshipUpdate
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


def _source_slice(source: str, span: Span) -> str:
    if span.end > len(source) or span.start == span.end:
        raise ValueError("Source span is outside the document or empty")
    return source[span.start : span.end]


def _require_exact(source: str, span: Span, written: str) -> None:
    if _source_slice(source, span) != written:
        raise ValueError("Field text does not match its source span")


class FullCitation(BaseModel):
    """A full citation with field-local logs and citation-local decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: FullCitationKind
    nodes: tuple[Node, ...]
    case_name: tuple[CaseNameField, ...] = ()
    court: tuple[CourtField, ...] = ()
    date: tuple[DateField, ...] = ()
    pin_cite: tuple[PinCiteField, ...] = ()
    colocation_id: tuple[RelationshipUpdate[str | None], ...] = ()
    root_id: tuple[RelationshipUpdate[str | None], ...] = ()

    def record(self, stage: str) -> Self:
        """Record one decision; named field methods may share its node."""
        node = Node(id=f"{self.id}:node:{len(self.nodes)}", stage=stage)
        return type(self).model_validate({**self.model_dump(mode="python"), "nodes": (*self.nodes, node)})

    def _decision_node_id(self) -> str:
        if len(self.nodes) == 1:
            raise ValueError("Record a decision node before changing citation fields")
        return self.nodes[-1].id

    def _with_log(self, **logs: tuple[CitationField | RelationshipUpdate, ...]) -> Self:
        """Validate the immutable citation after a named field method changes it."""
        return type(self).model_validate({**self.model_dump(mode="python"), **logs})

    def _through_node_count(self, count: int) -> Self:
        """Recover the citation and field readings through one node boundary."""
        if not 1 <= count <= len(self.nodes):
            raise ValueError("Citation cutoff must retain its creation node")
        nodes = self.nodes[:count]
        node_ids = {node.id for node in nodes}
        data = {**self.model_dump(mode="python"), "nodes": nodes}
        for name in type(self).model_fields:
            if name not in {"id", "kind", "nodes"}:
                data[name] = tuple(update for update in getattr(self, name) if update.node_id in node_ids)
        return type(self).model_validate(data)

    def with_case_name(self, source: str, span: Span, *, normalized: str) -> Self:
        """Quote a case name and record its current interpretation."""
        return self._with_log(
            case_name=(
                *self.case_name,
                CaseNameField.from_source(
                    source, span, normalized=normalized, node_id=self._decision_node_id()
                ),
            ),
        )

    def with_court(self, source: str, span: Span | None, *, normalized: str) -> Self:
        """Quote an explicit court or record a reporter-inferred court."""
        reading = (
            CourtField.from_source(source, span, normalized=normalized, node_id=self._decision_node_id())
            if span is not None
            else CourtField.inferred(normalized, node_id=self._decision_node_id())
        )
        return self._with_log(
            court=(*self.court, reading),
        )

    def with_date(self, source: str, span: Span, *, normalized: CitationDate) -> Self:
        """Quote a written date and record its parsed calendar components."""
        return self._with_log(
            date=(
                *self.date,
                DateField.from_source(source, span, normalized=normalized, node_id=self._decision_node_id()),
            ),
        )

    def with_pin_cite(self, source: str, span: Span, *, normalized: str) -> Self:
        """Quote a pinpoint reference and record its parsed value."""
        return self._with_log(
            pin_cite=(
                *self.pin_cite,
                PinCiteField.from_source(
                    source, span, normalized=normalized, node_id=self._decision_node_id()
                ),
            ),
        )

    def with_colocation(self, group_id: str) -> Self:
        """Append a parsing-group assignment."""
        return self._with_log(
            colocation_id=(
                *self.colocation_id,
                RelationshipUpdate(value=group_id, node_id=self._decision_node_id()),
            ),
        )

    def with_root(self, root_id: str) -> Self:
        """Append a root attachment without erasing earlier assignments."""
        return self._with_log(
            root_id=(*self.root_id, RelationshipUpdate(value=root_id, node_id=self._decision_node_id())),
        )

    def withdraw(self) -> Self:
        """Reattach to the dummy head, retaining all previous values."""
        return self.with_root(WITHDRAWN_ROOT_ID)

    def validate_source(self, source: str) -> None:
        """Check all stored quotes against the source, including after JSON loading."""
        for name in type(self).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            for entry in getattr(self, name):
                if isinstance(entry, CitationField):
                    entry.validate_source(source)

    @model_validator(mode="after")
    def _validate_history(self) -> Self:
        if not self.nodes:
            raise ValueError("Full citation must have a creation node")
        positions = {node.id: index for index, node in enumerate(self.nodes)}
        if len(positions) != len(self.nodes):
            raise ValueError("Duplicate citation node")
        for name in type(self).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            previous = -1
            for update in getattr(self, name):
                position = positions.get(update.node_id)
                if position is None:
                    raise ValueError(f"{name} refers to a missing node")
                if position <= previous:
                    raise ValueError(f"{name} updates are out of order")
                previous = position
        return self
