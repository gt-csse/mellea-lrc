"""Shared field histories and explicit changes to full citations."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID, FieldUpdate, Node
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
    case_name: tuple[FieldUpdate[str | None], ...] = ()
    court: tuple[FieldUpdate[str | None], ...] = ()
    date: tuple[FieldUpdate[CitationDate | None], ...] = ()
    pin_cite: tuple[FieldUpdate[str | None], ...] = ()
    colocation_id: tuple[FieldUpdate[str | None], ...] = ()
    root_id: tuple[FieldUpdate[str | None], ...] = ()

    def record(self, stage: str) -> Self:
        """Record one decision; named field methods may share its node."""
        node = Node(id=f"{self.id}:node:{len(self.nodes)}", stage=stage)
        return type(self).model_validate({**self.model_dump(mode="python"), "nodes": (*self.nodes, node)})

    def _decision_node_id(self) -> str:
        if len(self.nodes) == 1:
            raise ValueError("Record a decision node before changing citation fields")
        return self.nodes[-1].id

    def _with_log(self, **logs: tuple[FieldUpdate, ...]) -> Self:
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

    def with_case_name(self, source: str, name: str, span: Span) -> Self:
        """Append a case name only when the document says exactly that text."""
        _require_exact(source, span, name)
        return self._with_log(
            case_name=(*self.case_name, FieldUpdate(value=name, span=span, node_id=self._decision_node_id())),
        )

    def with_court(self, source: str, court: str, span: Span | None = None) -> Self:
        """Append a normalized court ID with its written span, if explicit."""
        if span is not None:
            _source_slice(source, span)
        return self._with_log(
            court=(*self.court, FieldUpdate(value=court, span=span, node_id=self._decision_node_id())),
        )

    def with_date(self, source: str, date: CitationDate, span: Span) -> Self:
        """Append a parsed date tied to written source evidence."""
        if str(date.year) not in _source_slice(source, span):
            raise ValueError("Date year does not match its source span")
        return self._with_log(
            date=(*self.date, FieldUpdate(value=date, span=span, node_id=self._decision_node_id())),
        )

    def with_pin_cite(self, source: str, pin: str, span: Span) -> Self:
        """Append an exact pin-cite reading from the document."""
        _require_exact(source, span, pin)
        return self._with_log(
            pin_cite=(*self.pin_cite, FieldUpdate(value=pin, span=span, node_id=self._decision_node_id())),
        )

    def with_colocation(self, group_id: str) -> Self:
        """Append a parsing-group assignment."""
        return self._with_log(
            colocation_id=(
                *self.colocation_id,
                FieldUpdate(value=group_id, node_id=self._decision_node_id()),
            ),
        )

    def with_root(self, root_id: str) -> Self:
        """Append a root attachment without erasing earlier assignments."""
        return self._with_log(
            root_id=(*self.root_id, FieldUpdate(value=root_id, node_id=self._decision_node_id())),
        )

    def withdraw(self) -> Self:
        """Reattach to the dummy head, retaining all previous values."""
        return self.with_root(WITHDRAWN_ROOT_ID)

    def validate_source(self, source: str) -> None:
        """Check literal readings after a JSON round trip as well as at write time."""
        for log in (self.case_name, self.pin_cite):
            for update in log:
                if update.value is not None:
                    if update.span is None:
                        raise ValueError("Literal field has no source span")
                    _require_exact(source, update.span, update.value)
        for update in self.court:
            if update.span is not None:
                _source_slice(source, update.span)
        for update in self.date:
            if update.value is not None and update.span is not None:
                if str(update.value.year) not in _source_slice(source, update.span):
                    raise ValueError("Date year does not match its source span")

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
