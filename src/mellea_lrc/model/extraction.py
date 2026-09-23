"""Durable, natively serializable extraction state.

A node records why a stage acted; its operations record what changed. The
materialized citations make ordinary reads cheap, while the append-only
operations retain every update needed to replay a checkpoint.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.preprocessed import PreprocessedDocument
from mellea_lrc.model.spans import Span


class CitationKind(str, Enum):
    REPORTER = "reporter"
    DOCKET = "docket"


class CitationField(str, Enum):
    LOCATOR_SPAN = "locator_span"
    LOCATOR_TEXT = "locator_text"
    VOLUME = "volume"
    REPORTER = "reporter"
    PAGE = "page"
    DOCKET_NUMBER = "docket_number"
    DOCKET_ENTRY = "docket_entry"
    DOCKET_ENTRY_SPAN = "docket_entry_span"
    CASE_NAME = "case_name"
    CASE_NAME_SPAN = "case_name_span"
    COURT = "court"
    COURT_SPAN = "court_span"
    DATE = "date"
    DATE_SPAN = "date_span"
    PIN_CITE = "pin_cite"
    PIN_CITE_SPAN = "pin_cite_span"
    COLOCATION_ID = "colocation_id"
    ROOT_ID = "root_id"


class OperationKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"


class CitationDate(BaseModel):
    """A written decision date, possibly only a year."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    month: int | None = None
    day: int | None = None


class Citation(BaseModel):
    """One written full locator occurrence, before or after root formation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: CitationKind
    locator_span: Span | None = None
    locator_text: str | None = None
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    docket_number: str | None = None
    docket_entry: str | None = None
    docket_entry_span: Span | None = None
    case_name: str | None = None
    case_name_span: Span | None = None
    court: str | None = None
    court_span: Span | None = None
    date: CitationDate | None = None
    date_span: Span | None = None
    pin_cite: str | None = None
    pin_cite_span: Span | None = None
    colocation_id: str | None = None
    root_id: str | None = None


FieldValue = Span | CitationDate | str | None
WITHDRAWN_ROOT_ID = "__withdrawn__"


class Operation(BaseModel):
    """One durable create or field update made by a stage node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    node_id: str
    kind: OperationKind
    citation_id: str
    citation_kind: CitationKind | None = None
    field: CitationField | None = None
    value: FieldValue = None


class Node(BaseModel):
    """One stage decision; it may materialize several field operations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    stage: str
    citation_id: str
    operation_ids: tuple[str, ...]


class Colocation(BaseModel):
    """Adjacent full locators sharing a citation site, not proven identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    citation_ids: tuple[str, ...]


class Document(PreprocessedDocument):
    """A complete extraction save; each later stage includes all earlier state."""

    citations: tuple[Citation, ...] = ()
    colocations: tuple[Colocation, ...] = ()
    nodes: tuple[Node, ...] = ()
    operations: tuple[Operation, ...] = ()
    completed_stages: tuple[str, ...] = ()

    @classmethod
    def from_preprocessed(cls, source: PreprocessedDocument) -> Self:
        """Start extraction without changing the preprocessed text or spans."""
        return cls.model_validate(source.model_dump(mode="python"))

    @classmethod
    def from_source(cls, source: Path | str) -> Self:
        """Preprocess a file path or supplied text, then start extraction."""
        from mellea_lrc.preprocessing import preprocess

        return cls.from_preprocessed(preprocess(source))

    @property
    def full_locators(self) -> tuple[Citation, ...]:
        """Every detected occurrence, including later duplicates of one root."""
        return tuple(citation for citation in self.citations if citation.locator_span is not None)

    @property
    def roots(self) -> tuple[Citation, ...]:
        """Canonical first occurrences after root formation."""
        return tuple(citation for citation in self.citations if citation.root_id == citation.id)

    def create_citation(self, stage: str, citation_id: str, kind: CitationKind) -> Self:
        """Initialize only the citation type; fields are separate updates."""
        return self._record(stage, citation_id, create_kind=kind)

    def update_fields(self, stage: str, citation_id: str, changes: dict[CitationField, FieldValue]) -> Self:
        """Write any number of fields through one decision node."""
        return self._record(stage, citation_id, changes=changes)

    def withdraw_citation(self, stage: str, citation_id: str) -> Self:
        """Reattach a citation to the dummy head without erasing its history."""
        return self.update_fields(stage, citation_id, {CitationField.ROOT_ID: WITHDRAWN_ROOT_ID})

    def _record(
        self,
        stage: str,
        citation_id: str,
        *,
        create_kind: CitationKind | None = None,
        changes: dict[CitationField, FieldValue] | None = None,
    ) -> Self:
        """Atomically append a decision and materialize its field operations."""
        current = {citation.id: citation for citation in self.citations}
        if create_kind is None and citation_id not in current:
            raise KeyError(f"Unknown citation: {citation_id}")
        if create_kind is not None and citation_id in current:
            raise ValueError(f"Citation already exists: {citation_id}")
        node_id = f"{stage}:{citation_id}:{len(self.nodes)}"
        operations: list[Operation] = []
        if create_kind is not None:
            current[citation_id] = Citation(id=citation_id, kind=create_kind)
            operations.append(
                Operation(
                    id=f"op:{len(self.operations) + len(operations)}",
                    node_id=node_id,
                    kind=OperationKind.CREATE,
                    citation_id=citation_id,
                    citation_kind=create_kind,
                )
            )
        citation = current[citation_id]
        for field, value in (changes or {}).items():
            if getattr(citation, field.value) == value:
                continue
            citation = Citation.model_validate({**citation.model_dump(mode="python"), field.value: value})
            operations.append(
                Operation(
                    id=f"op:{len(self.operations) + len(operations)}",
                    node_id=node_id,
                    kind=OperationKind.UPDATE,
                    citation_id=citation_id,
                    field=field,
                    value=value,
                )
            )
        if not operations:
            return self
        current[citation_id] = citation
        node = Node(
            id=node_id,
            stage=stage,
            citation_id=citation_id,
            operation_ids=tuple(operation.id for operation in operations),
        )
        ordered = tuple(
            sorted(
                current.values(),
                key=lambda item: (item.locator_span.start if item.locator_span else len(self.text), item.id),
            )
        )
        return self.model_copy(
            update={
                "citations": ordered,
                "nodes": (*self.nodes, node),
                "operations": (*self.operations, *operations),
            }
        )

    def complete(self, stage: str, *, colocations: tuple[Colocation, ...] | None = None) -> Self:
        """Mark an inspectable stage complete, including one with no findings."""
        if stage in self.completed_stages:
            return self
        changes: dict[str, object] = {"completed_stages": (*self.completed_stages, stage)}
        if colocations is not None:
            changes["colocations"] = colocations
        return self.model_copy(update=changes)

    @model_validator(mode="after")
    def _validate_replay(self) -> Self:
        """Reject a checkpoint whose visible citations disagree with its log."""
        replay: dict[str, Citation] = {}
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node in extraction log")
        by_node: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for operation in self.operations:
            if operation.node_id not in by_node:
                raise ValueError("Extraction operation has no decision node")
            by_node[operation.node_id].append(operation.id)
            if operation.kind is OperationKind.CREATE:
                if operation.citation_kind is None or operation.citation_id in replay:
                    raise ValueError("Invalid citation creation in extraction log")
                replay[operation.citation_id] = Citation(
                    id=operation.citation_id, kind=operation.citation_kind
                )
            else:
                if operation.field is None or operation.citation_id not in replay:
                    raise ValueError("Invalid field update in extraction log")
                citation = replay[operation.citation_id]
                replay[operation.citation_id] = Citation.model_validate(
                    {**citation.model_dump(mode="python"), operation.field.value: operation.value}
                )
        if any(tuple(by_node[node.id]) != node.operation_ids for node in self.nodes):
            raise ValueError("Extraction node disagrees with its operations")
        if replay != {citation.id: citation for citation in self.citations}:
            raise ValueError("Citation state disagrees with extraction operations")
        ids = set(replay)
        grouped: dict[str, list[str]] = {}
        for citation in self.citations:
            if citation.colocation_id is not None:
                grouped.setdefault(citation.colocation_id, []).append(citation.id)
        listed = {group.id: group.citation_ids for group in self.colocations}
        if {key: tuple(value) for key, value in grouped.items()} != listed:
            raise ValueError("Colocation groups disagree with citation state")
        for group in self.colocations:
            if len(group.citation_ids) < 2 or any(identifier not in ids for identifier in group.citation_ids):
                raise ValueError("Invalid colocation reference")
            if any(replay[identifier].colocation_id != group.id for identifier in group.citation_ids):
                raise ValueError("Colocation disagrees with citation state")
        return self
