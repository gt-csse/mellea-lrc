"""The cumulative, natively serializable document across pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import model_validator

from mellea_lrc.model.citations import (
    FullCitation,
    FullCitationKind,
    FullCitationVariant,
    full_citation_type,
)
from mellea_lrc.model.colocation import Colocation
from mellea_lrc.model.operations import (
    WITHDRAWN_ROOT_ID,
    CitationField,
    FieldValue,
    Node,
    Operation,
    OperationKind,
)
from mellea_lrc.model.preprocessed_document import PreprocessedDocument


class Document(PreprocessedDocument):
    """A complete save whose materialized citations replay from its operations."""

    citations: tuple[FullCitationVariant, ...] = ()
    colocations: tuple[Colocation, ...] = ()
    nodes: tuple[Node, ...] = ()
    operations: tuple[Operation, ...] = ()
    completed_stages: tuple[str, ...] = ()

    @classmethod
    def from_preprocessed(cls, source: PreprocessedDocument) -> Self:
        """Start citation work without changing source text or offsets."""
        return cls.model_validate(source.model_dump(mode="python"))

    @classmethod
    def from_source(cls, source: Path | str) -> Self:
        """Preprocess a file path or supplied text, then start citation work."""
        from mellea_lrc.preprocessing import preprocess

        return cls.from_preprocessed(preprocess(source))

    @property
    def full_locators(self) -> tuple[FullCitation, ...]:
        """Every detected full citation occurrence, including repeated roots."""
        return tuple(citation for citation in self.citations if citation.locator_span is not None)

    @property
    def roots(self) -> tuple[FullCitation, ...]:
        """Canonical first occurrences after root formation."""
        return tuple(citation for citation in self.citations if citation.root_id == citation.id)

    def create_citation(self, stage: str, citation_id: str, kind: FullCitationKind) -> Self:
        """Initialize a concrete full citation type without filling its fields."""
        return self._record(stage, citation_id, create_kind=kind)

    def update_fields(self, stage: str, citation_id: str, changes: dict[CitationField, FieldValue]) -> Self:
        """Write any number of type-checked fields through one decision node."""
        return self._record(stage, citation_id, changes=changes)

    def withdraw_citation(self, stage: str, citation_id: str) -> Self:
        """Reattach a citation to the dummy head without erasing its history."""
        return self.update_fields(stage, citation_id, {CitationField.ROOT_ID: WITHDRAWN_ROOT_ID})

    def _record(
        self,
        stage: str,
        citation_id: str,
        *,
        create_kind: FullCitationKind | None = None,
        changes: dict[CitationField, FieldValue] | None = None,
    ) -> Self:
        """Append a decision and materialize its create or field operations."""
        current = {citation.id: citation for citation in self.citations}
        if create_kind is None and citation_id not in current:
            raise KeyError(f"Unknown citation: {citation_id}")
        if create_kind is not None and citation_id in current:
            raise ValueError(f"Citation already exists: {citation_id}")
        node_id = f"{stage}:{citation_id}:{len(self.nodes)}"
        operations: list[Operation] = []
        if create_kind is not None:
            current[citation_id] = full_citation_type(create_kind)(id=citation_id)
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
            if field.value not in type(citation).model_fields:
                raise ValueError(f"{type(citation).__name__} has no {field.value} field")
            if getattr(citation, field.value) == value:
                continue
            citation = type(citation).model_validate(
                {**citation.model_dump(mode="python"), field.value: value}
            )
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
        """Mark an inspectable stage complete, even if it found nothing."""
        if stage in self.completed_stages:
            return self
        changes: dict[str, object] = {"completed_stages": (*self.completed_stages, stage)}
        if colocations is not None:
            changes["colocations"] = colocations
        return self.model_copy(update=changes)

    @model_validator(mode="after")
    def _validate_replay(self) -> Self:
        """Reject a checkpoint that does not replay to its visible state."""
        replay: dict[str, FullCitation] = {}
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node in document log")
        if len({operation.id for operation in self.operations}) != len(self.operations):
            raise ValueError("Duplicate operation in document log")
        by_node: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for operation in self.operations:
            if operation.node_id not in by_node:
                raise ValueError("Operation has no decision node")
            by_node[operation.node_id].append(operation.id)
            if operation.kind is OperationKind.CREATE:
                if operation.citation_kind is None or operation.citation_id in replay:
                    raise ValueError("Invalid full citation creation in document log")
                replay[operation.citation_id] = full_citation_type(operation.citation_kind)(
                    id=operation.citation_id
                )
            else:
                if operation.field is None or operation.citation_id not in replay:
                    raise ValueError("Invalid field update in document log")
                citation = replay[operation.citation_id]
                if operation.field.value not in type(citation).model_fields:
                    raise ValueError("Field does not belong to citation type")
                replay[operation.citation_id] = type(citation).model_validate(
                    {**citation.model_dump(mode="python"), operation.field.value: operation.value}
                )
        if any(tuple(by_node[node.id]) != node.operation_ids for node in self.nodes):
            raise ValueError("Decision node disagrees with its operations")
        if len({citation.id for citation in self.citations}) != len(self.citations):
            raise ValueError("Duplicate citation in document state")
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
