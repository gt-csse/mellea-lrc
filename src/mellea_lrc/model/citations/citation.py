"""Citation-local history shared by full citations and short forms."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID, Node, NodeLinked, RelationshipUpdate
from mellea_lrc.model.span import Span


class Citation(BaseModel):
    """An immutable citation whose field readings point to decision nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: str
    nodes: tuple[Node, ...]
    root_id: tuple[RelationshipUpdate[str | None], ...] = ()

    @property
    def site_span(self) -> Span:
        """The source site used to order citation occurrences."""
        raise NotImplementedError

    def record(self, stage: str) -> Self:
        """Record one decision; named field methods may share its node."""
        node = Node(id=f"{self.id}:node:{len(self.nodes)}", stage=stage)
        return type(self).model_validate({**self.model_dump(mode="python"), "nodes": (*self.nodes, node)})

    def _decision_node_id(self) -> str:
        if len(self.nodes) == 1:
            raise ValueError("Record a decision node before changing citation fields")
        return self.nodes[-1].id

    def _with_log(self, **logs: tuple[NodeLinked, ...]) -> Self:
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
            raise ValueError("Citation must have a creation node")
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
