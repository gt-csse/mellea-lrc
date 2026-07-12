"""Citation-level execution model.

This module defines the citation-node substrate used by newer workflows. It is
intentionally independent from the document-stage dataclasses: extraction stays
the source artifact, while citation nodes hold execution state and trace steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea_lrc.core.citations import CanonicalCitation
    from mellea_lrc.core.spans import Span

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...] | MappingProxyType[str, "JSONValue"]
JSONMapping: TypeAlias = MappingProxyType[str, JSONValue]


class CitationNodeStatus(str, Enum):
    """Coarse execution status for one citation node."""

    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETE = "complete"


class CitationStepStatus(str, Enum):
    """Outcome status for a single citation-node operation."""

    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class CitationNodeInput:
    """Stable citation input copied from extraction for independent execution."""

    citation_id: str
    citation_span: Span
    matched_locator_text: str
    matched_citation_text: str
    citation: CanonicalCitation
    asserted_decision_date: str | None = None
    resolves_to: str | None = None

    def __post_init__(self) -> None:
        if not self.citation_id:
            msg = "Citation node input must have a citation_id"
            raise ValueError(msg)
        if not self.matched_locator_text:
            msg = f"Citation node input {self.citation_id!r} must have matched_locator_text"
            raise ValueError(msg)
        if not self.matched_citation_text:
            msg = f"Citation node input {self.citation_id!r} must have matched_citation_text"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CitationStep:
    """One auditable transition or observation for a citation node."""

    operation: str
    status: CitationStepStatus
    summary: str
    step_id: str | None = None
    depends_on: tuple[str, ...] = ()
    lane: str | None = None
    data: JSONMapping = field(default_factory=lambda: MappingProxyType({}))
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.operation:
            msg = "Citation step operation must not be empty"
            raise ValueError(msg)
        if not self.summary:
            msg = f"Citation step {self.operation!r} must have a summary"
            raise ValueError(msg)
        if self.step_id == "":
            msg = f"Citation step {self.operation!r} step_id must not be empty"
            raise ValueError(msg)
        dependencies = tuple(self.depends_on)
        if any(not dependency for dependency in dependencies):
            msg = f"Citation step {self.operation!r} dependencies must not be empty"
            raise ValueError(msg)
        if self.lane == "":
            msg = f"Citation step {self.operation!r} lane must not be empty"
            raise ValueError(msg)
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(self, "data", freeze_json_mapping(self.data))


@dataclass(frozen=True, slots=True)
class CitationNode:
    """Execution state and trace for one citation."""

    input: CitationNodeInput
    status: CitationNodeStatus = CitationNodeStatus.READY
    steps: tuple[CitationStep, ...] = ()

    @property
    def citation_id(self) -> str:
        """Return the document-local citation identifier."""
        return self.input.citation_id

    def append_step(
        self,
        step: CitationStep,
        *,
        status: CitationNodeStatus | None = None,
    ) -> CitationNode:
        """Return a new node with ``step`` appended and optional new status."""
        return replace(self, status=status or _status_after_step(step), steps=(*self.steps, step))


@dataclass(frozen=True, slots=True)
class CitationNodeDocument:
    """Document-level projection whose execution truth lives in citation nodes."""

    text: str
    nodes: tuple[CitationNode, ...]

    def __post_init__(self) -> None:
        ids = [node.citation_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            msg = "Citation node identifiers must be unique within a document"
            raise ValueError(msg)
        for node in self.nodes:
            if node.input.citation_span.end > len(self.text):
                msg = f"Citation node {node.citation_id!r} citation_span exceeds document text"
                raise ValueError(msg)

    def node_by_id(self, citation_id: str) -> CitationNode:
        """Return one node by document-local citation id."""
        for node in self.nodes:
            if node.citation_id == citation_id:
                return node
        msg = f"Unknown citation node id: {citation_id!r}"
        raise KeyError(msg)


def freeze_json_mapping(value: Mapping[object, object]) -> JSONMapping:
    """Deep-freeze a JSON-like mapping for immutable trace data."""
    return MappingProxyType({str(key): freeze_json_value(item) for key, item in value.items()})


def freeze_json_value(value: object) -> JSONValue:
    """Deep-freeze JSON-like trace data."""
    if isinstance(value, MappingProxyType):
        return freeze_json_mapping(value)
    if isinstance(value, dict):
        return freeze_json_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(freeze_json_value(item) for item in value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    msg = f"Unsupported citation step data value: {type(value).__name__}"
    raise TypeError(msg)


def _status_after_step(step: CitationStep) -> CitationNodeStatus:
    if step.status is CitationStepStatus.FAILED:
        return CitationNodeStatus.FAILED
    if step.status is CitationStepStatus.BLOCKED:
        return CitationNodeStatus.BLOCKED
    return CitationNodeStatus.READY
