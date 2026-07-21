"""Typed document and node types for post-extraction validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationRecord
    from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument


class ValidationNodeStatus(str, Enum):
    """Execution status of one validation operation."""

    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


class LocatorLookupOutcome(str, Enum):
    """Typed outcomes of the exact locator lookup node."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED_CITATION = "unsupported_citation"
    INCOMPLETE_LOCATOR = "incomplete_locator"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExactLocatorLookupNode:
    """One exact reporter-locator lookup against CourtListener.

    Only ``FOUND`` continues into the currently implemented branch. Other
    outcomes are explicit terminal records, not implicit fallback behavior.
    """

    node_id: str
    status: ValidationNodeStatus
    outcome: LocatorLookupOutcome
    locator: str | None
    record: CourtListenerCitationRecord | None = None
    candidate_count: int = 0
    error: str | None = None
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            msg = "Validation node_id must not be empty"
            raise ValueError(msg)
        if self.outcome is LocatorLookupOutcome.FOUND:
            if self.status is not ValidationNodeStatus.SUCCEEDED or self.record is None:
                msg = "A found locator node requires a succeeded status and one record"
                raise ValueError(msg)
            if self.candidate_count != 1:
                msg = "A found locator node requires candidate_count=1"
                raise ValueError(msg)
        elif self.record is not None:
            msg = "Only a found locator node may carry a record"
            raise ValueError(msg)


# This is intentionally a one-member alias for the first validation slice.
# Expand it into a union as operation-specific node types are introduced, e.g.
# ``ExactLocatorLookupNode | CaseIdentityNode | ...``.
ValidationNode: TypeAlias = ExactLocatorLookupNode


@dataclass(frozen=True, slots=True)
class CitationValidation:
    """Ordered validation-node progression for one extracted citation."""

    citation: ExtractedCitation
    nodes: tuple[ValidationNode, ...] = ()

    @property
    def citation_id(self) -> str:
        """Return the stable identifier from extraction."""
        return self.citation.citation_id

    def append(self, node: ValidationNode) -> CitationValidation:
        """Return a new citation validation with one node appended."""
        known_ids = {item.node_id for item in self.nodes}
        if node.node_id in known_ids:
            msg = f"Duplicate validation node_id: {node.node_id!r}"
            raise ValueError(msg)
        if any(dependency not in known_ids for dependency in node.depends_on):
            msg = f"Validation node {node.node_id!r} has an unknown dependency"
            raise ValueError(msg)
        return replace(self, nodes=(*self.nodes, node))


@dataclass(frozen=True, slots=True)
class ValidatedDocument:
    """Post-extraction validation state for every citation in one document."""

    source: ExtractedDocument
    citations: tuple[CitationValidation, ...]

    def __post_init__(self) -> None:
        source_ids = tuple(item.citation_id for item in self.source.citations)
        validation_ids = tuple(item.citation_id for item in self.citations)
        if validation_ids != source_ids:
            msg = "Citation validations must exactly match extracted citations in order"
            raise ValueError(msg)

    @property
    def text(self) -> str:
        """Return the immutable extracted-document text."""
        return self.source.text

    def citation_by_id(self, citation_id: str) -> CitationValidation:
        """Return one citation's validation progression."""
        for citation in self.citations:
            if citation.citation_id == citation_id:
                return citation
        msg = f"Unknown citation validation id: {citation_id!r}"
        raise KeyError(msg)
