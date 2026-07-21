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


class FieldCheckOutcome(str, Enum):
    """Deterministic comparison outcome for one citation field."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


class CaseNameCheckOutcome(str, Enum):
    """Outcomes of deterministic and Mellea-backed case-name checking."""

    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    NOT_SEMANTIC_MATCH = "not_semantic_match"
    UNASSESSABLE = "unassessable"
    FAILED = "failed"


class CaseNameReextractionOutcome(str, Enum):
    """Outcomes of extracting citation-bound parties from local context."""

    ACCEPTED = "accepted"
    EMPTY = "empty"
    FAILED = "failed"


class CaseSearchOutcome(str, Enum):
    """Outcomes of searching for a case after exact lookup fails."""

    NOT_IMPLEMENTED = "not_implemented"


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


@dataclass(frozen=True, slots=True)
class CaseNameCheckNode:
    """Case-name comparison after a found locator lookup."""

    node_id: str
    status: ValidationNodeStatus
    outcome: CaseNameCheckOutcome
    extracted_case_name: str | None
    retrieved_case_name: str | None
    depends_on: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CaseNameReextractionNode:
    """Grounded plaintiff/defendant evidence extracted from local context."""

    node_id: str
    status: ValidationNodeStatus
    outcome: CaseNameReextractionOutcome
    plaintiff: str | None
    defendant: str | None
    depends_on: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CaseSearchNode:
    """Placeholder for case search using re-extracted party evidence."""

    node_id: str
    status: ValidationNodeStatus
    outcome: CaseSearchOutcome
    depends_on: tuple[str, ...]
    candidate_count: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RecheckedCaseNameNode:
    """Terminal case-name decision using re-extracted parties."""

    node_id: str
    status: ValidationNodeStatus
    outcome: CaseNameCheckOutcome
    extracted_case_name: str | None
    retrieved_case_name: str | None
    depends_on: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class YearCheckNode:
    """Deterministic decision-year comparison after a found locator lookup."""

    node_id: str
    status: ValidationNodeStatus
    outcome: FieldCheckOutcome
    extracted_year: str | None
    retrieved_year: str | None
    depends_on: tuple[str, ...]


# Expand this union as operation-specific validation nodes are introduced.
ValidationNode: TypeAlias = (
    ExactLocatorLookupNode
    | CaseNameCheckNode
    | CaseNameReextractionNode
    | CaseSearchNode
    | RecheckedCaseNameNode
    | YearCheckNode
)


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
        if not node.node_id:
            msg = "Validation node_id must not be empty"
            raise ValueError(msg)
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
