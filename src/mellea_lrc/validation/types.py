"""Typed document and node types for post-extraction validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping

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


class MelleaCaseNameCheckOutcome(str, Enum):
    """Outcomes of semantic comparison after an exact case-name mismatch."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class MelleaCaseNameReextractionOutcome(str, Enum):
    """Results of Mellea re-extracting locally grounded case parties."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class MelleaCaseNameQueryPreparationOutcome(str, Enum):
    """Results of preparing a CourtListener query from re-extracted parties."""

    PREPARED = "prepared"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class OpinionSearchOutcome(str, Enum):
    """Results of searching CourtListener's opinion corpus."""

    SEARCHED = "searched"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class RecapSearchOutcome(str, Enum):
    """Results of searching CourtListener's RECAP corpus."""

    SEARCHED = "searched"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class DocketCourtRetrievalOutcome(str, Enum):
    """Results of retrieving a CourtListener docket's court identifier."""

    FOUND = "found"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class CandidateSelectionOutcome(str, Enum):
    """Results of applying the bounded candidate-validation guard."""

    ALL_SELECTED = "all_selected"
    DEFERRED_OVER_LIMIT = "deferred_over_limit"


class CandidateEvaluationOutcome(str, Enum):
    """Readiness of one independently evaluable retrieved candidate."""

    READY = "ready"


class CandidateEvaluationSource(str, Enum):
    """Retrieval route that produced a candidate evaluation node."""

    LOCATOR_LOOKUP = "locator_lookup"
    OPINION_SEARCH = "opinion_search"
    RECAP_SEARCH = "recap_search"


MIN_AMBIGUOUS_CANDIDATE_COUNT = 2


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
    candidates: tuple[CourtListenerCitationRecord, ...] = ()
    candidate_count: int = 0
    status_message: str | None = None
    outcome_message: str | None = None
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
        elif self.outcome is LocatorLookupOutcome.AMBIGUOUS:
            if (
                self.status is not ValidationNodeStatus.SUCCEEDED
                or self.record is not None
                or self.candidate_count < MIN_AMBIGUOUS_CANDIDATE_COUNT
                or len(self.candidates) != self.candidate_count
            ):
                msg = "An ambiguous locator node requires its complete candidate records"
                raise ValueError(msg)
        elif self.record is not None or self.candidates:
            msg = "Only a found or ambiguous locator node may carry candidate records"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ExactCaseNameCheckNode:
    """Exact normalized case-name comparison after a found locator lookup."""

    node_id: str
    status: ValidationNodeStatus
    outcome: FieldCheckOutcome
    extracted_case_name: str | None
    retrieved_case_name: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


@dataclass(frozen=True, slots=True)
class MelleaCaseNameCheckNode:
    """Mellea semantic comparison of otherwise unmatched case names."""

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaCaseNameCheckOutcome
    extracted_case_name: str
    retrieved_case_name: str
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MelleaCaseNameReextractionNode:
    """Plaintiff and defendant re-extracted from citation-local text by Mellea."""

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaCaseNameReextractionOutcome
    plaintiff: str | None
    defendant: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MelleaCaseNameQueryPreparationNode:
    """Mellea-prepared terms and deterministic query for candidate retrieval."""

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaCaseNameQueryPreparationOutcome
    query: str | None
    query_plaintiff: str | None
    query_defendant: str | None
    court_id: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class OpinionSearchNode:
    """One CourtListener opinion-corpus search from prepared case-name terms."""

    node_id: str
    status: ValidationNodeStatus
    outcome: OpinionSearchOutcome
    query: str | None
    result_count: int | None
    results: tuple[Mapping[str, object], ...]
    next_cursor: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RecapSearchNode:
    """One CourtListener RECAP-corpus search from prepared case-name terms."""

    node_id: str
    status: ValidationNodeStatus
    outcome: RecapSearchOutcome
    query: str | None
    result_count: int | None
    results: tuple[Mapping[str, object], ...]
    next_cursor: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateSelectionNode:
    """Bounded decision on whether one retrieval result set may be evaluated."""

    node_id: str
    status: ValidationNodeStatus
    outcome: CandidateSelectionOutcome
    total_candidate_count: int
    selected_candidate_count: int
    selection_limit: int
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateEvaluationNode:
    """One selected retrieved candidate made ready for field-check subtrees."""

    node_id: str
    status: ValidationNodeStatus
    outcome: CandidateEvaluationOutcome
    source: CandidateEvaluationSource
    candidate_index: int
    cluster_id: str | None
    case_name: str | None
    date_filed: str | None
    court_id: str | None
    docket_id: str | None
    record: CourtListenerCitationRecord | Mapping[str, object]
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None

    @property
    def year(self) -> str | None:
        """Return the filed-year prefix when the opinion result provides one."""
        return self.date_filed[:4] if self.date_filed else None


@dataclass(frozen=True, slots=True)
class MelleaReextractedCaseNameCheckNode:
    """Semantic comparison using re-extracted plaintiff and defendant evidence."""

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaCaseNameCheckOutcome
    reextracted_case_name: str | None
    retrieved_case_name: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DocketCourtRetrievalNode:
    """Court identifier retrieved from the docket linked to a found citation."""

    node_id: str
    status: ValidationNodeStatus
    outcome: DocketCourtRetrievalOutcome
    docket_id: str | None
    court_id: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CourtCheckNode:
    """Exact comparison of Eyecite and CourtListener court identifiers."""

    node_id: str
    status: ValidationNodeStatus
    outcome: FieldCheckOutcome
    extracted_court_id: str | None
    retrieved_court_id: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


@dataclass(frozen=True, slots=True)
class YearCheckNode:
    """Deterministic decision-year comparison after a found locator lookup."""

    node_id: str
    status: ValidationNodeStatus
    outcome: FieldCheckOutcome
    extracted_year: str | None
    retrieved_year: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


# Expand this union as operation-specific validation nodes are introduced.
ValidationNode: TypeAlias = (
    ExactLocatorLookupNode
    | ExactCaseNameCheckNode
    | MelleaCaseNameCheckNode
    | MelleaCaseNameReextractionNode
    | MelleaCaseNameQueryPreparationNode
    | OpinionSearchNode
    | RecapSearchNode
    | CandidateSelectionNode
    | CandidateEvaluationNode
    | MelleaReextractedCaseNameCheckNode
    | DocketCourtRetrievalNode
    | CourtCheckNode
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
