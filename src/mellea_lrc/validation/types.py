"""Typed document and node types for post-extraction validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea_lrc.core.spans import Span
    from mellea_lrc.courtlistener.opinion_models import CourtListenerOpinionCluster
    from mellea_lrc.extraction.types import CitationRecord, Document
    from mellea_lrc.llm import IvrRun


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


class DocketRootSearchOutcome(str, Enum):
    """Results of retrieving CourtListener docket candidates for one docket root."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    EXCEEDS_REVIEW_LIMIT = "exceeds_review_limit"
    FAILED = "failed"


class MelleaDocketNumberReviewOutcome(str, Enum):
    """Results of one grounded model review of a docket locator's number."""

    UNCHANGED = "unchanged"
    CORRECTED = "corrected"
    NO_DOCKET_NUMBER = "no_docket_number"
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


class ReporterPageRetrievalOutcome(str, Enum):
    """Results of retrieving a reporter page from citation-aware opinion HTML."""

    FOUND = "found"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class MelleaPinpointCheckOutcome(str, Enum):
    """Semantic support findings from one retrieved reporter page."""

    SUPPORTS = "supports"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class MelleaCitingPropositionExtractionOutcome(str, Enum):
    """Results of identifying the proposition attributed to one citation."""

    IDENTIFIED = "identified"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class EvidenceQuoteMatchMethod(str, Enum):
    """How a model-proposed quote was grounded in retrieved page text."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"


class CandidateSelectionOutcome(str, Enum):
    """Fact recorded by the bounded candidate-review guard."""

    ALL_SELECTED = "all_selected"
    EXCEEDS_REVIEW_LIMIT = "exceeds_review_limit"


class CandidateEvaluationOutcome(str, Enum):
    """Readiness of one independently evaluable retrieved candidate."""

    READY = "ready"


class CandidateEvaluationSource(str, Enum):
    """Retrieval route that produced a candidate evaluation node."""

    LOCATOR_LOOKUP = "locator_lookup"
    DOCKET_SEARCH = "docket_search"
    OPINION_SEARCH = "opinion_search"
    RECAP_SEARCH = "recap_search"


class AggregatedFieldOutcome(str, Enum):
    """Field-level outcome projected into a candidate assessment."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class LocatorCandidateAssessmentOutcome(str, Enum):
    """Overall conclusion for one uniquely located opinion candidate."""

    MATCH = "match"
    MISMATCH = "mismatch"
    PARTIAL_MATCH = "partial_match"


class LocatorCitationSummaryOutcome(str, Enum):
    """Completion state of the exact-locator candidate evidence summary."""

    COMPLETE = "complete"


class MelleaLocatorCandidateChoiceOutcome(str, Enum):
    """Result of a grounded model choice among reviewed locator candidates."""

    SELECTED = "selected"
    NO_MATCH = "no_match"
    FAILED = "failed"


class LocatorIdentityResolutionOutcome(str, Enum):
    """Whether the reporter-locator checkpoint selected one candidate.

    Deferred outcomes name the next admissible route. They are neither a
    negative identity decision nor an execution failure.
    """

    RESOLVED = "resolved"
    NO_MATCH = "no_match"
    DEFERRED_TO_SEARCH = "deferred_to_search"
    DEFERRED_TO_FUTURE_IMPLEMENTATION = "deferred_to_future_implementation"


class SearchCandidateAssessmentOutcome(str, Enum):
    """Limited conclusion for one candidate returned by a search."""

    POSSIBLE_MATCH = "possible_match"
    MISMATCH = "mismatch"


CandidateAssessmentOutcome: TypeAlias = LocatorCandidateAssessmentOutcome | SearchCandidateAssessmentOutcome


class CandidateProvenance(str, Enum):
    """CourtListener corpus that produced a summarized candidate."""

    OPINION = "opinion"
    DOCKET = "docket"
    RECAP = "recap"


class CitationSummaryAssessmentOutcome(str, Enum):
    """Strongest candidate conclusion exposed at citation scope."""

    MATCH = "match"
    POSSIBLE_MATCH = "possible_match"
    MISMATCH = "mismatch"
    NOT_FOUND = "not_found"


class SearchCitationSummaryOutcome(str, Enum):
    """Completion state of a search-derived citation summary."""

    COMPLETE = "complete"


MIN_AMBIGUOUS_CANDIDATE_COUNT = 2


@dataclass(frozen=True, slots=True)
class ExactLocatorLookupNode:
    """One exact reporter-locator lookup against CourtListener.

    Only ``FOUND`` continues into the currently implemented branch. Other
    outcomes are explicit terminal nodes, not implicit fallback behavior.
    """

    node_id: str
    status: ValidationNodeStatus
    outcome: LocatorLookupOutcome
    locator: str | None
    cluster: CourtListenerOpinionCluster | None = None
    candidate_clusters: tuple[CourtListenerOpinionCluster, ...] = ()
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
            if self.status is not ValidationNodeStatus.SUCCEEDED or self.cluster is None:
                msg = "A found locator node requires a succeeded status and one cluster"
                raise ValueError(msg)
            if self.candidate_count != 1:
                msg = "A found locator node requires candidate_count=1"
                raise ValueError(msg)
        elif self.outcome is LocatorLookupOutcome.AMBIGUOUS:
            if (
                self.status is not ValidationNodeStatus.SUCCEEDED
                or self.cluster is not None
                or self.candidate_count < MIN_AMBIGUOUS_CANDIDATE_COUNT
                or len(self.candidate_clusters) != self.candidate_count
            ):
                msg = "An ambiguous locator node requires its complete candidate clusters"
                raise ValueError(msg)
        elif self.cluster is not None or self.candidate_clusters:
            msg = "Only a found or ambiguous locator node may carry candidate clusters"
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
    run: IvrRun | None = None


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
    run: IvrRun | None = None


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
    run: IvrRun | None = None


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
class DocketRootSearchNode:
    """All CourtListener docket-search evidence for one docket root.

    Search is the retrieval mechanism for docket identifiers.  The stated
    court, when present, narrows the query; it is never required.  Candidate
    records remain intact so the later unique and ambiguity stages can resume
    without repeating the network call.
    """

    node_id: str
    status: ValidationNodeStatus
    outcome: DocketRootSearchOutcome
    docket_number: str | None
    query: str | None
    candidate_count: int
    candidates: tuple[Mapping[str, object], ...]
    next_cursor: str | None
    depends_on: tuple[str, ...] = ()
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
    docket_number: str | None
    record: CourtListenerOpinionCluster | Mapping[str, object]
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
    run: IvrRun | None = None


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
class DocketNumberCheckNode:
    """Comparison of the docket number stated in the filing and search record."""

    node_id: str
    status: ValidationNodeStatus
    outcome: FieldCheckOutcome
    extracted_docket_number: str | None
    retrieved_docket_number: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


@dataclass(frozen=True, slots=True)
class MelleaDocketNumberReviewNode:
    """One source-grounded model opinion on a docket locator's identifier.

    The model can only return a number found inside the locator the filing
    wrote.  A corrected value is therefore a correction of our earlier parse,
    never a newly invented docket.  The node keeps the full IVR repair record;
    its terminal outcome lets later stages tell a completed review from an
    unasked one without traversing the trace.
    """

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaDocketNumberReviewOutcome
    source_locator: str
    extracted_docket_number: str | None
    proposed_docket_number: str | None
    grounded_docket_number: str | None
    reason: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None
    run: IvrRun | None = None


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
class ReporterPageEvidence:
    """One reporter page recovered from a CourtListener sub-opinion."""

    opinion_id: str
    opinion_type: str
    text: str


@dataclass(frozen=True, slots=True)
class ReporterPageRetrievalNode:
    """Serialization-ready reporter-page evidence for one opinion candidate."""

    node_id: str
    status: ValidationNodeStatus
    outcome: ReporterPageRetrievalOutcome
    cluster_id: str | None
    reporter_citation: str | None
    pin_cite: str | None
    citation_index: int | None
    evidence: ReporterPageEvidence | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MelleaCitingPropositionExtractionNode:
    """Grounded citing proposition and its offsets in the source document."""

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaCitingPropositionExtractionOutcome
    context_span: Span
    reasoning: str | None
    proposition: str | None
    proposition_span: Span | None
    proposition_match_method: EvidenceQuoteMatchMethod | None
    proposition_match_score: float | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MelleaPinpointCheckNode:
    """Semantic judgment with evidence offsets into its retrieval dependency's page text."""

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaPinpointCheckOutcome
    reasoning: str | None
    evidence_quote: str | None
    evidence_span: Span | None
    evidence_match_method: EvidenceQuoteMatchMethod | None
    evidence_match_score: float | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LocatorCandidateAssessmentNode:
    """Table-ready conclusion for one candidate from a complete-locator route."""

    node_id: str
    status: ValidationNodeStatus
    outcome: LocatorCandidateAssessmentOutcome
    candidate_index: int
    extracted_citation: str | None
    extracted_case_name: str | None
    retrieved_case_name: str | None
    case_name_outcome: AggregatedFieldOutcome
    case_name_evidence: str
    extracted_year: str | None
    retrieved_year: str | None
    year_outcome: AggregatedFieldOutcome
    extracted_court_id: str | None
    retrieved_court_id: str | None
    court_outcome: AggregatedFieldOutcome
    docket_id: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


@dataclass(frozen=True, slots=True)
class CitationSummaryPinpoint:
    """Frontend-ready projection of one candidate's pinpoint comparison."""

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaPinpointCheckOutcome
    reporter_citation: str | None
    pin_cite: str | None
    opinion_id: str | None
    opinion_type: str | None
    reporter_page_text: str | None
    citing_context_span: Span
    citation_span: Span
    proposition: str | None
    proposition_span: Span | None
    reasoning: str | None
    evidence_quote: str | None
    evidence_span: Span | None
    evidence_match_method: EvidenceQuoteMatchMethod | None
    evidence_match_score: float | None
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CitationSummaryCandidate:
    """Frontend-ready, provenance-tagged candidate exposed by a citation summary."""

    provenance: CandidateProvenance
    candidate_index: int
    assessment_node_id: str
    outcome: CandidateAssessmentOutcome
    extracted_citation: str | None
    extracted_case_name: str | None
    retrieved_case_name: str | None
    case_name_outcome: AggregatedFieldOutcome
    case_name_evidence: str
    extracted_year: str | None
    retrieved_year: str | None
    year_outcome: AggregatedFieldOutcome
    extracted_court_id: str | None
    retrieved_court_id: str | None
    court_outcome: AggregatedFieldOutcome
    docket_id: str | None
    docket_number: str | None = None
    opinion_url: str | None = None
    docket_url: str | None = None
    pinpoint: CitationSummaryPinpoint | None = None


@dataclass(frozen=True, slots=True)
class LocatorCitationSummaryNode:
    """List of every fully evaluated candidate from one complete-locator route."""

    node_id: str
    status: ValidationNodeStatus
    outcome: LocatorCitationSummaryOutcome
    overall_outcome: CitationSummaryAssessmentOutcome | None
    pinpoint_requires_review: bool | None
    candidates: tuple[CitationSummaryCandidate, ...]
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


@dataclass(frozen=True, slots=True)
class MelleaLocatorCandidateChoiceNode:
    """Grounded model decision for a complete, bounded locator candidate list.

    The model receives every reviewed candidate and target-only local context.
    It reparses the stated fields while selecting one representative candidate
    or returning no match.  Reading a candidate opinion could provide finer
    tie-breaking, but that couples identity to the later opinion stage; keep
    that as a TODO rather than silently adding it to this checkpoint.
    """

    node_id: str
    status: ValidationNodeStatus
    outcome: MelleaLocatorCandidateChoiceOutcome
    candidate_indices: tuple[int, ...]
    selected_candidate_index: int | None
    reparsed_case_name: str | None
    reparsed_court: str | None
    reparsed_date: str | None
    rationale: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None
    error: str | None = None
    run: IvrRun | None = None

    def __post_init__(self) -> None:
        if not self.candidate_indices:
            msg = "A model candidate choice requires at least one reviewed candidate"
            raise ValueError(msg)
        if self.outcome is MelleaLocatorCandidateChoiceOutcome.SELECTED:
            if self.selected_candidate_index not in self.candidate_indices:
                msg = "A selected model candidate must be one of the reviewed candidates"
                raise ValueError(msg)
        elif self.selected_candidate_index is not None:
            msg = "Only a selected model choice may identify a candidate"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LocatorIdentityResolutionNode:
    """Explicit terminal identity decision for this locator checkpoint.

    The candidate summary remains the complete evidence record.  This node
    points to the selected assessment and to the deterministic summary or model
    choice that made the decision, so callers do not need graph traversal to
    reconstruct the identity result.
    """

    node_id: str
    status: ValidationNodeStatus
    outcome: LocatorIdentityResolutionOutcome
    selected_candidate_index: int | None
    selected_assessment_node_id: str | None
    matching_candidate_indices: tuple[int, ...]
    selection_evidence_node_id: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None

    def __post_init__(self) -> None:
        if self.outcome is LocatorIdentityResolutionOutcome.RESOLVED:
            if self.selected_candidate_index is None or self.selected_assessment_node_id is None:
                msg = "A resolved locator identity requires one selected candidate and assessment"
                raise ValueError(msg)
            if self.selection_evidence_node_id is None:
                msg = "A resolved locator identity requires selection evidence"
                raise ValueError(msg)
        elif self.selected_candidate_index is not None or self.selected_assessment_node_id is not None:
            msg = "Only a resolved locator identity may select a candidate"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class OpinionSearchCandidateAssessmentNode:
    """Serialization-ready conclusion for one opinion-search candidate."""

    node_id: str
    status: ValidationNodeStatus
    outcome: SearchCandidateAssessmentOutcome
    candidate_index: int
    extracted_citation: str | None
    extracted_case_name: str | None
    retrieved_case_name: str | None
    case_name_outcome: AggregatedFieldOutcome
    case_name_evidence: str
    extracted_year: str | None
    retrieved_year: str | None
    year_outcome: AggregatedFieldOutcome
    extracted_court_id: str | None
    retrieved_court_id: str | None
    court_outcome: AggregatedFieldOutcome
    docket_id: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


@dataclass(frozen=True, slots=True)
class RecapSearchCandidateAssessmentNode:
    """Serialization-ready conclusion for one RECAP-search candidate."""

    node_id: str
    status: ValidationNodeStatus
    outcome: SearchCandidateAssessmentOutcome
    candidate_index: int
    extracted_citation: str | None
    extracted_case_name: str | None
    retrieved_case_name: str | None
    case_name_outcome: AggregatedFieldOutcome
    case_name_evidence: str
    extracted_year: str | None
    retrieved_year: str | None
    year_outcome: AggregatedFieldOutcome
    extracted_court_id: str | None
    retrieved_court_id: str | None
    court_outcome: AggregatedFieldOutcome
    docket_id: str | None
    depends_on: tuple[str, ...]
    status_message: str | None = None
    outcome_message: str | None = None


@dataclass(frozen=True, slots=True)
class SearchCitationSummaryNode:
    """Terminal list of assessed candidates from both search corpora."""

    node_id: str
    status: ValidationNodeStatus
    outcome: SearchCitationSummaryOutcome
    overall_outcome: CitationSummaryAssessmentOutcome | None
    candidates: tuple[CitationSummaryCandidate, ...]
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
    | DocketRootSearchNode
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
    | DocketNumberCheckNode
    | MelleaDocketNumberReviewNode
    | ReporterPageRetrievalNode
    | MelleaCitingPropositionExtractionNode
    | MelleaPinpointCheckNode
    | CourtCheckNode
    | LocatorCandidateAssessmentNode
    | LocatorCitationSummaryNode
    | MelleaLocatorCandidateChoiceNode
    | LocatorIdentityResolutionNode
    | OpinionSearchCandidateAssessmentNode
    | RecapSearchCandidateAssessmentNode
    | SearchCitationSummaryNode
    | YearCheckNode
)


@dataclass(frozen=True, slots=True)
class CitationValidation:
    """Ordered validation-node progression for one extracted citation."""

    citation: CitationRecord
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

    @property
    def aggregation(self) -> LocatorCitationSummaryNode | SearchCitationSummaryNode | None:
        """Return the route's candidate-evidence summary when one was produced."""
        summaries = tuple(
            node
            for node in self.nodes
            if isinstance(node, (LocatorCitationSummaryNode, SearchCitationSummaryNode))
        )
        return summaries[0] if len(summaries) == 1 else None

    @property
    def identity_resolution(self) -> LocatorIdentityResolutionNode | None:
        """Return the final exact-locator identity decision, when this route made one."""
        resolutions = tuple(node for node in self.nodes if isinstance(node, LocatorIdentityResolutionNode))
        if len(resolutions) > 1:
            msg = f"Citation {self.citation_id!r} has multiple locator identity decisions"
            raise ValueError(msg)
        return resolutions[0] if resolutions else None


@dataclass(frozen=True, slots=True)
class ValidatedDocument:
    """Validation state for active citations, with every raw record in source."""

    source: Document
    citations: tuple[CitationValidation, ...]

    def __post_init__(self) -> None:
        source_ids = tuple(item.citation_id for item in self.source.active_citations)
        validation_ids = tuple(item.citation_id for item in self.citations)
        if validation_ids != source_ids:
            msg = "Citation validations must exactly match active extracted citations in order"
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
