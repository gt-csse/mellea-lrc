"""Assessment types for LLM-assisted citation checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, TYPE_CHECKING, TypeAlias

from mellea_lrc.assessment.grounding.proposal import is_in_context
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.validation.types import ValidatedDocument

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mellea_lrc.core.spans import Span


class CaseNameAssessmentStatus(str, Enum):
    """Canonical outcomes for case-name assessment.

    First pass (before re-extraction): ``exact_match``, ``semantic_match``, or
    route to re-extraction via internal ``needs_assessment``.

    After re-extraction: ``exact_match``, ``semantic_match``, ``different_case``,
    ``irregular_form``, or ``reextraction_fail``.
    """

    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    DIFFERENT_CASE = "different_case"
    IRREGULAR_FORM = "irregular_form"
    REEXTRACTION_FAIL = "reextraction_fail"
    NEEDS_ASSESSMENT = "needs_assessment"


class YearAssessmentStatus(str, Enum):
    """Canonical outcomes for deterministic year assessment."""

    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    MISSING = "missing"


class AssessmentStatus(str, Enum):
    """Execution state for one citation in the assessment stage."""

    WAITING = "waiting"
    SKIPPED = "skipped"
    ASSESSED = "assessed"
    FAILED = "failed"


class AssessmentSkipReason(str, Enum):
    """Reason a citation is intentionally excluded from assessment."""

    UNSUPPORTED_CITATION_KIND = "unsupported_citation_kind"
    VALIDATION_NOT_ELIGIBLE = "validation_not_eligible"


@dataclass(frozen=True, slots=True)
class AssessmentMetadata:
    """Execution provenance for the assessment stage."""

    mellea_calls: int = 0
    mellea_concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.mellea_calls < 0:
            msg = "AssessmentMetadata.mellea_calls must not be negative"
            raise ValueError(msg)
        if self.mellea_concurrency is not None and self.mellea_concurrency < 1:
            msg = "AssessmentMetadata.mellea_concurrency must be positive when provided"
            raise ValueError(msg)
        if (self.mellea_calls == 0) != (self.mellea_concurrency is None):
            msg = "AssessmentMetadata concurrency is required exactly when Mellea calls occur"
            raise ValueError(msg)
        if self.mellea_concurrency is not None and self.mellea_concurrency > self.mellea_calls:
            msg = "AssessmentMetadata concurrency must not exceed Mellea calls"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One typed conversation turn retained as assessment provenance."""

    role: str
    content: str
    extra_data: ExtraData = field(default_factory=ExtraData)


@dataclass(frozen=True, slots=True)
class CaseNameAssessment:
    """Assessment result for one extracted case name."""

    citation_id: str
    status: CaseNameAssessmentStatus
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    message: str
    chat_history: tuple[ChatTurn, ...] | None = None


@dataclass(frozen=True, slots=True)
class YearAssessment:
    """Assessment result for citation year."""

    citation_id: str
    status: YearAssessmentStatus
    extracted_year: str | None
    courtlistener_year: str | None
    message: str


@dataclass(frozen=True, slots=True)
class CitationAssessmentResult:
    """Completed substantive assessment for one validated citation."""

    citation_id: str
    case_assess: CaseNameAssessment
    year_assess: YearAssessment


@dataclass(frozen=True, slots=True)
class WaitingCitationAssessment:
    """Eligible citation whose assessment has not been attempted."""

    status: ClassVar[AssessmentStatus] = AssessmentStatus.WAITING
    citation_id: str


@dataclass(frozen=True, slots=True)
class SkippedCitationAssessment:
    """Citation intentionally excluded from assessment."""

    status: ClassVar[AssessmentStatus] = AssessmentStatus.SKIPPED
    citation_id: str
    reason: AssessmentSkipReason
    message: str


@dataclass(frozen=True, slots=True)
class AssessedCitationAssessment:
    """Citation whose assessment completed successfully."""

    status: ClassVar[AssessmentStatus] = AssessmentStatus.ASSESSED
    citation_id: str
    result: CitationAssessmentResult


@dataclass(frozen=True, slots=True)
class FailedCitationAssessment:
    """Citation whose assessment was attempted but failed."""

    status: ClassVar[AssessmentStatus] = AssessmentStatus.FAILED
    citation_id: str
    error: str


CitationAssessment: TypeAlias = (
    WaitingCitationAssessment
    | SkippedCitationAssessment
    | AssessedCitationAssessment
    | FailedCitationAssessment
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessedDocument(ValidatedDocument):
    """A validated document with additive citation assessment history."""

    assessments: tuple[CitationAssessment, ...]
    assessment_metadata: AssessmentMetadata
    modified_citations: tuple[ModifiedExtractedCitation, ...] = ()
    reassessments: tuple[CitationAssessmentResult, ...] = ()

    @property
    def assessment_complete(self) -> bool:
        """Return whether no eligible citation remains unattempted."""
        return not any(isinstance(item, WaitingCitationAssessment) for item in self.assessments)

    def __post_init__(self) -> None:
        ValidatedDocument.__post_init__(self)
        citation_ids = {item.citation_id for item in self.citations}
        assessment_ids = _validated_assessment_record_ids(self.assessments)
        modified_ids = _unique_document_ids(
            (item.citation_id for item in self.modified_citations),
            "modified citation",
        )
        reassessment_ids = _validated_assessment_result_ids(self.reassessments, "reassessment")

        if assessment_ids != citation_ids:
            msg = "Assessment identifiers must exactly match extracted citation identifiers"
            raise ValueError(msg)
        if not modified_ids <= citation_ids:
            msg = "Modified citation identifiers must refer to extracted citations"
            raise ValueError(msg)
        if not reassessment_ids <= modified_ids:
            msg = "Reassessment identifiers must refer to modified citations"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ModifiedExtractedCitationProposal:
    """LLM-proposed corrected case name copied from local context."""

    case_name: str | None = None

    @property
    def extracted_case_name(self) -> str | None:
        """Alias used by reassessment and serialization."""
        return self.case_name

    def valid(self, document_context: str) -> bool:
        """Require the proposed case name to be present in the original context."""
        return bool(self.case_name) and is_in_context(self.case_name, document_context)


@dataclass(frozen=True, slots=True)
class ModifiedExtractedCitation:
    """A grounded modified extraction bound to document-local citation identity."""

    citation_id: str = ""
    span: Span | None = None
    matched_text: str | None = None
    case_name: str | None = None

    @property
    def extracted_case_name(self) -> str | None:
        """Alias used by reassessment and serialization."""
        return self.case_name

    @classmethod
    def from_proposal(
        cls,
        proposal: ModifiedExtractedCitationProposal,
        *,
        citation_id: str,
        span: Span | None,
    ) -> ModifiedExtractedCitation:
        """Bind an LLM-proposed extraction to document-local citation identity."""
        return cls(
            citation_id=citation_id,
            span=span,
            matched_text=proposal.case_name,
            case_name=proposal.case_name,
        )


@dataclass(frozen=True, slots=True)
class CaseNameAssessmentRun:
    """Case-name assessment plus any modified extraction history it produced."""

    assessment: CaseNameAssessment
    modified_citation: ModifiedExtractedCitationProposal | None = None
    reassessment: CaseNameAssessment | None = None


def _validated_assessment_record_ids(
    assessments: tuple[CitationAssessment, ...],
) -> set[str]:
    ids = _unique_document_ids((item.citation_id for item in assessments), "assessment")
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            _validate_assessment_result(item.result, item.citation_id, "assessment")
    return ids


def _validated_assessment_result_ids(
    assessments: tuple[CitationAssessmentResult, ...],
    label: str,
) -> set[str]:
    ids = _unique_document_ids((item.citation_id for item in assessments), label)
    for item in assessments:
        _validate_assessment_result(item, item.citation_id, label)
    return ids


def _validate_assessment_result(
    result: CitationAssessmentResult,
    citation_id: str,
    label: str,
) -> None:
    if (
        result.citation_id != citation_id
        or result.case_assess.citation_id != citation_id
        or result.year_assess.citation_id != citation_id
    ):
        msg = f"Nested {label} identifiers must match their parent citation identifier"
        raise ValueError(msg)


def _unique_document_ids(values: Iterable[str], label: str) -> set[str]:
    ids = list(values)
    if any(not item_id for item_id in ids):
        msg = f"{label.title()} identifiers must not be empty"
        raise ValueError(msg)
    if len(ids) != len(set(ids)):
        msg = f"{label.title()} identifiers must be unique within a document"
        raise ValueError(msg)
    return set(ids)
