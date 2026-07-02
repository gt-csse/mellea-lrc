"""Document-level assessment records and artifact types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, TypeAlias

from mellea_lrc.assessment.types.case_name import (
    CaseNameReassessed,
    CaseNameReassessmentFailed,
)
from mellea_lrc.validation.types import ValidatedDocument

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mellea_lrc.assessment.types.citation import CitationAssessmentResult
    from mellea_lrc.courtlistener.types import CitationMatch


class AssessmentStatus(str, Enum):
    """Execution state for one citation in the assessment stage."""

    WAITING = "waiting"
    SKIPPED = "skipped"
    ASSESSED = "assessed"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


class AssessmentSkipReason(str, Enum):
    """Reason a citation is intentionally excluded from assessment."""

    UNSUPPORTED_CITATION_KIND = "unsupported_citation_kind"
    VALIDATION_NOT_ELIGIBLE = "validation_not_eligible"


@dataclass(frozen=True, slots=True)
class AssessmentMetadata:
    """Execution provenance for the assessment stage."""

    mellea_concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.mellea_concurrency is not None and self.mellea_concurrency < 1:
            msg = "AssessmentMetadata.mellea_concurrency must be positive when provided"
            raise ValueError(msg)


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
    """Citation whose field assessments completed successfully."""

    status: ClassVar[AssessmentStatus] = AssessmentStatus.ASSESSED
    citation_id: str
    result: CitationAssessmentResult


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """One ambiguous candidate, assessed by the same logic as the found branch."""

    match: CitationMatch
    result: CitationAssessmentResult


@dataclass(frozen=True, slots=True)
class AmbiguousCitationAssessment:
    """An ambiguous citation whose found-branch assessment ran per candidate.

    Validation only pulls the candidates back; deciding which one the citation
    refers to is an opinion, so we delegate the normal found-branch assessment to
    each candidate and surface them side by side. Drawing a single conclusion
    across candidates is deferred. When more than ``_MAX`` candidates are
    returned we fail fast (``gated=True``, no per-candidate runs) rather than
    fan out an unbounded number of assessments.
    """

    status: ClassVar[AssessmentStatus] = AssessmentStatus.AMBIGUOUS
    citation_id: str
    candidates: tuple[CandidateAssessment, ...]
    gated: bool = False
    message: str = ""


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
    | AmbiguousCitationAssessment
    | FailedCitationAssessment
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessedDocument(ValidatedDocument):
    """A validated document with one assessment record per citation."""

    assessments: tuple[CitationAssessment, ...]
    assessment_metadata: AssessmentMetadata

    @property
    def assessment_complete(self) -> bool:
        """Return whether no citation assessment remains waiting."""
        return not any(isinstance(item, WaitingCitationAssessment) for item in self.assessments)

    def __post_init__(self) -> None:
        ValidatedDocument.__post_init__(self)
        citation_ids = tuple(item.citation_id for item in self.citations)
        assessment_ids = _unique_document_ids(
            (item.citation_id for item in self.assessments),
            "assessment",
        )
        if assessment_ids != citation_ids:
            msg = "Assessment identifiers must exactly match extracted citation identifiers in order"
            raise ValueError(msg)
        for item in self.assessments:
            if isinstance(item, AssessedCitationAssessment):
                self._validate_reextracted_span(item.citation_id, item.result)
            elif isinstance(item, AmbiguousCitationAssessment):
                for candidate in item.candidates:
                    self._validate_reextracted_span(item.citation_id, candidate.result)

    def _validate_reextracted_span(
        self,
        citation_id: str,
        result: CitationAssessmentResult,
    ) -> None:
        followup = result.case_name.followup
        if not isinstance(followup, (CaseNameReassessed, CaseNameReassessmentFailed)):
            return
        reextracted = followup.reextracted_case_name
        span = reextracted.case_name_span
        if span.end > len(self.text):
            msg = f"Re-extracted case-name span for {citation_id!r} exceeds document text"
            raise ValueError(msg)
        source_text = self.text[span.start : span.end]
        if " ".join(source_text.split()) != " ".join(reextracted.case_name.split()):
            msg = f"Re-extracted case name for {citation_id!r} does not match its document span"
            raise ValueError(msg)


def _unique_document_ids(values: Iterable[str], label: str) -> tuple[str, ...]:
    ids = list(values)
    if any(not item_id for item_id in ids):
        msg = f"{label.title()} identifiers must not be empty"
        raise ValueError(msg)
    if len(ids) != len(set(ids)):
        msg = f"{label.title()} identifiers must be unique within a document"
        raise ValueError(msg)
    return tuple(ids)
