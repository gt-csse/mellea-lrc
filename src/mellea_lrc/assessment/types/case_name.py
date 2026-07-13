"""Case-name assessment types."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeAlias

from mellea_lrc.assessment.context import is_text_in_context
from mellea_lrc.assessment.types.common import ChatTurn
from mellea_lrc.core.spans import Span


class CaseNameAssessmentStatus(str, Enum):
    """Canonical substantive conclusions for case-name assessment."""

    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    NOT_SEMANTIC_MATCH = "not_semantic_match"
    UNASSESSABLE = "unassessable"


class CaseNameFollowupStatus(str, Enum):
    """Execution outcome after the initial case-name assessment."""

    NOT_REQUIRED = "not_required"
    REASSESSED = "reassessed"
    REEXTRACTION_FAILED = "reextraction_failed"
    REASSESSMENT_FAILED = "reassessment_failed"


@dataclass(frozen=True, slots=True)
class CaseNameAssessment:
    """Substantive comparison of an extracted and retrieved case name."""

    status: CaseNameAssessmentStatus
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    message: str
    chat_history: tuple[ChatTurn, ...] | None = None


@dataclass(frozen=True, slots=True)
class CaseNameProposal:
    """Non-empty case name proposed from a local document context."""

    case_name: str

    def __post_init__(self) -> None:
        if not self.case_name.strip():
            msg = "CaseNameProposal.case_name must not be empty"
            raise ValueError(msg)

    def valid(self, document_context: str) -> bool:
        """Return whether the proposal is present in the supplied context."""
        return is_text_in_context(self.case_name, document_context)


@dataclass(frozen=True, slots=True)
class ReextractedCaseName:
    """Case name grounded at document-local character offsets."""

    case_name: str
    case_name_span: Span

    def __post_init__(self) -> None:
        if not self.case_name.strip():
            msg = "ReextractedCaseName.case_name must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CaseNameReassessmentNotRequired:
    """The initial case-name assessment required no re-extraction."""

    status: ClassVar[CaseNameFollowupStatus] = CaseNameFollowupStatus.NOT_REQUIRED


@dataclass(frozen=True, slots=True)
class CaseNameReextractionFailed:
    """Case-name re-extraction failed before a grounded value existed."""

    status: ClassVar[CaseNameFollowupStatus] = CaseNameFollowupStatus.REEXTRACTION_FAILED
    error: str


@dataclass(frozen=True, slots=True)
class CaseNameReassessed:
    """A grounded case name was successfully reassessed."""

    status: ClassVar[CaseNameFollowupStatus] = CaseNameFollowupStatus.REASSESSED
    reextracted_case_name: ReextractedCaseName
    result: CaseNameAssessment


@dataclass(frozen=True, slots=True)
class CaseNameReassessmentFailed:
    """A grounded case name was produced but its assessment failed."""

    status: ClassVar[CaseNameFollowupStatus] = CaseNameFollowupStatus.REASSESSMENT_FAILED
    reextracted_case_name: ReextractedCaseName
    error: str


CaseNameFollowup: TypeAlias = (
    CaseNameReassessmentNotRequired
    | CaseNameReextractionFailed
    | CaseNameReassessed
    | CaseNameReassessmentFailed
)


@dataclass(frozen=True, slots=True)
class CaseNameAssessmentRun:
    """Initial case-name assessment and its field-local follow-up outcome."""

    initial: CaseNameAssessment
    followup: CaseNameFollowup

    def __post_init__(self) -> None:
        needs_followup = self.initial.status == CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH
        if needs_followup == isinstance(self.followup, CaseNameReassessmentNotRequired):
            msg = "Case-name assessment and follow-up states are inconsistent"
            raise ValueError(msg)
        if isinstance(self.followup, CaseNameReassessed) and self.followup.result.status not in {
            CaseNameAssessmentStatus.EXACT_MATCH,
            CaseNameAssessmentStatus.SEMANTIC_MATCH,
            CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
        }:
            msg = "Successful case-name reassessment must contain a terminal conclusion"
            raise ValueError(msg)
