"""Assessment types for LLM-assisted citation checks."""

from dataclasses import dataclass
from enum import Enum


class CaseNameAssessmentStatus(str, Enum):
    """Canonical outcomes for case-name assessment."""

    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    NEEDS_SEMANTIC_ASSESSMENT = "needs_semantic_assessment"
    EXTRACTION_ERROR = "extraction_error"


@dataclass(frozen=True, slots=True)
class CaseNameAssessment:
    """Assessment result for one extracted case name."""

    citation_id: str
    status: CaseNameAssessmentStatus
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    message: str

