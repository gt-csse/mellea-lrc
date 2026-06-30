"""Citation-court assessment types."""

from dataclasses import dataclass
from enum import Enum


class CourtAssessmentStatus(str, Enum):
    """Canonical outcomes for deterministic court assessment."""

    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CourtAssessment:
    """Comparison of an extracted court and the retrieved docket court."""

    status: CourtAssessmentStatus
    extracted_court: str | None
    courtlistener_court_id: str | None
    message: str
