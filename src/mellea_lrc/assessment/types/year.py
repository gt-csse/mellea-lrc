"""Citation-year assessment types."""

from dataclasses import dataclass
from enum import Enum


class YearAssessmentStatus(str, Enum):
    """Canonical outcomes for deterministic year assessment."""

    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class YearAssessment:
    """Substantive comparison of extracted and retrieved citation years."""

    status: YearAssessmentStatus
    extracted_year: str | None
    courtlistener_year: str | None
    message: str
