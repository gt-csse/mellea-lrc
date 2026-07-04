"""Citation-court assessment types."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class CourtAssessmentStatus(str, Enum):
    """Canonical outcomes for deterministic court assessment."""

    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CourtAssessment:
    """Substantive comparison of a citation court slug and the CourtListener court."""

    status: CourtAssessmentStatus
    extracted_court: str | None
    courtlistener_court_id: str | None
    message: str
    source: Literal["direct", "reporter_inferred"] = "direct"
