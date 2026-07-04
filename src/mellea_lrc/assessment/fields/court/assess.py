"""Deterministic citation-court assessment."""

from typing import Literal

from mellea_lrc.assessment.types.court import (
    CourtAssessment,
    CourtAssessmentStatus,
)


def assess_court(
    *,
    extracted_court: str | None,
    courtlistener_court_id: str | None,
    source: Literal["direct", "reporter_inferred"] = "direct",
) -> CourtAssessment:
    """Compare a citation court slug with the CourtListener court ID."""
    if not extracted_court or not courtlistener_court_id:
        return CourtAssessment(
            status=CourtAssessmentStatus.MISSING,
            extracted_court=extracted_court,
            courtlistener_court_id=courtlistener_court_id,
            message="Extracted or CourtListener court is missing.",
            source=source,
        )
    if extracted_court == courtlistener_court_id:
        return CourtAssessment(
            status=CourtAssessmentStatus.EXACT_MATCH,
            extracted_court=extracted_court,
            courtlistener_court_id=courtlistener_court_id,
            message="Extracted court exactly matches CourtListener.",
            source=source,
        )
    return CourtAssessment(
        status=CourtAssessmentStatus.MISMATCH,
        extracted_court=extracted_court,
        courtlistener_court_id=courtlistener_court_id,
        message="Extracted court does not match CourtListener.",
        source=source,
    )
