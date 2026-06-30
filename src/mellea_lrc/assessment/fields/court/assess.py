"""Deterministic citation-court assessment."""

from mellea_lrc.assessment.types.court import CourtAssessment, CourtAssessmentStatus


def assess_court_exact_match(
    *,
    extracted_court: str | None,
    courtlistener_court_id: str | None,
) -> CourtAssessment:
    """Compare the extracted court slug with the retrieved docket court ID."""
    if not extracted_court or not courtlistener_court_id:
        return CourtAssessment(
            status=CourtAssessmentStatus.MISSING,
            extracted_court=extracted_court,
            courtlistener_court_id=courtlistener_court_id,
            message="Extracted or CourtListener court is missing.",
        )
    if extracted_court == courtlistener_court_id:
        return CourtAssessment(
            status=CourtAssessmentStatus.EXACT_MATCH,
            extracted_court=extracted_court,
            courtlistener_court_id=courtlistener_court_id,
            message="Extracted court exactly matches CourtListener.",
        )
    return CourtAssessment(
        status=CourtAssessmentStatus.MISMATCH,
        extracted_court=extracted_court,
        courtlistener_court_id=courtlistener_court_id,
        message="Extracted court does not match CourtListener.",
    )
