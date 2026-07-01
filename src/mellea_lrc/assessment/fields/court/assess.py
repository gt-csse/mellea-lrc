"""Deterministic citation-court assessment."""

from mellea_lrc.assessment.fields.court.inference import infer_court_from_reporter
from mellea_lrc.assessment.types.court import (
    CourtAssessment,
    CourtAssessmentRun,
    CourtAssessmentStatus,
    CourtFollowupNotRequired,
    CourtInferredFromReporter,
)


def assess_court_exact_match(
    *,
    extracted_court: str | None,
    courtlistener_court_id: str | None,
) -> CourtAssessment:
    """Compare a citation court slug with the CourtListener court ID."""
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


def assess_court(
    *,
    extracted_court: str | None,
    courtlistener_court_id: str | None,
    reporter: str | None,
) -> CourtAssessmentRun:
    """Assess court against CourtListener, applying reporter inference when appropriate."""
    initial = assess_court_exact_match(
        extracted_court=extracted_court,
        courtlistener_court_id=courtlistener_court_id,
    )
    if initial.status is not CourtAssessmentStatus.MISSING:
        return CourtAssessmentRun(initial=initial, followup=CourtFollowupNotRequired())

    inferred = infer_court_from_reporter(reporter)
    if inferred is None:
        return CourtAssessmentRun(initial=initial, followup=CourtFollowupNotRequired())

    result = assess_court_exact_match(
        extracted_court=inferred,
        courtlistener_court_id=courtlistener_court_id,
    )
    return CourtAssessmentRun(
        initial=initial,
        followup=CourtInferredFromReporter(
            reporter=reporter,
            citation_court_before=extracted_court,
            result=result,
        ),
    )
