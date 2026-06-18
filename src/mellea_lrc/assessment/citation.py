"""Citation-level assessment helpers."""

from mellea_lrc.assessment.types import (
    YearAssessment,
    YearAssessmentStatus,
)


def assess_year_exact_match(
    *,
    citation_id: str,
    extracted_year: str | None,
    courtlistener_year: str | None,
) -> YearAssessment:
    """Assess citation year with deterministic string equality."""
    if not extracted_year or not courtlistener_year:
        return YearAssessment(
            citation_id=citation_id,
            status=YearAssessmentStatus.MISSING,
            extracted_year=extracted_year,
            courtlistener_year=courtlistener_year,
            message="Extracted year or CourtListener year is missing.",
        )
    if extracted_year == courtlistener_year:
        return YearAssessment(
            citation_id=citation_id,
            status=YearAssessmentStatus.EXACT_MATCH,
            extracted_year=extracted_year,
            courtlistener_year=courtlistener_year,
            message="Extracted year exactly matches CourtListener.",
        )
    return YearAssessment(
        citation_id=citation_id,
        status=YearAssessmentStatus.MISMATCH,
        extracted_year=extracted_year,
        courtlistener_year=courtlistener_year,
        message="Extracted year does not match CourtListener.",
    )
