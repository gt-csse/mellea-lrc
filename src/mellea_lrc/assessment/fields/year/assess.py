"""Deterministic citation-year assessment."""

from mellea_lrc.assessment.types.year import YearAssessment, YearAssessmentStatus


def assess_year_exact_match(
    *,
    extracted_year: str | None,
    courtlistener_year: str | None,
) -> YearAssessment:
    """Compare the extracted citation year with the retrieved year."""
    if not extracted_year or not courtlistener_year:
        return YearAssessment(
            status=YearAssessmentStatus.MISSING,
            extracted_year=extracted_year,
            courtlistener_year=courtlistener_year,
            message="Extracted or CourtListener year is missing.",
        )
    if extracted_year == courtlistener_year:
        return YearAssessment(
            status=YearAssessmentStatus.EXACT_MATCH,
            extracted_year=extracted_year,
            courtlistener_year=courtlistener_year,
            message="Extracted year exactly matches CourtListener.",
        )
    return YearAssessment(
        status=YearAssessmentStatus.MISMATCH,
        extracted_year=extracted_year,
        courtlistener_year=courtlistener_year,
        message="Extracted year does not match CourtListener.",
    )
