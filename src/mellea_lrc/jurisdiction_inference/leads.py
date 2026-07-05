"""Citation Jurisdiction Leads inference."""

from mellea_lrc.jurisdiction_inference.registry import (
    VALID_REPORTERS,
    REPORTER_MLZ_JURISDICTIONS,
)
from mellea_lrc.jurisdiction_inference.types import (
    JurisdictionInference,
    ReporterLead,
    CourtLead,
    ReporterLeadStatus,
    CourtLeadStatus,
)
from mellea_lrc.courtlistener.taxonomy import get_court_taxonomy, is_recognized_court
from mellea_lrc.jurisdiction_inference.translation import triangulate_court_id


def evaluate_court_lead(extracted_court: str | None) -> CourtLead:
    """Evaluate a citation's explicit extracted court string."""
    if extracted_court is None or not extracted_court.strip():
        return CourtLead(
            extracted_court=None,
            status=CourtLeadStatus.MISSING_COURT,
            cl_court_taxonomy=None,
        )

    canonical_court = extracted_court.strip().lower()
    if not is_recognized_court(canonical_court):
        return CourtLead(
            extracted_court=canonical_court,
            status=CourtLeadStatus.UNRECOGNIZED,
            cl_court_taxonomy=None,
        )

    taxonomy = get_court_taxonomy(canonical_court)
    return CourtLead(
        extracted_court=canonical_court,
        status=CourtLeadStatus.RESOLVED,
        cl_court_taxonomy=taxonomy,
    )


def evaluate_reporter_lead(reporter: str | None) -> ReporterLead:
    if reporter is None or not reporter.strip():
        return ReporterLead(
            reporter=None,
            status=ReporterLeadStatus.MISSING_REPORTER,
            mlz_jurisdictions=(),
        )

    canonical = reporter.strip()
    if canonical not in VALID_REPORTERS:
        return ReporterLead(
            reporter=canonical,
            status=ReporterLeadStatus.UNRECOGNIZED,
            mlz_jurisdictions=(),
        )

    return ReporterLead(
        reporter=canonical,
        status=ReporterLeadStatus.RECOGNIZED,
        mlz_jurisdictions=tuple(REPORTER_MLZ_JURISDICTIONS.get(canonical, [])),
    )
