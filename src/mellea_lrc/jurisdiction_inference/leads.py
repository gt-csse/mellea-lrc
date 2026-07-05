"""Citation Jurisdiction Inference."""

from mellea_lrc.core.citations import Reporter
from mellea_lrc.jurisdiction_inference.registry import (
    VALID_REPORTERS,
    REPORTER_MLZ_JURISDICTIONS,
)
from mellea_lrc.jurisdiction_inference.types import (
    ReporterInference,
    CourtInference,
    ReporterInferenceStatus,
    CourtInferenceStatus,
)
from mellea_lrc.courtlistener.taxonomy import get_court_taxonomy, is_recognized_court


def evaluate_court_inference(extracted_court: str | None) -> CourtInference:
    """Evaluate a citation's explicit extracted court string."""
    if extracted_court is None or not extracted_court.strip():
        return CourtInference(
            extracted_court=None,
            status=CourtInferenceStatus.MISSING_COURT,
            cl_court_taxonomy=None,
        )

    canonical_court = extracted_court.strip().lower()
    if not is_recognized_court(canonical_court):
        return CourtInference(
            extracted_court=canonical_court,
            status=CourtInferenceStatus.UNRECOGNIZED,
            cl_court_taxonomy=None,
        )

    taxonomy = get_court_taxonomy(canonical_court)
    return CourtInference(
        extracted_court=canonical_court,
        status=CourtInferenceStatus.RESOLVED,
        cl_court_taxonomy=taxonomy,
    )


def evaluate_reporter_inference(
    reporter: Reporter | None,
) -> ReporterInference:
    if reporter is None:
        return ReporterInference(
            reporter=None,
            status=ReporterInferenceStatus.MISSING_REPORTER,
            mlz_jurisdictions=(),
        )

    edition = reporter.edition.strip()
    if not edition or edition not in VALID_REPORTERS:
        return ReporterInference(
            reporter=reporter,
            status=ReporterInferenceStatus.UNRECOGNIZED,
            mlz_jurisdictions=(),
        )

    return ReporterInference(
        reporter=reporter,
        status=ReporterInferenceStatus.RECOGNIZED,
        mlz_jurisdictions=tuple(REPORTER_MLZ_JURISDICTIONS.get(edition, [])),
    )
