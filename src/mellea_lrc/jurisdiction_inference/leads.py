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
from mellea_lrc.courtlistener.taxonomy import get_courts_db_classification, is_recognized_court


def evaluate_court_inference(extracted_court: str | None) -> CourtInference:
    """Evaluate a citation's explicit extracted court string.

    Returns the `courts_db_classification` (a snapshot of the Free Law Project
    `courts-db` package) for the slug, if recognized. The classification is a
    lookup of an already-extracted court slug; it is not a verification of the
    citation's locator and it does not assert that the cited case exists.
    """
    if extracted_court is None or not extracted_court.strip():
        return CourtInference(
            extracted_court=None,
            status=CourtInferenceStatus.MISSING_COURT,
            courts_db_classification=None,
        )

    canonical_court = extracted_court.strip().lower()
    if not is_recognized_court(canonical_court):
        return CourtInference(
            extracted_court=canonical_court,
            status=CourtInferenceStatus.UNRECOGNIZED,
            courts_db_classification=None,
        )

    classification = get_courts_db_classification(canonical_court)
    return CourtInference(
        extracted_court=canonical_court,
        status=CourtInferenceStatus.RESOLVED,
        courts_db_classification=classification,
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

    edition = reporter.edition_short_name.strip()
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
