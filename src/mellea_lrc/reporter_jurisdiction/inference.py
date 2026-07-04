"""Pure reporter-to-jurisdiction inference."""

from mellea_lrc.reporter_jurisdiction.registry import (
    EXHAUSTIVE_REPORTERS,
    VALID_REPORTERS,
)
from mellea_lrc.reporter_jurisdiction.types import (
    ReporterJurisdictionEvidence,
    ReporterJurisdictionInference,
    ReporterJurisdictionStatus,
)
from mellea_lrc.courtlistener.taxonomy import get_court_taxonomy

REGISTRY_SOURCE = "mellea-lrc curated reporter registry"


def infer_reporter_jurisdiction(reporter: str | None) -> ReporterJurisdictionInference:
    """Return explicit jurisdiction evidence for a canonical reporter value."""
    if reporter is None or not reporter.strip():
        return ReporterJurisdictionInference(
            reporter=None,
            status=ReporterJurisdictionStatus.MISSING_REPORTER,
        )

    canonical = reporter.strip()

    if canonical not in VALID_REPORTERS:
        return ReporterJurisdictionInference(
            reporter=canonical,
            status=ReporterJurisdictionStatus.UNRECOGNIZED,
        )

    scope = EXHAUSTIVE_REPORTERS.get(canonical)
    if scope is not None:
        taxonomy = get_court_taxonomy(scope.court_id)
        return ReporterJurisdictionInference(
            reporter=canonical,
            status=ReporterJurisdictionStatus.EXHAUSTIVE_REPORTER,
            court_ids=(scope.court_id,),
            evidence=(
                ReporterJurisdictionEvidence(
                    source=REGISTRY_SOURCE,
                    statement=scope.statement,
                ),
            ),
            cl_court_taxonomy=taxonomy,
        )

    return ReporterJurisdictionInference(
        reporter=canonical,
        status=ReporterJurisdictionStatus.VALID_REPORTER,
    )
