"""Pure reporter-to-jurisdiction inference."""

from mellea_lrc.reporter_jurisdiction.registry import (
    RECOGNIZED_WITHOUT_CONSTRAINT,
    REPORTER_SCOPES,
)
from mellea_lrc.reporter_jurisdiction.types import (
    ReporterJurisdictionEvidence,
    ReporterJurisdictionInference,
    ReporterJurisdictionStatus,
)

REGISTRY_SOURCE = "mellea-lrc curated reporter registry"


def infer_reporter_jurisdiction(reporter: str | None) -> ReporterJurisdictionInference:
    """Return explicit jurisdiction evidence for a canonical reporter value."""
    if reporter is None or not reporter.strip():
        return ReporterJurisdictionInference(
            reporter=None,
            status=ReporterJurisdictionStatus.MISSING_REPORTER,
        )

    canonical = reporter.strip()
    scope = REPORTER_SCOPES.get(canonical)
    if scope is not None:
        return ReporterJurisdictionInference(
            reporter=canonical,
            status=ReporterJurisdictionStatus.CONSTRAINED,
            court_ids=scope.court_ids,
            court_classes=scope.court_classes,
            jurisdiction_ids=scope.jurisdiction_ids,
            coverage=scope.coverage,
            evidence=(
                ReporterJurisdictionEvidence(
                    source=REGISTRY_SOURCE,
                    statement=scope.statement,
                ),
            ),
        )

    if canonical in RECOGNIZED_WITHOUT_CONSTRAINT:
        return ReporterJurisdictionInference(
            reporter=canonical,
            status=ReporterJurisdictionStatus.RECOGNIZED_WITHOUT_CONSTRAINT,
        )
    return ReporterJurisdictionInference(
        reporter=canonical,
        status=ReporterJurisdictionStatus.UNRECOGNIZED,
    )

