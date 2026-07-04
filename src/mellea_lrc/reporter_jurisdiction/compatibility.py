"""Candidate compatibility checks for reporter jurisdiction evidence."""

from mellea_lrc.reporter_jurisdiction.types import (
    CourtClass,
    ReporterCoverage,
    ReporterJurisdictionCompatibility,
    ReporterJurisdictionCompatibilityStatus,
    ReporterJurisdictionInference,
)


def compare_reporter_jurisdiction(
    inference: ReporterJurisdictionInference,
    *,
    candidate_court_id: str | None,
    candidate_court_class: CourtClass | None = None,
    candidate_jurisdiction_id: str | None = None,
) -> ReporterJurisdictionCompatibility:
    """Compare explicit candidate metadata without resolving candidate identity."""
    supplied = (candidate_court_id, candidate_court_class, candidate_jurisdiction_id)
    if not any(supplied):
        return _result(
            ReporterJurisdictionCompatibilityStatus.INDETERMINATE,
            supplied,
            "Candidate has no court metadata to compare.",
        )

    comparisons = (
        (candidate_court_id, inference.court_ids),
        (candidate_court_class, inference.court_classes),
        (candidate_jurisdiction_id, inference.jurisdiction_ids),
    )
    comparable_results = tuple(
        value in constraints for value, constraints in comparisons if value is not None and constraints
    )
    if comparable_results and all(comparable_results):
        return _result(
            ReporterJurisdictionCompatibilityStatus.COMPATIBLE,
            supplied,
            "Candidate metadata is compatible with a reporter-derived constraint.",
        )
    if comparable_results and inference.coverage is ReporterCoverage.EXHAUSTIVE:
        return _result(
            ReporterJurisdictionCompatibilityStatus.INCOMPATIBLE,
            supplied,
            "Candidate metadata falls outside exhaustive reporter-derived constraints.",
        )
    return _result(
        ReporterJurisdictionCompatibilityStatus.INDETERMINATE,
        supplied,
        "Reporter evidence is insufficient for deterministic compatibility.",
    )


def _result(
    status: ReporterJurisdictionCompatibilityStatus,
    supplied: tuple[str | None, CourtClass | None, str | None],
    message: str,
) -> ReporterJurisdictionCompatibility:
    return ReporterJurisdictionCompatibility(
        status=status,
        candidate_court_id=supplied[0],
        candidate_court_class=supplied[1],
        candidate_jurisdiction_id=supplied[2],
        message=message,
    )
