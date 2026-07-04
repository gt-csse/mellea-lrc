import pytest

from mellea_lrc.reporter_jurisdiction import (
    CourtClass,
    ReporterCoverage,
    ReporterJurisdictionCompatibilityStatus,
    ReporterJurisdictionStatus,
    compare_reporter_jurisdiction,
    infer_reporter_jurisdiction,
)


def test_exact_court_is_exhaustive_singleton_projection() -> None:
    inference = infer_reporter_jurisdiction("U.S.")

    assert inference.status is ReporterJurisdictionStatus.CONSTRAINED
    assert inference.coverage is ReporterCoverage.EXHAUSTIVE
    assert inference.court_ids == ("scotus",)
    assert inference.court_classes == (CourtClass.FEDERAL_SUPREME,)
    assert inference.exact_court_id == "scotus"
    assert inference.evidence


@pytest.mark.parametrize(
    ("reporter", "court_class"),
    (
        ("F.3d", CourtClass.FEDERAL_APPELLATE),
        ("F.4th", CourtClass.FEDERAL_APPELLATE),
        ("F. Supp. 2d", CourtClass.FEDERAL_DISTRICT),
        ("F. Supp. 3d", CourtClass.FEDERAL_DISTRICT),
    ),
)
def test_reporter_can_constrain_court_class_without_exact_court(
    reporter: str,
    court_class: CourtClass,
) -> None:
    inference = infer_reporter_jurisdiction(reporter)

    assert inference.status is ReporterJurisdictionStatus.CONSTRAINED
    assert inference.court_classes == (court_class,)
    assert inference.exact_court_id is None


def test_missing_unrecognized_and_unconstrained_are_distinct() -> None:
    assert (
        infer_reporter_jurisdiction(None).status
        is ReporterJurisdictionStatus.MISSING_REPORTER
    )
    assert (
        infer_reporter_jurisdiction("Imaginary Reporter").status
        is ReporterJurisdictionStatus.UNRECOGNIZED
    )
    assert (
        infer_reporter_jurisdiction("WL").status
        is ReporterJurisdictionStatus.RECOGNIZED_WITHOUT_CONSTRAINT
    )


def test_compatibility_uses_explicit_candidate_class() -> None:
    inference = infer_reporter_jurisdiction("F.4th")

    compatible = compare_reporter_jurisdiction(
        inference,
        candidate_court_id="ca10",
        candidate_court_class=CourtClass.FEDERAL_APPELLATE,
        candidate_jurisdiction_id="us-federal",
    )
    incompatible = compare_reporter_jurisdiction(
        inference,
        candidate_court_id="azd",
        candidate_court_class=CourtClass.FEDERAL_DISTRICT,
        candidate_jurisdiction_id="us-federal",
    )

    assert compatible.status is ReporterJurisdictionCompatibilityStatus.COMPATIBLE
    assert incompatible.status is ReporterJurisdictionCompatibilityStatus.INCOMPATIBLE


def test_exact_court_can_exclude_wrong_candidate() -> None:
    result = compare_reporter_jurisdiction(
        infer_reporter_jurisdiction("U.S."),
        candidate_court_id="ca9",
    )

    assert result.status is ReporterJurisdictionCompatibilityStatus.INCOMPATIBLE
