import pytest

from mellea_lrc.reporter_jurisdiction import (
    ReporterJurisdictionStatus,
    infer_reporter_jurisdiction,
)
from mellea_lrc.reporter_jurisdiction.registry import EXHAUSTIVE_REPORTERS, VALID_REPORTERS
from mellea_lrc.courtlistener import is_recognized_court


def test_exhaustive_reporters_are_subset_of_valid_reporters() -> None:
    """Every key in EXHAUSTIVE_REPORTERS must appear in VALID_REPORTERS."""
    assert set(EXHAUSTIVE_REPORTERS.keys()).issubset(VALID_REPORTERS)


def test_exhaustive_reporter_infers_exact_court() -> None:
    inference = infer_reporter_jurisdiction("U.S.")

    assert inference.status is ReporterJurisdictionStatus.EXHAUSTIVE_REPORTER
    assert inference.court_ids == ("scotus",)
    assert inference.exact_court_id == "scotus"
    assert inference.evidence


@pytest.mark.parametrize(
    "reporter",
    ("F.3d", "F.4th", "F. Supp. 2d", "F. Supp. 3d", "B.R.", "WL", "LEXIS"),
)
def test_valid_reporter_without_exact_court(reporter: str) -> None:
    inference = infer_reporter_jurisdiction(reporter)

    assert inference.status is ReporterJurisdictionStatus.VALID_REPORTER
    assert inference.court_ids == ()
    assert inference.exact_court_id is None


def test_missing_and_unrecognized_terminate() -> None:
    assert (
        infer_reporter_jurisdiction(None).status
        is ReporterJurisdictionStatus.MISSING_REPORTER
    )
    assert (
        infer_reporter_jurisdiction("Imaginary Reporter").status
        is ReporterJurisdictionStatus.UNRECOGNIZED
    )


def test_exhaustive_is_a_valid_reporter_by_status_hierarchy() -> None:
    """EXHAUSTIVE_REPORTER is a VALID_REPORTER — both are non-terminal statuses."""
    valid_statuses = {
        ReporterJurisdictionStatus.VALID_REPORTER,
        ReporterJurisdictionStatus.EXHAUSTIVE_REPORTER,
    }
    assert infer_reporter_jurisdiction("U.S.").status in valid_statuses
    assert infer_reporter_jurisdiction("F.3d").status in valid_statuses


def test_exact_court_id_is_none_for_non_exhaustive() -> None:
    assert infer_reporter_jurisdiction("WL").exact_court_id is None
    assert infer_reporter_jurisdiction("F.4th").exact_court_id is None
    assert infer_reporter_jurisdiction(None).exact_court_id is None
    assert infer_reporter_jurisdiction("Unknown").exact_court_id is None


def test_exhaustive_reporters_map_to_recognized_courts() -> None:
    """Every exact court ID from EXHAUSTIVE_REPORTERS must be recognized."""
    for scope in EXHAUSTIVE_REPORTERS.values():
        assert is_recognized_court(scope.court_id), f"Court {scope.court_id} is not recognized"

