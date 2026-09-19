"""Grounding contracts for model-led docket-number recovery."""

from mellea_lrc.validation.field_checks.mellea_docket_number_review import _grounded_number


def test_docket_review_accepts_only_whitespace_variation_from_the_source_locator() -> None:
    source = "Case No. 1: 24-cv-08760"

    assert _grounded_number(source, "1:24-cv-08760") == "1: 24-cv-08760"
    assert _grounded_number(source, "Case No. 1:24-cv-08760") is None
    assert _grounded_number(source, "1:24-cv-0876O") == "1: 24-cv-08760"


def test_docket_review_refuses_an_ambiguous_or_non_source_number() -> None:
    source = "Case Nos. 1:24-cv-08760 and 1:24-cv-08761"

    assert _grounded_number(source, "1:24-cv-08760") == "1:24-cv-08760"
    assert _grounded_number(source, "1:24-cv-09999") is None
    assert _grounded_number(source, "1:24-cv-08762") is None
