"""Grounding contracts for model-led docket-number recovery."""

from mellea_lrc.validation.field_checks.mellea_docket_citation_reextraction import _grounded_number


def test_docket_review_grounds_whitespace_and_minor_character_variation() -> None:
    source = "Case No. 1: 24-cv-08760"

    assert _grounded_number(source, "1:24-cv-08760") == "1: 24-cv-08760"
    assert _grounded_number(source, "Case No. 1:24-cv-08760") is None
    assert _grounded_number(source, "1:24-cv-0876O") == "1: 24-cv-08760"


def test_docket_review_refuses_an_ambiguous_or_non_source_number() -> None:
    source = "Case Nos. 1:24-cv-08760 and 1:24-cv-08761"

    assert _grounded_number(source, "1:24-cv-08760") == "1:24-cv-08760"
    assert _grounded_number(source, "1:24-cv-09999") is None
    assert _grounded_number(source, "1:24-cv-08762") is None


def test_docket_review_grounds_an_opaque_identifier_that_begins_with_letters() -> None:
    source = "No. CIV 31-00420 AX/QZ"

    assert _grounded_number(source, "CIV 31-00420 AX/QZ") == "CIV 31-00420 AX/QZ"
