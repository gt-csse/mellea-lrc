"""Grounding contracts for model-led docket-number recovery."""

from types import SimpleNamespace

from mellea_lrc.validation.field_checks.mellea_docket_citation_reextraction import (
    _grounded_number,
    _validate_reextraction,
)


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


def test_docket_review_rejects_case_name_absent_from_source_citation() -> None:
    output = (
        '{"case_name":"Brown v. Board","docket_number":"20 Civ. 6835",'
        '"court":null,"date":null,"pin_cite":null,"reason":"Re-read citation."}'
    )
    context = SimpleNamespace(last_output=lambda: SimpleNamespace(value=output))

    validation = _validate_reextraction(
        context,
        source_locator="20 Civ. 6835",
        source_citation="Smith v. Jones, No. 20 Civ. 6835 (S.D.N.Y.).",
    )

    assert validation.as_bool() is False
    assert "case_name" in validation.reason
