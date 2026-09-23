"""Pin-cite readings have typed targets and never store failed normalization."""

import pytest

from mellea_lrc.extraction.normalization import normalize_pin_cite
from mellea_lrc.model import (
    CaseNameField,
    CourtField,
    Document,
    PinCiteField,
    PinCiteKind,
    PinCiteTarget,
    Span,
)
from mellea_lrc.extraction import grow_roots


@pytest.mark.parametrize(
    ("quote", "first", "last", "kind"),
    [
        ("495", 495, 495, PinCiteKind.PAGE),
        ("*7", 7, 7, PinCiteKind.STAR),
        ("123-25", 123, 125, PinCiteKind.PAGE),
        ("199–02", 199, 202, PinCiteKind.PAGE),
        ("*2-3", 2, 3, PinCiteKind.STAR),
    ],
)
def test_simple_parsed_pin_cites_have_typed_ranges(
    quote: str, first: int, last: int, kind: PinCiteKind
) -> None:
    assert normalize_pin_cite(quote) == (PinCiteTarget(first=first, last=last, kind=kind),)


@pytest.mark.parametrize("quote", ["", "at 4", "p. 4", "3-2", "*0", "4a"])
def test_unrecognized_or_invalid_pin_cites_raise(quote: str) -> None:
    with pytest.raises(ValueError):
        normalize_pin_cite(quote)


def test_parsed_field_cannot_store_missing_normalization() -> None:
    source = "Smith v. Jones, 347 U.S. 483, 495 (1954)."
    span = Span(source.index("495"), source.index("495") + 3)
    for value in (None, ()):
        with pytest.raises(ValueError):
            PinCiteField.from_source(source, span, normalized=value, node_id="pin:node:0")
    for field in (CaseNameField, CourtField):
        for value in (None, "", "   "):
            with pytest.raises(ValueError):
                field.from_source(source, span, normalized=value, node_id="field:node:0")


def test_pipeline_pin_cite_normalization_roundtrips() -> None:
    document = grow_roots(Document.from_source("See 347 U.S. 483, 495-97 (1954)."))
    pin = document.citations[0].pin_cite[-1]
    assert pin.quote == "495-97"
    assert pin.normalized == (PinCiteTarget(first=495, last=497, kind=PinCiteKind.PAGE),)
    assert Document.model_validate_json(document.model_dump_json()) == document
