"""Pin-cite readings have typed targets and explicit normalization state."""

import pytest

from mellea_lrc.extraction import grow_roots
from mellea_lrc.model import (
    CaseName,
    CaseNameField,
    CaseNameKind,
    CourtField,
    Document,
    PinCiteField,
    PinCiteKind,
    PinCiteTarget,
    Span,
)
from mellea_lrc.model.citations.fields.pin_cite import normalize_pin_cite


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


def test_normalizable_field_requires_a_matching_payload_and_no_error() -> None:
    source = "Smith v. Jones, 347 U.S. 483, 495 (1954)."
    span = Span(source.index("495"), source.index("495") + 3)
    pin = PinCiteField.from_source(source, span, node_id="pin:node:0")
    valid = pin.model_dump(mode="json")
    assert pin.normalizable is True
    assert pin.get_normalized()
    with pytest.raises(AttributeError):
        pin.normalized
    assert valid["normalizable"] is True
    assert valid["normalization_error"] is None
    for value in (None, ()):
        with pytest.raises(ValueError):
            PinCiteField.model_validate({**valid, "normalized": value})
    with pytest.raises(ValueError):
        PinCiteField.model_validate({**valid, "normalization_error": "unexpected error"})
    without_state = valid.copy()
    del without_state["normalizable"]
    with pytest.raises(ValueError):
        PinCiteField.model_validate(without_state)

    name_span = Span(0, len("Smith v. Jones"))
    name = CaseNameField.from_source(source, name_span, node_id="name:node:0")
    with pytest.raises(ValueError):
        CaseNameField.model_validate({**name.model_dump(mode="json"), "normalized": None})
    with pytest.raises(ValueError, match="does not match its quote"):
        CaseNameField.model_validate(
            {
                **name.model_dump(mode="json"),
                "normalized": CaseName(kind=CaseNameKind.ADVERSARIAL, plaintiff="Smith", defendant="Brown"),
            }
        )

    court_source = "D. Ariz."
    court = CourtField.from_source(court_source, Span(0, len(court_source)), node_id="court:node:0")
    for value in (None, "", "   "):
        with pytest.raises(ValueError):
            CourtField.model_validate({**court.model_dump(mode="json"), "normalized": value})


def test_failed_field_rejects_inconsistent_serialized_state() -> None:
    source = "D. Fiction"
    failed = CourtField.from_source(source, Span(0, len(source)), node_id="court:node:0")
    saved = failed.model_dump(mode="json")
    assert saved["normalizable"] is False
    assert saved["normalized"] is None
    assert isinstance(saved["normalization_error"], str)

    valid = CourtField.from_source("D. Ariz.", Span(0, len("D. Ariz.")), node_id="court:node:0")
    inconsistent = (
        {**saved, "normalizable": True},
        {**saved, "normalized": valid.model_dump(mode="json")["normalized"]},
        {**saved, "normalization_error": None},
        {**saved, "normalization_error": ""},
    )
    for payload in inconsistent:
        with pytest.raises(ValueError):
            CourtField.model_validate(payload)

    without_state = saved.copy()
    del without_state["normalizable"]
    with pytest.raises(ValueError):
        CourtField.model_validate(without_state)
    assert CourtField.model_validate_json(failed.model_dump_json()) == failed


def test_pipeline_pin_cite_normalization_roundtrips() -> None:
    document = grow_roots(Document.from_source("See 347 U.S. 483, 495-97 (1954)."))
    pin = document.citations[0].pin_cite[-1]
    assert pin.quote == "495-97"
    assert pin.get_normalized() == (PinCiteTarget(first=495, last=497, kind=PinCiteKind.PAGE),)
    assert Document.model_validate_json(document.model_dump_json()) == document
