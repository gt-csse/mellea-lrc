"""Pin-cite readings have typed targets and explicit normalization state."""

import asyncio
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
        ("998 -1003", 998, 1003, PinCiteKind.PAGE),
        ("337 - 38", 337, 338, PinCiteKind.PAGE),
        ("*2 -3", 2, 3, PinCiteKind.STAR),
    ],
)
def test_simple_parsed_pin_cites_have_typed_ranges(
    quote: str, first: int, last: int, kind: PinCiteKind
) -> None:
    assert normalize_pin_cite(quote) == (PinCiteTarget(first=first, last=last, kind=kind),)


@pytest.mark.parametrize("quote", ["", "p. 4", "3-2", "*0", "4a", "495 nn. 3 and 4"])
def test_unrecognized_or_invalid_pin_cites_raise(quote: str) -> None:
    with pytest.raises(ValueError):
        normalize_pin_cite(quote)


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        (
            "495, 497-99",
            (
                PinCiteTarget(first=495, last=495, kind=PinCiteKind.PAGE),
                PinCiteTarget(first=497, last=499, kind=PinCiteKind.PAGE),
            ),
        ),
        (
            "¶¶ 4, 6-8",
            (
                PinCiteTarget(first=4, last=4, kind=PinCiteKind.PARAGRAPH),
                PinCiteTarget(first=6, last=8, kind=PinCiteKind.PARAGRAPH),
            ),
        ),
        ("¶ 4", (PinCiteTarget(first=4, last=4, kind=PinCiteKind.PARAGRAPH),)),
        ("495 n.3", (PinCiteTarget(first=495, last=495, kind=PinCiteKind.PAGE, footnote="3"),)),
        ("495 & n.3", (PinCiteTarget(first=495, last=495, kind=PinCiteKind.PAGE, footnote="3"),)),
        ("495 nn.3-4", (PinCiteTarget(first=495, last=495, kind=PinCiteKind.PAGE, footnote="3-4"),)),
        ("at 4", (PinCiteTarget(first=4, last=4, kind=PinCiteKind.PAGE),)),
        ("at *4", (PinCiteTarget(first=4, last=4, kind=PinCiteKind.STAR),)),
    ],
)
def test_structured_pin_cites_normalize_and_pipeline_quotes_are_grounded(
    quote: str, expected: tuple[PinCiteTarget, ...]
) -> None:
    assert normalize_pin_cite(quote) == expected

    source = f"See 347 U.S. 483, {quote} (1954)."
    document = asyncio.run(grow_roots(Document.from_source(source)))
    pin = document.citations[0].pin_cite[-1]
    written_pin = quote.removeprefix("at ")

    assert pin.quote == written_pin
    assert source[pin.span.start : pin.span.end] == written_pin
    assert pin.get_normalized() == expected
    assert Document.model_validate_json(document.model_dump_json()) == document


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
    document = asyncio.run(grow_roots(Document.from_source("See 347 U.S. 483, 495-97 (1954).")))
    pin = document.citations[0].pin_cite[-1]
    assert pin.quote == "495-97"
    assert pin.get_normalized() == (PinCiteTarget(first=495, last=497, kind=PinCiteKind.PAGE),)
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_spaced_range_is_read_whole_with_its_exact_source_span() -> None:
    source = "See 347 U.S. 483, 998 -1003 (1954)."
    document = asyncio.run(grow_roots(Document.from_source(source)))
    pin = document.citations[0].pin_cite[-1]

    assert pin.quote == "998 -1003"
    assert source[pin.span.start : pin.span.end] == pin.quote
    assert pin.get_normalized() == (PinCiteTarget(first=998, last=1003, kind=PinCiteKind.PAGE),)
    assert Document.model_validate_json(document.model_dump_json()) == document


@pytest.mark.parametrize("quote", ["495a", "495, 497a", "495 n.x", "998 -", "907-\n\n08"])
def test_malformed_pin_continuation_is_kept_for_review(quote: str) -> None:
    source = f"See 347 U.S. 483, {quote} (1954)."
    document = asyncio.run(grow_roots(Document.from_source(source)))
    pin = document.citations[0].pin_cite[-1]

    assert pin.quote == quote
    assert source[pin.span.start : pin.span.end] == quote
    assert pin.normalizable is False
    assert pin.normalization_error
    with pytest.raises(ValueError, match="not normalizable"):
        pin.get_normalized()
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_adjacent_court_ordinal_is_not_a_pin_cite() -> None:
    document = asyncio.run(grow_roots(Document.from_source("See 155 A.D.3d 781, 2d Dept. 2017.")))

    assert document.citations[0].pin_cite == ()


def test_parallel_reporter_volume_is_not_a_second_pin_target() -> None:
    source = "See 347 U.S. 483, 495, 150 X.2d 250 (1954)."
    document = asyncio.run(grow_roots(Document.from_source(source)))

    pin = document.citations[0].pin_cite[-1]
    assert pin.quote == "495"
    assert pin.get_normalized() == (PinCiteTarget(first=495, last=495, kind=PinCiteKind.PAGE),)
