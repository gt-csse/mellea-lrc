"""Pinpoint fields retain exact source quotes and typed, inclusive targets."""

import pytest
from pydantic import ValidationError

from mellea_lrc.model.citations.fields.pin_cite import (
    PinCiteField,
    PinCiteKind,
    PinCiteTarget,
)
from mellea_lrc.model.span import Span


def read(quote):
    source = f"A citation at {quote}; the next sentence."
    start = source.index(quote)
    field = PinCiteField.from_source(
        source, Span(start=start, end=start + len(quote)), node_id="read-pinpoint"
    )
    field.validate_source(source)
    return field


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("557", [(557, 557, "page", None)]),
        ("at 557–59", [(557, 559, "page", None)]),
        ("AT\n557 – 559", [(557, 559, "page", None)]),
        ("199-02", [(199, 202, "page", None)]),
        ("*5–7", [(5, 7, "star", None)]),
        ("at *5, 8–9", [(5, 5, "star", None), (8, 9, "star", None)]),
        ("¶ 12", [(12, 12, "paragraph", None)]),
        ("¶¶ 12–14, 18", [(12, 14, "paragraph", None), (18, 18, "paragraph", None)]),
        ("557,\n560", [(557, 557, "page", None), (560, 560, "page", None)]),
        ("557 n. 2", [(557, 557, "page", "2")]),
        ("557 & n. 2", [(557, 557, "page", "2")]),
        ("557 and nn. 2–4", [(557, 557, "page", "2-4")]),
        ("557 fn. 2", [(557, 557, "page", "2")]),
        ("557 fn 2", [(557, 557, "page", "2")]),
        ("557, 559 n. 2", [(557, 557, "page", None), (559, 559, "page", "2")]),
        ("*5, ¶ 12, 15", [(5, 5, "star", None), (12, 12, "paragraph", None), (15, 15, "paragraph", None)]),
    ],
)
def test_pinpoint_targets_preserve_notation_semantics_and_roundtrip(quote, expected):
    field = read(quote)
    assert field.quote == quote
    assert field.node_id == "read-pinpoint"
    assert [
        (target.first, target.last, target.kind.value, target.footnote) for target in field.get_normalized()
    ] == expected
    assert PinCiteField.model_validate_json(field.model_dump_json()) == field


@pytest.mark.parametrize(
    "quote",
    [
        "0",
        "557-550",
        "557–\n559",
        "557 n. 2, 3",
        "557 nn. 2, 3",
        "557 n. 2, 559",
        "557; 559",
        "557 through 559",
        "557 or 559",
        "557-",
        "557,",
        "at",
        "¶¶",
        "557 later prose",
    ],
)
def test_unparseable_or_ambiguous_pinpoints_remain_explicit_failed_readings(quote):
    field = read(quote)
    assert field.quote == quote
    assert not field.normalizable
    assert field.normalization_error
    assert field.model_dump()["normalized"] is None
    assert PinCiteField.model_validate_json(field.model_dump_json()) == field
    with pytest.raises(ValueError, match="not normalizable"):
        field.get_normalized()


def test_source_grounding_rejects_different_text_at_the_recorded_span():
    field = PinCiteField.from_source("at *5–7", Span(start=0, end=7), node_id="read-pinpoint")
    assert field.quote == "at *5–7"
    field.validate_source("at *5–7")
    with pytest.raises(ValueError, match="does not match"):
        field.validate_source("at *6–8")


@pytest.mark.parametrize(
    "changes",
    [
        {"first": 558},
        {"last": 560},
        {"kind": "star"},
        {"footnote": "2"},
    ],
)
def test_forged_normalized_targets_are_rejected(changes):
    field = read("557–59")
    payload = field.model_dump(mode="json")
    payload["normalized"][0].update(changes)
    with pytest.raises(ValidationError, match="does not match its quote"):
        PinCiteField.model_validate(payload)


@pytest.mark.parametrize("span", [Span(start=0, end=0), Span(start=0, end=4)])
def test_source_boundary_rejects_empty_and_out_of_source_spans(span):
    with pytest.raises(ValueError, match="outside the source or empty"):
        PinCiteField.from_source("557", span, node_id="read-pinpoint")


def test_target_requires_positive_ascending_ranges_and_nonempty_footnotes():
    for first, last, footnote in [(0, 1, None), (5, 4, None), (5, 5, "")]:
        with pytest.raises(ValidationError):
            PinCiteTarget(first=first, last=last, kind=PinCiteKind.PAGE, footnote=footnote)


def test_nullable_footnote_is_still_required_in_structured_response_schema():
    schema = PinCiteTarget.model_json_schema()
    assert set(schema["required"]) == {"first", "last", "kind", "footnote"}


def test_failed_reading_cannot_also_store_a_normalized_target():
    payload = read("557").model_dump(mode="json")
    payload.update(normalizable=False, normalization_error="historical normalization failure")
    with pytest.raises(ValidationError, match="needs an error and no value"):
        PinCiteField.model_validate(payload)
