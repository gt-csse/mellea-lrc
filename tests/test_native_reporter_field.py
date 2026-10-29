"""Full reporter fields keep exact source evidence and resolved edition identity."""

import pytest
from eyecite.models import Reporter
from pydantic import ValidationError

from mellea_lrc.model.citations.fields.reporter import FullReporterLocator
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.reporters import full_reporter_readings


@pytest.mark.parametrize(
    ("written", "volume", "edition", "page"),
    [
        ("347 U.S. 483", 347, "U.S.", "483"),
        ("347 U. S. 483", 347, "U.S.", "483"),
        ("347 U.\n\nS. 483", 347, "U.S.", "483"),
        ("937\n\nS.W.2d  796", 937, "S.W.2d", "796"),
        ("539  F.  App'x  937", 539, "F. App'x", "937"),
        ("777 F. App ' x 516", 777, "F. App'x", "516"),
        ("58 N.Y .2d 916", 58, "N.Y.2d", "916"),
        ("2005  WL  465431", 2005, "WL", "465431"),
        ("347U.S.483", 347, "U.S.", "483"),
    ],
)
def test_full_reporter_locator_retains_exact_quote_and_typed_value(written, volume, edition, page):
    source = f"See {written}."
    span = Span(start=4, end=4 + len(written))
    field = FullReporterLocator.from_source(source, span, node_id="read-reporter")

    assert field.quote == written
    assert field.span == span
    assert field.node_id == "read-reporter"
    value = field.get_normalized()
    assert value.volume == volume
    assert isinstance(value.reporter, Reporter)
    assert value.reporter.short_name
    assert value.edition == edition
    assert value.page == page
    assert FullReporterLocator.model_validate_json(field.model_dump_json()) == field
    field.validate_source(source)


def test_reader_retains_distinct_offsets_for_repeated_locator_spellings():
    source = "See 2005  WL  465431. Later, 2005\nWL\n465431."
    readings = full_reporter_readings(source)

    assert [source[start:end] for start, end in (item.span for item in readings)] == [
        "2005  WL  465431",
        "2005\nWL\n465431",
    ]
    assert len({item.span for item in readings}) == 2


def test_ambiguous_reporter_is_a_failed_reading_instead_of_a_raw_normalization():
    written = "1 Wash. 2"
    field = FullReporterLocator.from_source(written, Span(start=0, end=len(written)), node_id="read-reporter")

    assert field.quote == written
    assert not field.normalizable
    assert "ambiguous reporter" in field.normalization_error
    assert field.model_dump(mode="json")["normalized"] is None
    with pytest.raises(ValueError, match="not normalizable"):
        field.get_normalized()
    assert FullReporterLocator.model_validate_json(field.model_dump_json()) == field


@pytest.mark.parametrize(
    "written",
    [
        "347 U.S. at 489",
        "347 U.S. 483, 489",
        "See 347 U.S. 483",
        "347 U.S. 483 and 556 U.S. 662",
        "347 Unknown 483",
        "not a locator",
    ],
)
def test_normalization_requires_one_complete_full_locator_quote(written):
    field = FullReporterLocator.from_source(written, Span(start=0, end=len(written)), node_id="read-reporter")
    assert not field.normalizable
    assert field.quote == written
    assert field.normalization_error
    assert FullReporterLocator.model_validate_json(field.model_dump_json()) == field


@pytest.mark.parametrize("component", ["volume", "edition", "page", "reporter"])
def test_serialized_normalized_components_must_match_source_quote(component):
    written = "347 U.S. 483"
    field = FullReporterLocator.from_source(written, Span(start=0, end=len(written)), node_id="read-reporter")
    stored = field.model_dump(mode="json")
    changes = {"volume": 348, "edition": "F.2d", "page": "484"}
    if component == "reporter":
        stored["normalized"]["reporter"]["short_name"] = "F."
    else:
        stored["normalized"][component] = changes[component]
    with pytest.raises(ValidationError, match="normalization"):
        FullReporterLocator.model_validate(stored)


def test_loaded_locator_rechecks_exact_source_grounding():
    source = "See 347 U.S. 483."
    field = FullReporterLocator.from_source(source, Span(start=4, end=16), node_id="read-reporter")
    with pytest.raises(ValueError, match="does not match its source span"):
        field.validate_source("See 347 U.S. 484.")


def test_locator_quoting_boundary_rejects_empty_blank_and_out_of_source_spans():
    for source, span in [
        ("347 U.S. 483", Span(start=0, end=0)),
        (" ", Span(start=0, end=1)),
        ("347 U.S. 483", Span(start=0, end=30)),
    ]:
        with pytest.raises(ValueError):
            FullReporterLocator.from_source(source, span, node_id="read-reporter")
