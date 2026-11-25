"""Short reporter locators retain their source and normalize their identity."""

import pytest
from pydantic import ValidationError

from mellea_lrc.model.citations.fields import FullReporterLocator, ShortReporterLocator
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.reporters import full_reporter_readings, short_reporter_readings


@pytest.mark.parametrize(
    "quote",
    (
        "347 U.S. at 489",
        "347  U. S.  at\n489",
        "347U.S.at489",
        "347 U.S. at p. 489–90",
        "347 U.S. at 489, 501 n.2",
    ),
)
def test_short_reporter_field_roundtrips_an_exact_quote(quote: str) -> None:
    source = f"See Smith, {quote}."
    (reading,) = short_reporter_readings(source)
    span = Span(*reading.span)

    field = ShortReporterLocator.from_source(source, span, node_id="short:node:0")

    assert field.quote == quote
    assert span.start == source.index(quote)
    assert field.get_normalized().volume == 347
    assert field.get_normalized().reporter.short_name == "U.S."
    assert field.get_normalized().edition == "U.S."
    assert set(type(field.get_normalized()).model_fields) == {"volume", "reporter", "edition"}
    field.validate_source(source)
    assert ShortReporterLocator.model_validate_json(field.model_dump_json()) == field


@pytest.mark.parametrize(
    "following",
    (
        "100 F.3d at 20.",
        "100 F.3d 1 (2000).",
        "42 U.S.C. § 1983.",
        "Id. at 496.",
        "Smith, supra, at 496.",
    ),
)
def test_short_reporter_pinpoint_stops_before_next_citation(following: str) -> None:
    quote = "347 U.S. at 489, 501 n.2"
    source = f"Smith v. Jones, 347 U.S. 483. See Smith, {quote}, {following}"

    readings = short_reporter_readings(source)

    assert [source[slice(*reading.span)] for reading in readings] == (
        [quote, "100 F.3d at 20"] if following.startswith("100 F.3d at") else [quote]
    )
    assert [source[slice(*reading.span)] for reading in full_reporter_readings(source)] == (
        ["347 U.S. 483", "100 F.3d 1"] if following.startswith("100 F.3d 1") else ["347 U.S. 483"]
    )


@pytest.mark.parametrize("quote", ("1 Wash. at 2", "347 U.S. 483", "347 U.S. at 489; compare"))
def test_short_reporter_field_retains_unresolved_quotes(quote: str) -> None:
    field = ShortReporterLocator.from_source(quote, Span(0, len(quote)), node_id="short:node:0")

    assert field.quote == quote
    assert not field.normalizable
    assert field.normalization_error
    with pytest.raises(ValueError, match="not normalizable"):
        field.get_normalized()


def test_full_reporter_field_does_not_accept_short_reporter_quotes() -> None:
    quote = "347 U.S. at 489"
    field = FullReporterLocator.from_source(quote, Span(0, len(quote)), node_id="full:node:0")
    assert not field.normalizable


def test_short_reporter_field_checks_normalization_on_load() -> None:
    quote = "347 U.S. at 489"
    field = ShortReporterLocator.from_source(quote, Span(0, len(quote)), node_id="short:node:0")
    saved = field.model_dump(mode="json")
    saved["normalized"]["volume"] = 348

    with pytest.raises(ValidationError, match="normalization"):
        ShortReporterLocator.model_validate(saved)
