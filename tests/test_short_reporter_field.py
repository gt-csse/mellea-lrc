"""A short reporter quote retains its source and its distinct citation kind."""

import pytest

from mellea_lrc.model.citations.fields.reporter import FullReporterLocator
from mellea_lrc.model.citations.fields.short_reporter import ShortReporterLocator
from mellea_lrc.model.span import Span


@pytest.mark.parametrize(
    ("written", "pin_page"),
    [
        ("347 U.S. at 489", "489"),
        ("347  U. S.  at  489", "489"),
        ("347 U.S. at 489-90", "489"),
    ],
)
def test_short_reporter_field_normalizes_an_exact_quote(written: str, pin_page: str) -> None:
    source = f"See Smith, {written}."
    start = source.index(written)
    span = Span(start, start + len(written))

    field = ShortReporterLocator.from_source(source, span, node_id="short:node:0")

    assert field.quote == written
    assert field.span == span
    assert field.normalizable is True
    assert field.get_normalized().volume == 347
    assert field.get_normalized().reporter.short_name == "U.S."
    assert field.get_normalized().edition == "U.S."
    assert field.get_normalized().pin_page == pin_page
    assert "page" not in type(field.get_normalized()).model_fields
    assert ShortReporterLocator.model_validate_json(field.model_dump_json()) == field


def test_short_and_full_reporter_fields_require_their_own_eyecite_kind() -> None:
    short_quote = "347 U.S. at 489"
    full_quote = "347 U.S. 483"

    short_as_full = FullReporterLocator.from_source(
        short_quote, Span(0, len(short_quote)), node_id="full:node:0"
    )
    full_as_short = ShortReporterLocator.from_source(
        full_quote, Span(0, len(full_quote)), node_id="short:node:0"
    )

    assert short_as_full.normalizable is False
    assert full_as_short.normalizable is False


def test_ambiguous_short_reporter_remains_an_exact_failed_reading() -> None:
    quote = "1 Wash. at 2"
    field = ShortReporterLocator.from_source(quote, Span(0, len(quote)), node_id="short:node:0")

    assert field.quote == quote
    assert field.normalizable is False
    assert "ambiguous short reporter locator" in field.normalization_error
    assert field.model_dump(mode="json")["normalized"] is None
    with pytest.raises(ValueError, match="not normalizable"):
        field.get_normalized()
    assert ShortReporterLocator.model_validate_json(field.model_dump_json()) == field


def test_short_reporter_value_is_checked_against_the_quote_on_load() -> None:
    quote = "347 U.S. at 489"
    field = ShortReporterLocator.from_source(quote, Span(0, len(quote)), node_id="short:node:0")
    saved = field.model_dump(mode="json")
    saved["normalized"]["pin_page"] = "490"

    with pytest.raises(ValueError, match="normalization"):
        ShortReporterLocator.model_validate(saved)
