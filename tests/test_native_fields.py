"""Typed source-grounded fields retain quotes and checked normalization."""

import pytest
from pydantic import ValidationError

from mellea_lrc.model.citations.fields import (
    CaseName,
    CaseNameField,
    CaseNameKind,
    CitationDate,
    DateField,
    DocketEntryField,
    FullDocketLocator,
)
from mellea_lrc.model.span import Span


def quoted_span(source: str, quote: str) -> Span:
    start = source.index(quote)
    return Span(start=start, end=start + len(quote))


@pytest.mark.parametrize(
    ("quote", "kind", "rendered"),
    [
        ("Alpha\n v.  Beta", CaseNameKind.ADVERSARIAL, "Alpha v. Beta"),
        ("In re Alpha Corporation", CaseNameKind.IN_RE, "In re Alpha Corporation"),
        ("Matter of Alpha Corporation", CaseNameKind.IN_RE, "In re Alpha Corporation"),
        ("Ex parte Alpha", CaseNameKind.EX_PARTE, "Ex parte Alpha"),
        ("Alpha", CaseNameKind.PARTIAL, "Alpha"),
    ],
)
def test_case_name_reading_keeps_quote_and_typed_parts(quote, kind, rendered):
    source = f"Introduction: {quote}, 556 U.S. 662."
    field = CaseNameField.from_source(source, quoted_span(source, quote), node_id="read-name")
    assert field.quote == quote
    assert field.get_normalized().kind is kind
    assert field.get_normalized().as_citation() == rendered
    assert CaseNameField.model_validate_json(field.model_dump_json()) == field
    field.validate_source(source)


@pytest.mark.parametrize("quote", ["Alpha v.", "v. Beta", "Alpha vs. Beta", "In re", "!!!"])
def test_unparseable_case_names_remain_explicit_failed_readings(quote):
    field = CaseNameField.from_source(quote, Span(start=0, end=len(quote)), node_id="read-name")
    assert not field.normalizable
    assert field.normalization_error
    assert field.model_dump()["normalized"] is None
    assert CaseNameField.model_validate_json(field.model_dump_json()) == field
    with pytest.raises(ValueError, match="not normalizable"):
        field.get_normalized()


def test_unstated_name_is_a_typed_value_without_a_quoted_reading():
    name = CaseName(kind=CaseNameKind.NOT_STATED)
    assert name.model_dump()["plaintiff"] is None
    with pytest.raises(ValueError, match="no citation text"):
        name.as_citation()
    with pytest.raises(ValidationError, match="no printed parts"):
        CaseName(kind=CaseNameKind.NOT_STATED, partial="Alpha")
    with pytest.raises(ValidationError, match="cannot be a quoted"):
        CaseNameField.from_model("Alpha", Span(start=0, end=5), name, node_id="review-name")


def test_model_name_normalization_keeps_exact_source_grounding():
    source = "Alpha\n 13\n v. Beta"
    normalized = CaseName(kind=CaseNameKind.ADVERSARIAL, plaintiff="Alpha", defendant="Beta")
    field = CaseNameField.from_model(
        source, Span(start=0, end=len(source)), normalized, node_id="review-name"
    )
    assert field.normalized_by == "model"
    assert field.get_normalized() == normalized
    assert field.quote == source
    with pytest.raises(ValueError, match="does not match"):
        field.validate_source("Gamma\n 13\n v. Beta")


@pytest.mark.parametrize(
    ("quote", "year", "month", "day"),
    [("2024", 2024, None, None), ("Sept. 2024", 2024, 9, None), ("June 3, 2021", 2021, 6, 3)],
)
def test_dates_preserve_declared_precision(quote, year, month, day):
    field = DateField.from_source(quote, Span(start=0, end=len(quote)), node_id="read-date")
    assert field.get_normalized() == CitationDate(year=year, month=month, day=day)
    assert DateField.model_validate_json(field.model_dump_json()) == field


def test_invalid_calendar_date_fails_normalization_instead_of_becoming_absent():
    quote = "February 30, 2024"
    field = DateField.from_source(quote, Span(start=0, end=len(quote)), node_id="read-date")
    assert not field.normalizable
    assert "Invalid citation date" in field.normalization_error
    with pytest.raises(ValueError, match="not normalizable"):
        field.get_normalized()


def test_docket_number_is_preserved_as_an_opaque_source_value():
    source = "See Case No. 1:24-cv-00123 (D. Md.)."
    field = FullDocketLocator.from_source(
        source,
        quoted_span(source, "Case No. 1:24-cv-00123"),
        quoted_span(source, "1:24-cv-00123"),
        node_id="read-docket",
    )
    assert field.get_normalized().docket_number == "1:24-cv-00123"
    assert FullDocketLocator.model_validate_json(field.model_dump_json()) == field
    with pytest.raises(ValueError, match="inside the locator"):
        FullDocketLocator.from_source(source, field.span, Span(start=0, end=3), node_id="read-docket")


@pytest.mark.parametrize(
    ("quote", "entry"), [("ECF No. 128", "128"), ("Doc. 12-3", "12-3"), ("D.I. 52", "52")]
)
def test_optional_docket_entries_are_distinct_from_docket_numbers(quote, entry):
    field = DocketEntryField.from_source(quote, Span(start=0, end=len(quote)), node_id="read-entry")
    assert field.get_normalized() == entry
    assert DocketEntryField.model_validate_json(field.model_dump_json()) == field


def test_field_rejects_forged_normalization_and_inconsistent_failure_storage():
    field = DateField.from_source("2024", Span(start=0, end=4), node_id="read-date")
    data = field.model_dump()
    data["normalized"] = {"year": 2023, "month": None, "day": None}
    with pytest.raises(ValidationError, match="does not match its quote"):
        DateField.model_validate(data)
    data = field.model_dump()
    data["normalizable"] = False
    data["normalization_error"] = "Cannot parse"
    with pytest.raises(ValidationError, match="needs an error and no value"):
        DateField.model_validate(data)


def test_exact_quoting_boundary_rejects_empty_blank_and_out_of_source_spans():
    for span in (Span(start=0, end=0), Span(start=0, end=4), Span(start=1, end=2)):
        with pytest.raises(ValueError):
            CaseNameField.from_source("A  ", span, node_id="read-name")
