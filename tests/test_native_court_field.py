"""Court fields preserve written evidence and canonical database identities."""

import pytest
from pydantic import ValidationError

from mellea_lrc.model.citations.fields import court as court_fields
from mellea_lrc.model.citations.fields.court import Court, CourtField
from mellea_lrc.model.span import Span


@pytest.mark.parametrize(
    ("quote", "court_id"),
    [
        ("9th Cir.", "ca9"),
        ("2 d\nCir.", "ca2"),
        ("D. Md.", "mdd"),
        ("D. Maryland", "mdd"),
        ("S.D.N.Y.", "nysd"),
        ("E.D. Pa.", "paed"),
        ("D. Mass.", "mad"),
        ("D. Mont.", "mtd"),
        ("N.D. W. Va.", "wvnd"),
        ("Bankr. S.D.N.Y.", "nysb"),
        ("Cal.", "cal"),
        ("N.Y. App. Div.", "nyappdiv"),
        ("1st Dept.", "nyappdiv"),
        ("2d Dep’t", "nyappdiv"),
        ("3d Department", "nyappdiv"),
        ("4th Dept.", "nyappdiv"),
    ],
)
def test_written_court_field_retains_quote_and_canonical_identity(quote, court_id):
    source = f"Alpha v. Beta, 42 F.4th 10 ({quote} 2024)."
    start = source.index(quote)
    span = Span(start=start, end=start + len(quote))
    field = CourtField.from_source(source, span, node_id="read-court")

    assert field.quote == quote
    assert field.span == span
    assert field.node_id == "read-court"
    assert field.get_normalized() == Court.from_id(court_id)
    assert field.get_normalized().name
    assert CourtField.model_validate_json(field.model_dump_json()) == field
    field.validate_source(source)


@pytest.mark.parametrize("quote", ["N.Y.", "Mars Supreme Court", "D.", "9th Cir. or 2d Cir."])
def test_unresolved_court_reading_keeps_evidence_and_explicit_failure(quote):
    field = CourtField.from_source(quote, Span(start=0, end=len(quote)), node_id="read-court")

    assert field.quote == quote
    assert not field.normalizable
    assert field.model_dump()["normalized"] is None
    assert field.normalization_error
    assert CourtField.model_validate_json(field.model_dump_json()) == field
    with pytest.raises(ValueError, match="not normalizable"):
        field.get_normalized()


def test_ambiguous_direct_label_is_not_overridden_by_a_generated_alias(monkeypatch):
    # A future database revision may give one label to multiple courts. A
    # generated abbreviation must not turn that uncertainty into a selection.
    monkeypatch.setattr(court_fields, "_court_index", lambda: {("d", "md"): frozenset({"mdd", "mad"})})
    monkeypatch.setattr(court_fields, "_bluebook_court_index", lambda: {("d", "md"): frozenset({"mdd"})})
    quote = "D. Md."
    field = CourtField.from_source(quote, Span(start=0, end=len(quote)), node_id="read-court")

    assert court_fields.court_id_if_unique(quote) is None
    assert not field.normalizable
    assert field.quote == quote


def test_inferred_court_has_a_canonical_identity_without_inventing_source_text():
    field = CourtField.inferred("scotus", node_id="infer-court")

    assert field.node_id == "infer-court"
    assert field.get_normalized() == Court.from_id("scotus")
    assert field.quote is None
    assert field.span is None
    assert CourtField.model_validate_json(field.model_dump_json()) == field


def test_unknown_inferred_court_id_is_rejected():
    with pytest.raises(ValueError, match="Unknown normalized court ID"):
        CourtField.inferred("not-a-court-id", node_id="infer-court")


def test_written_court_rechecks_exact_source_and_normalized_identity():
    quote = "9th Cir."
    field = CourtField.from_source(quote, Span(start=0, end=len(quote)), node_id="read-court")
    with pytest.raises(ValueError, match="does not match its source span"):
        field.validate_source("2nd Cir.")

    stored = field.model_dump()
    stored["normalized"] = Court.from_id("ca2").model_dump()
    with pytest.raises(ValidationError, match="does not match its quote"):
        CourtField.model_validate(stored)


def test_inferred_court_rejects_a_name_that_does_not_belong_to_its_id():
    stored = CourtField.inferred("scotus", node_id="infer-court").model_dump()
    stored["normalized"]["name"] = "A different court"
    with pytest.raises(ValidationError, match="does not match its court ID"):
        CourtField.model_validate(stored)


def test_inferred_court_cannot_be_an_empty_or_failed_reading():
    with pytest.raises(ValidationError, match="needs a normalized court ID"):
        CourtField(node_id="infer-court", normalizable=False, normalized=None, normalization_error="Unknown")


@pytest.mark.parametrize("span", [Span(start=0, end=0), Span(start=0, end=99)])
def test_written_court_requires_a_nonempty_in_source_span(span):
    with pytest.raises(ValueError):
        CourtField.from_source("9th Cir.", span, node_id="read-court")
