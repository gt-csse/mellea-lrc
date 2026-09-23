"""Court labels use both external abbreviation data and preserve unresolved readings."""

import pytest

from mellea_lrc.extraction import grow_roots
from mellea_lrc.model import CourtField, Document, Span
from mellea_lrc.model.citations.fields.court import court_id_if_unique


@pytest.mark.parametrize(
    ("written", "court_id"),
    [
        ("M.D. Pa.", "pamd"),
        ("Bankr. M.D. Pa.", "pamb"),
        ("Bankr. S.D. Fla.", "flsb"),
    ],
)
def test_bluebook_state_abbreviation_resolves_courts_db_label(written: str, court_id: str) -> None:
    # Courts-db uses Penn. or Florida here; reporters-db supplies Pa. or Fla.
    field = CourtField.from_source(written, Span(0, len(written)), node_id="court:node:0")

    assert field.quote == written
    assert field.span == Span(0, len(written))
    assert field.normalizable is True
    assert field.get_normalized().id == court_id
    assert CourtField.model_validate_json(field.model_dump_json()) == field


def test_bluebook_fallback_is_used_by_document_extraction() -> None:
    source = "See Case No. 1:24-cv-00123 (M.D. Pa. 2024)."
    document = grow_roots(Document.from_source(source))
    court = document.citations[0].court[-1]

    assert court.quote == "M.D. Pa."
    assert source[court.span.start : court.span.end] == court.quote
    assert court.get_normalized().id == "pamd"
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_bluebook_fallback_does_not_choose_an_ambiguous_court() -> None:
    written = "Ct. App."
    assert court_id_if_unique(written) is None
    field = CourtField.from_source(written, Span(0, len(written)), node_id="court:node:0")

    assert field.quote == written
    assert field.span == Span(0, len(written))
    assert field.normalizable is False
    assert field.normalization_error
    with pytest.raises(ValueError, match="not normalizable"):
        field.get_normalized()
    assert CourtField.model_validate_json(field.model_dump_json()) == field


def test_bluebook_fallback_does_not_override_a_direct_courts_db_match() -> None:
    assert court_id_if_unique("D. La.") == "lad"
