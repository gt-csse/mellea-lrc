"""New York department labels retain their text while sharing one court ID."""

import pytest

from mellea_lrc.extraction import grow_roots
from mellea_lrc.model import CourtField, Document, Span
from mellea_lrc.model.citations.fields.court import court_id_if_unique


@pytest.mark.parametrize(
    "written",
    [
        "1st Dept",
        "1st Dep't",
        "2d Dept.",
        "2nd Dep’t",
        "3 rd  Dept.",
        "3d Dep't",
        "4th Department",
    ],
)
def test_new_york_department_label_collapses_to_appellate_division(written: str) -> None:
    field = CourtField.from_source(written, Span(0, len(written)), node_id="court:node:0")

    assert field.quote == written
    assert field.span == Span(0, len(written))
    assert field.get_normalized().id == "nyappdiv"
    assert CourtField.model_validate_json(field.model_dump_json()) == field


@pytest.mark.parametrize("written", ["5th Dept.", "2d Dept. of Transportation", "Dept. of State"])
def test_unrelated_department_text_is_not_a_new_york_appellate_court(written: str) -> None:
    assert court_id_if_unique(written) is None


def test_extraction_preserves_a_department_quote_and_normalizes_its_court() -> None:
    source = "See 139 A.D.3d 695 (2d Dept. 2016)."
    document = grow_roots(Document.from_source(source))
    court = document.citations[0].court[-1]

    assert court.quote == "2d Dept."
    assert source[court.span.start : court.span.end] == court.quote
    assert court.get_normalized().id == "nyappdiv"
    assert Document.model_validate_json(document.model_dump_json()) == document
