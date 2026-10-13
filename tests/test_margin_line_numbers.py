"""Tests for reclassifying pleading-paper margin line numbers as furniture."""

from __future__ import annotations

import pytest

pytest.importorskip("docling_core", reason="Docling is an optional preprocessing dependency")

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument

from mellea_lrc.preprocessing.margin_line_numbers import (
    margin_line_numbers,
    reclassify_margin_line_numbers,
)

# The real geometry, read off document 022 of false-citation-bench.
GUTTER_LEFT, GUTTER_RIGHT = 47.5, 60.0
PROSE_LEFT, PROSE_RIGHT = 72.0, 543.4


def _document() -> DoclingDocument:
    return DoclingDocument(name="filing")


def _add(
    document: DoclingDocument,
    text: str,
    *,
    left: float,
    right: float,
    page: int = 1,
    label: str = "text",
) -> object:
    return document.add_text(
        label=label,
        text=text,
        prov=ProvenanceItem(
            page_no=page,
            bbox=BoundingBox(l=left, t=100.0, r=right, b=90.0),
            charspan=(0, len(text)),
        ),
    )


def _pleading_page(document: DoclingDocument, *prose: str, page: int = 1, lines: int = 28) -> None:
    """One page of pleading paper: a numbered left margin, then the prose."""
    for number in range(1, lines + 1):
        _add(document, str(number), left=GUTTER_LEFT, right=GUTTER_RIGHT, page=page)
    for line in prose:
        _add(document, line, left=PROSE_LEFT, right=PROSE_RIGHT, page=page)


def test_a_citation_broken_across_a_page_is_made_whole() -> None:
    """The case this exists for: the margin of page 8 fell inside the citation.

    Page 7 ends mid-sentence at `214 F.3d`; page 8 opens with its own line
    numbers and only then continues `1058`. Nothing about the citation was
    damaged -- it was interrupted by material that is not part of the text.
    """
    document = _document()
    _pleading_page(document, "decision in Advanced Textile , 214 F.3d", page=7)
    _pleading_page(document, "1058 (9th Cir. 2000), as restricting", page=8)

    reclassify_margin_line_numbers(document)

    assert document.export_to_text() == (
        "decision in Advanced Textile , 214 F.3d\n\n1058 (9th Cir. 2000), as restricting"
    )


def test_every_margin_number_is_moved_and_nothing_else_is() -> None:
    """Two pages of pleading paper is fifty-six numbers and no lost prose."""
    document = _document()
    _pleading_page(document, "first page prose", page=1)
    _pleading_page(document, "second page prose", page=2)

    reclassify_margin_line_numbers(document)

    furniture = [item for item, _ in document.iterate_items(included_content_layers={ContentLayer.FURNITURE})]
    assert len(furniture) == 56
    assert document.export_to_text() == "first page prose\n\nsecond page prose"


def test_a_numbered_list_in_the_body_is_not_a_margin() -> None:
    """Position is half the test, and it is the half prose cannot fake.

    A list item carries a bare integer just as a margin number does. It sits in
    the text column, so it stays.
    """
    document = _document()
    for number in range(1, 9):
        _add(document, str(number), left=PROSE_LEFT, right=PROSE_RIGHT, label="list_item")

    assert margin_line_numbers(document) == []


def _stacked(
    document: DoclingDocument,
    values: list[int],
    *,
    left: float,
    right: float,
    origin: CoordOrigin = CoordOrigin.BOTTOMLEFT,
) -> None:
    """A column of numbers down the page, each lower than the one above it."""
    for index, number in enumerate(values):
        top = 100.0 + index * 24.0 if origin is CoordOrigin.TOPLEFT else 740.0 - index * 24.0
        document.add_text(
            label="text",
            text=str(number),
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(
                    l=left,
                    t=top,
                    r=right,
                    b=top + 10.0 if origin is CoordOrigin.TOPLEFT else top - 10.0,
                    coord_origin=origin,
                ),
                charspan=(0, len(str(number))),
            ),
        )


@pytest.mark.parametrize("left,right", [(72.0, 90.0), (23.3, 38.4)])
def test_split_body_list_markers_are_preserved(left: float, right: float) -> None:
    document = _document()
    for number in range(1, 7):
        _add(document, str(number), left=left, right=right, label="list_item")
        _add(document, "A substantive numbered body paragraph.", left=100.0, right=PROSE_RIGHT)
    before = document.export_to_text()

    assert margin_line_numbers(document) == []
    reclassify_margin_line_numbers(document)
    assert document.export_to_text() == before


def test_counting_body_column_left_of_indented_prose_is_preserved() -> None:
    document = _document()
    _stacked(document, [1, 2, 3, 4, 5, 6], left=72.0, right=90.0)
    _add(document, "Body prose indented beyond the number column.", left=100.0, right=PROSE_RIGHT)

    assert margin_line_numbers(document) == []


def test_noncounting_column_left_of_prose_is_preserved() -> None:
    document = _document()
    _stacked(document, [40, 12, 12, 8, 3, 1], left=GUTTER_LEFT, right=GUTTER_RIGHT)
    _add(document, "Amount claimed per quarter, in thousands.", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert margin_line_numbers(document) == []


@pytest.mark.parametrize("origin", [CoordOrigin.TOPLEFT, CoordOrigin.BOTTOMLEFT])
def test_margin_counting_respects_coordinate_origin(origin: CoordOrigin) -> None:
    document = _document()
    _stacked(document, [1, 2, 3, 4, 5, 6], left=GUTTER_LEFT, right=GUTTER_RIGHT, origin=origin)
    _add(document, "Body prose.", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert len(margin_line_numbers(document)) == 6


def test_a_margin_is_found_when_the_prose_starts_in_it_too() -> None:
    """Some filings number the right margin, and Docling merges most of them.

    What is left is a column at the page edge, and the prose items Docling
    built around the numbers it absorbed begin at that same edge -- so the
    column is not *left* of the prose and the position test cannot separate
    them. The numbers still count down the page, which prose does not.
    """
    document = _document()
    _stacked(document, [1, 2, 3, 10, 11, 12, 13, 14], left=23.3, right=38.4)
    _add(document, "fix evidentiary failures. In re Motors Liquidation Co., 23", left=23.3, right=558.0)
    _add(document, "361-62 (2d Cir. 2020) ('[A] reply brief cannot 24", left=23.3, right=558.0)

    assert len(margin_line_numbers(document)) == 8


def test_a_column_at_the_page_edge_that_does_not_count_is_not_a_margin() -> None:
    """Line numbers rise. A column of quantities repeats and falls."""
    document = _document()
    _stacked(document, [40, 12, 12, 8, 3, 1], left=23.3, right=38.4)
    _add(document, "Amount claimed per quarter, in thousands", left=23.3, right=558.0)

    assert margin_line_numbers(document) == []


def test_a_counting_column_inside_the_text_is_not_a_margin() -> None:
    """A numbered list counts too, and it is not at the edge of the page."""
    document = _document()
    _stacked(document, [1, 2, 3, 4, 5, 6], left=PROSE_LEFT, right=PROSE_RIGHT)
    _add(document, "prose beside it", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert margin_line_numbers(document) == []


def test_prose_in_a_narrow_left_column_is_not_a_margin() -> None:
    """Being numeric is the other half, and it is the half geometry cannot fake."""
    document = _document()
    _add(document, "Plaintiff", left=GUTTER_LEFT, right=GUTTER_RIGHT)
    _add(document, "body text", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert margin_line_numbers(document) == []


def test_a_reporter_page_alone_on_a_line_is_never_a_margin_number() -> None:
    """`1058` is the value the whole exercise is trying to protect."""
    document = _document()
    _add(document, "1058", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert margin_line_numbers(document) == []
    assert "1058" in document.export_to_text()


def test_a_three_digit_number_in_the_margin_is_left_alone() -> None:
    """Margins count lines on a page; they do not reach three digits."""
    document = _document()
    _add(document, "404", left=GUTTER_LEFT, right=GUTTER_RIGHT)
    _add(document, "body text", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert margin_line_numbers(document) == []


def test_the_dashed_margin_style_is_recognised() -> None:
    """Some of these filings write the margin as `- 4 -` rather than `4`."""
    document = _document()
    for number in range(1, 9):
        _add(document, f"- {number} -", left=GUTTER_LEFT, right=GUTTER_RIGHT)
    _add(document, "body text", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert len(margin_line_numbers(document)) == 8


def test_a_lone_number_in_the_margin_is_not_a_column() -> None:
    """One number is a stray, not a margin, and removing it could lose content."""
    document = _document()
    _add(document, "4", left=GUTTER_LEFT, right=GUTTER_RIGHT)
    _add(document, "body text", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert margin_line_numbers(document) == []


def test_an_absorbed_line_number_does_not_defeat_the_page() -> None:
    """Docling sometimes glues the first line numbers onto the text beside them.

    Those items do not shift the page edge or break the surviving counting
    column. This is the geometry observed in document 011, page 1.
    """
    document = _document()
    for absorbed in ("1 JULIE A. TOTTEN (STATE BAR NO. 166470)", "3 ORRICK, HERRINGTON LLP"):
        _add(document, absorbed, left=GUTTER_LEFT, right=PROSE_RIGHT)
    for line in ("The Orrick Building", "San Francisco, CA", "Attorneys for Defendant"):
        _add(document, line, left=PROSE_LEFT, right=PROSE_RIGHT)
    for number in range(6, 29):
        _add(document, str(number), left=GUTTER_LEFT, right=GUTTER_RIGHT)

    assert len(margin_line_numbers(document)) == 23


def test_a_page_of_nothing_but_numbers_is_left_alone() -> None:
    """A page with no prose does not establish a line-number margin.

    Declining is the safe direction: leaving a real margin in costs recall,
    while removing real content costs a citation outright.
    """
    document = _document()
    for number in range(1, 29):
        _add(document, str(number), left=GUTTER_LEFT, right=GUTTER_RIGHT)

    assert margin_line_numbers(document) == []


def test_running_the_rule_twice_changes_nothing_the_second_time() -> None:
    document = _document()
    _pleading_page(document, "prose")

    reclassify_margin_line_numbers(document)
    once = document.export_to_text()
    reclassify_margin_line_numbers(document)

    assert document.export_to_text() == once == "prose"


def test_an_item_without_provenance_is_left_alone() -> None:
    """Not every backend records geometry; without it this rule cannot judge."""
    document = _document()
    document.add_text(label="text", text="7")
    _add(document, "body text", left=PROSE_LEFT, right=PROSE_RIGHT)

    assert margin_line_numbers(document) == []
    assert document.export_to_text() == "7\n\nbody text"


def test_a_document_with_no_text_layer_has_no_margin() -> None:
    """The rule runs on every conversion, so an unrecognised shape must not raise.

    A stub document, or a Docling version exposing its items differently, has
    no margin to find. Reaching straight for the attribute stopped preprocessing
    outright rather than reporting that nothing was removed.
    """

    class Bare:
        pass

    reclassify_margin_line_numbers(Bare())
