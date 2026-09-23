"""Tests for locating a table of authorities in the exported text."""

from __future__ import annotations

import pytest

pytest.importorskip("docling_core", reason="Docling is an optional preprocessing dependency")

from docling_core.types.doc.document import DoclingDocument, TableCell, TableData  # noqa: E402

from mellea_lrc.model.spans import Span  # noqa: E402
from mellea_lrc.preprocessing.document_index import (  # noqa: E402
    index_table_spans,
    is_within,
)

ENTRY = "Doe v. Megless , 654 F.3d 404 (3d Cir. 2011) ......... 8, 12"
PROSE = "Plaintiff relies on Doe v. Megless , 654 F.3d 404, 408 (3d Cir. 2011)."


def _table(document: DoclingDocument, text: str, label: str) -> None:
    data = TableData(
        num_rows=1,
        num_cols=1,
        table_cells=[
            TableCell(
                text=text,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
            )
        ],
    )
    document.add_table(data=data, label=label)


def test_an_index_table_is_located_in_the_exported_text() -> None:
    """The span must cover the entry, so a citation inside it can be recognised."""
    document = DoclingDocument(name="brief")
    _table(document, ENTRY, "document_index")
    document.add_text(label="text", text=PROSE)

    (span,) = index_table_spans(document)
    text = document.export_to_text()

    assert "654 F.3d 404" in text[span.start : span.end]
    assert PROSE not in text[span.start : span.end]


def test_the_argued_citation_is_outside_the_index() -> None:
    """This is the distinction the module exists to draw.

    The same case appears twice: once listed, once argued from. Only the second
    attaches a proposition to a page, and only the second is worth checking.
    """
    document = DoclingDocument(name="brief")
    _table(document, ENTRY, "document_index")
    document.add_text(label="text", text=PROSE)

    regions = index_table_spans(document)
    text = document.export_to_text()
    argued = text.index("654 F.3d 404", text.index(PROSE))

    assert not is_within(Span(argued, argued + 12), regions)


def test_an_ordinary_table_is_not_an_index() -> None:
    """A brief may hold data tables, and a citation in one is still argued."""
    document = DoclingDocument(name="brief")
    _table(document, ENTRY, "table")
    document.add_text(label="text", text=PROSE)

    assert index_table_spans(document) == ()


def test_a_document_with_no_tables_reports_nothing() -> None:
    """Most filings have no index at all, and that path must stay free."""
    document = DoclingDocument(name="brief")
    document.add_text(label="text", text=PROSE)

    assert index_table_spans(document) == ()


def test_the_document_is_left_as_it_was_found() -> None:
    """Locating the index must not change what the document exports.

    The measurement works by rendering twice, and a restore that failed would
    silently drop the index from the text every consumer downstream reads.
    """
    document = DoclingDocument(name="brief")
    _table(document, ENTRY, "document_index")
    document.add_text(label="text", text=PROSE)
    before = document.export_to_text()

    index_table_spans(document)

    assert document.export_to_text() == before


def test_containment_is_required_rather_than_overlap() -> None:
    """A citation abutting an index is argued text, and must not be discarded."""
    regions = (Span(10, 40),)

    assert is_within(Span(12, 30), regions)
    assert not is_within(Span(35, 55), regions)
    assert not is_within(Span(5, 15), regions)


def test_two_indexes_are_located_separately() -> None:
    """A brief opens with a table of contents and a table of authorities.

    Each is measured against the unmodified rendering, so the span of the second
    is where the reader will find it rather than where it would sit had the
    first been removed.
    """
    document = DoclingDocument(name="brief")
    _table(document, "I. Introduction ......... 1", "document_index")
    _table(document, ENTRY, "document_index")
    document.add_text(label="text", text=PROSE)
    text = document.export_to_text()

    first, second = index_table_spans(document)

    assert "I. Introduction" in text[first.start : first.end]
    assert "654 F.3d 404" in text[second.start : second.end]
    assert first.end <= second.start


def test_an_entry_repeated_in_the_body_does_not_fragment_the_span() -> None:
    """An index entry repeats verbatim in the body -- that is what an index is.

    Comparing the two renderings with a block matcher lets it pair fragments of
    the removed index against those repetitions, which returned an index as 175
    two-character spans rather than as one.
    """
    document = DoclingDocument(name="brief")
    _table(document, ENTRY, "document_index")
    for _ in range(3):
        document.add_text(label="text", text=ENTRY)
    text = document.export_to_text()

    (span,) = index_table_spans(document)

    assert text[span.start : span.end].count("654 F.3d 404") == 1
    assert span.start < text.index(ENTRY, span.end)
