"""Tests for reading which pages a pin cite claims.

A pin cite is kept as the filing wrote it, so two spellings of one claim --
`247-48` and `247 - 248` -- are different strings. Scoring a page claim on
string equality would charge the second as a miss, and reading only the first
number would not see the second page at all.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.pin_cites import PinCiteKind, PinCitePages, read_pin_cite
from mellea_lrc.extraction import extract_from_plain_text
from mellea_lrc.serialization.extracted_document import serialize_extracted_document


def test_a_single_page_is_one_run() -> None:
    assert read_pin_cite("678") == (PinCitePages(first=678, last=678, kind=PinCiteKind.PAGE),)


def test_a_citation_with_no_pin_cite_claims_no_pages() -> None:
    assert read_pin_cite(None) == ()
    assert read_pin_cite("") == ()


def test_the_bluebook_shorthand_is_expanded() -> None:
    """`588-90` ends at 590: the second number replaces the digits it drops."""
    assert read_pin_cite("588-90") == (PinCitePages(first=588, last=590, kind=PinCiteKind.PAGE),)
    assert read_pin_cite("1053 -54") == (PinCitePages(first=1053, last=1054, kind=PinCiteKind.PAGE),)


def test_a_full_second_number_is_taken_as_written() -> None:
    assert read_pin_cite("3  -  4") == (PinCitePages(first=3, last=4, kind=PinCiteKind.PAGE),)


def test_a_comma_separates_places_that_are_not_continuous() -> None:
    """`1068, 1071 -72` is one page and then two more, not a range of five."""
    assert read_pin_cite("1068, 1071 -72") == (
        PinCitePages(first=1068, last=1068, kind=PinCiteKind.PAGE),
        PinCitePages(first=1071, last=1072, kind=PinCiteKind.PAGE),
    )


def test_a_star_page_is_not_a_printed_page() -> None:
    """`*3` is Westlaw's numbering, so page 3 of the reporter is a different claim."""
    assert read_pin_cite("*3") == (PinCitePages(first=3, last=3, kind=PinCiteKind.STAR),)
    assert read_pin_cite("*2 -3") == (PinCitePages(first=2, last=3, kind=PinCiteKind.STAR),)


def test_a_paragraph_is_not_a_page() -> None:
    assert read_pin_cite("¶¶26-28") == (PinCitePages(first=26, last=28, kind=PinCiteKind.PARAGRAPH),)


def test_a_label_written_once_governs_what_follows_it() -> None:
    assert read_pin_cite("*3, 5") == (
        PinCitePages(first=3, last=3, kind=PinCiteKind.STAR),
        PinCitePages(first=5, last=5, kind=PinCiteKind.STAR),
    )


def test_a_pin_cite_holding_more_than_pages_is_not_read() -> None:
    """`657 n.1` is a page and a footnote on it, and which is the claim is a reading."""
    assert read_pin_cite("657 n.1") == (PinCitePages(first=None, last=None, kind=PinCiteKind.NONCONFORMING),)


def test_a_nonconforming_run_states_no_pages() -> None:
    with pytest.raises(ValueError, match="states no pages"):
        PinCitePages(first=657, last=657, kind=PinCiteKind.NONCONFORMING)


def test_a_page_run_states_a_first_and_a_last() -> None:
    with pytest.raises(ValueError, match="first and a last"):
        PinCitePages(first=None, last=None, kind=PinCiteKind.PAGE)


def test_a_run_cannot_end_before_it_begins() -> None:
    with pytest.raises(ValueError, match="cannot end at"):
        PinCitePages(first=570, last=544, kind=PinCiteKind.PAGE)


def test_an_extracted_citation_carries_the_pages_it_claims() -> None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text("Bell Atl. Corp. v. Twombly, 550 U.S. 544, 555 -56 (2007).")
    citation = next(c for c in document.citations if isinstance(c.citation, FullCaseCitation))

    assert citation.citation.pin_cite == "555 -56"
    assert citation.pin_cite_pages == (PinCitePages(first=555, last=556, kind=PinCiteKind.PAGE),)


def test_the_pages_reach_the_serialized_artifact() -> None:
    """A consumer scoring page claims reads them without parsing the pin cite again."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text("Bell Atl. Corp. v. Twombly, 550 U.S. 544, 555 -56 (2007).")

    payload = serialize_extracted_document(document)

    assert payload["citations"][0]["pin_cite_pages"] == [{"first": 555, "last": 556, "kind": "page"}]
