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

from mellea_lrc.model.citations import FullCaseCitation
from mellea_lrc.model.pin_cites import PinCiteKind, PinCitePages, read_pin_cite
from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text


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


def test_a_footnote_pin_names_the_page_the_footnote_is_on() -> None:
    """Rule 3.2(b): the page, then `n.` and the footnote's own number."""
    assert read_pin_cite("657 n.1") == (
        PinCitePages(first=657, last=657, kind=PinCiteKind.PAGE, footnote="1"),
    )
    assert read_pin_cite("570 nn.10-12") == (
        PinCitePages(first=570, last=570, kind=PinCiteKind.PAGE, footnote="10-12"),
    )


def test_a_footnote_marker_survives_the_spacing_a_filing_or_a_converter_leaves() -> None:
    assert read_pin_cite("550 n. 16")[0].footnote == "16"
    assert read_pin_cite("167   n.14")[0].footnote == "14"


def test_several_footnotes_on_one_page_are_one_place() -> None:
    """`570 nn.10, 12` names one page, so the comma inside it does not split it."""
    assert read_pin_cite("570 nn.10, 12") == (
        PinCitePages(first=570, last=570, kind=PinCiteKind.PAGE, footnote="10, 12"),
    )


def test_a_footnote_can_sit_on_a_star_page() -> None:
    """The marker is not a kind, so it rides beside whichever kind is there."""
    assert read_pin_cite("*3 n.1") == (PinCitePages(first=3, last=3, kind=PinCiteKind.STAR, footnote="1"),)


def test_what_cannot_be_read_says_what_it_holds() -> None:
    """`UNREAD` is about this reader: `slip op. at 3` is a proper citation."""
    unread = read_pin_cite("slip op. at 3")
    assert len(unread) == 1
    assert unread[0].kind is PinCiteKind.UNREAD
    assert unread[0].first is None
    assert "slip op. at" in (unread[0].note or "")


def test_an_unread_run_states_no_pages() -> None:
    with pytest.raises(ValueError, match="states no pages"):
        PinCitePages(first=657, last=657, kind=PinCiteKind.UNREAD, note="why")


def test_an_unread_run_must_say_why() -> None:
    with pytest.raises(ValueError, match="what stopped the read"):
        PinCitePages(first=None, last=None, kind=PinCiteKind.UNREAD)


def test_a_run_that_was_read_carries_no_note() -> None:
    with pytest.raises(ValueError, match="carries no note"):
        PinCitePages(first=570, last=570, kind=PinCiteKind.PAGE, note="why")


def test_a_page_run_states_a_first_and_a_last() -> None:
    with pytest.raises(ValueError, match="first and a last"):
        PinCitePages(first=None, last=None, kind=PinCiteKind.PAGE)


def test_a_run_cannot_end_before_it_begins() -> None:
    with pytest.raises(ValueError, match="cannot end at"):
        PinCitePages(first=570, last=544, kind=PinCiteKind.PAGE)


def test_an_extracted_citation_carries_the_pages_it_claims() -> None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text("Bell Atl. Corp. v. Twombly, 550 U.S. 544, 555 -56 (2007).")
    citation = next(c for c in document.citations if isinstance(c.fields, FullCaseCitation))

    assert citation.fields.pin_cite.text == "555 -56"
    assert citation.pin_cite_pages == (PinCitePages(first=555, last=556, kind=PinCiteKind.PAGE),)


def test_the_pages_reach_the_serialized_artifact() -> None:
    """A consumer scoring page claims reads them without parsing the pin cite again."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text("Bell Atl. Corp. v. Twombly, 550 U.S. 544, 555 -56 (2007).")

    payload = document.model_dump(mode="json")

    assert payload["citations"][0]["fields"]["pin_cite"] == {
        "span": {"start": 42, "end": 49},
        "text": "555 -56",
        "pages": [{"first": 555, "last": 556, "kind": "page", "footnote": None, "note": None}],
    }


def test_a_range_that_ends_before_it_begins_is_unread() -> None:
    """`808 F. Supp. 3d 97, 980-814` is written in a court's own order about
    fabricated citations, which is where a page range like that comes from. It
    is a claim this reader cannot turn into pages, and raising would stop a
    document being read at all over one citation in it."""
    pages = read_pin_cite("980-814")

    assert [page.kind for page in pages] == [PinCiteKind.UNREAD]
    assert pages[0].note == "the range 980-814 ends before it begins"
