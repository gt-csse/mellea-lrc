"""Tests for pointing at the pin cite, not merely reporting it.

A pin cite is scored on its own, because the page is the one part of a citation
a retrieval cannot settle: finding the case confirms the name, the court and the
year, and says nothing about whether the proposition is on page 570. Scoring it
needs a span, and eyecite supplies one for a single citation kind out of six.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from mellea_lrc.core.citations import CitationKind
from mellea_lrc.extraction import extract_from_plain_text
from mellea_lrc.extraction.types import ExtractedCitation


def _citations(text: str) -> tuple[ExtractedCitation, ...]:
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text).citations


def _of_kind(text: str, kind: CitationKind) -> ExtractedCitation:
    return next(c for c in _citations(text) if c.citation.kind is kind)


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        # The pin cite follows the locator, outside it.
        ("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).", CitationKind.FULL_CASE, "678"),
        ("See 28 U.S.C. § 636(b)(1)(A).", CitationKind.FULL_LAW, "(b)(1)(A)"),
        # The pin cite is the tail of the locator: a short form's page is part
        # of its identifier, so there is nothing after it to point at. The `at`
        # a short form writes is not in the span -- every kind reports the page
        # alone, so the four spellings eyecite returns become one.
        (
            "Iqbal, 556 U.S. 662 (2009). See 556 U.S. at 678.",
            CitationKind.SHORT_CASE,
            "678",
        ),
        (
            "Iqbal, 556 U.S. 662 (2009). Id. at 679.",
            CitationKind.ID,
            "679",
        ),
        (
            "Smith v. Jones, 5 F.3d 1 (1993). Later, Smith, supra, at 15, applies.",
            CitationKind.SUPRA,
            "15",
        ),
    ],
)
def test_the_span_holds_the_pin_cite_the_citation_states(
    text: str, kind: CitationKind, expected: str
) -> None:
    citation = _of_kind(text, kind)

    assert citation.pin_cite_span is not None
    assert text[citation.pin_cite_span.start : citation.pin_cite_span.end] == expected
    assert citation.citation.pin_cite == expected


def test_the_span_survives_the_whitespace_the_relaxation_forgives() -> None:
    """The point of relaxing the pin cite pattern is a span that still lands.

    `544,  570` is what justified text leaves behind. Reading the page and then
    pointing at the wrong characters would be worse than not reading it.
    """
    text = "Bell Atl. Corp. v. Twombly,  550  U.S.  544,  570  (2007)."
    citation = _of_kind(text, CitationKind.FULL_CASE)

    assert citation.pin_cite_span is not None
    assert text[citation.pin_cite_span.start : citation.pin_cite_span.end] == "570"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Iqbal, 556 U.S. 662 (2009). Id. at 679.", CitationKind.ID),
        ("Foo v. Bar, 1 U.S. 2 (1999). We rely on Bar , at 7.", CitationKind.REFERENCE),
        ("Iqbal, 556 U.S. 662 (2009). See 556 U.S. at 678.", CitationKind.SHORT_CASE),
        ("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).", CitationKind.FULL_CASE),
    ],
)
def test_every_kind_spells_the_page_the_same_way(text: str, kind: CitationKind) -> None:
    """One page, one spelling, whatever joined it to the citation.

    eyecite returns `678`, `at 679` and `, at 7` for the same kind of claim,
    because full citations are cleaned in `helpers.py`, short forms keep the
    `at` they are written with, and references are built straight from
    `match.groupdict()` with nothing cleaned at all. A scored column asks which
    page, so the connector is removed and the label -- `¶`, `n.`, `*` -- is not.
    """
    citation = _of_kind(text, kind)

    assert citation.citation.pin_cite is not None
    assert citation.citation.pin_cite[0].isdigit()
    assert citation.pin_cite_span is not None
    start, end = citation.pin_cite_span.start, citation.pin_cite_span.end
    assert text[start:end] == citation.citation.pin_cite


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        # A star page in an unreported opinion: `at *3` keeps its star.
        (
            "Doe v. Roe, 2006 WL 1984362, at *3 (D. Ariz. July 13, 2006).",
            CitationKind.FULL_CASE,
            "*3",
        ),
        # Paragraphs, as a decision numbered by paragraph is pinpointed.
        ("Doe v. Roe, 5 F.3d 1 (1993). Id. ¶¶ 26-28.", CitationKind.ID, "¶¶ 26-28"),
    ],
)
def test_a_label_is_not_a_connector_and_stays(text: str, kind: CitationKind, expected: str) -> None:
    """`*3` is not page 3 and `¶ 26` is not page 26.

    A connector says how the page was joined to the citation and carries nothing
    about which page it is. A label says what is being pointed at. Stripping one
    normalises a spelling; stripping the other asserts what the filing did not.
    """
    citation = _of_kind(text, kind)

    assert citation.citation.pin_cite == expected
    assert citation.pin_cite_span is not None
    assert text[citation.pin_cite_span.start : citation.pin_cite_span.end] == expected


def test_a_citation_stating_no_page_has_no_span() -> None:
    """Absence is an answer. A pin cite nobody wrote must not be pointed at."""
    citation = _of_kind("Marbury v. Madison, 5 U.S. 137 (1803).", CitationKind.FULL_CASE)

    assert citation.citation.pin_cite is None
    assert citation.pin_cite_span is None


def test_the_span_lies_inside_the_citation_it_belongs_to() -> None:
    """A pin cite outside its own citation's extent belongs to another one.

    The containment is what keeps a parallel citation's page from being read as
    this one's: `390 U.S. 727, 731, 88 S.Ct. 1323` states one page, and the
    second volume number is a citation of its own.
    """
    text = "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323 (1968)."
    for citation in _citations(text):
        if citation.pin_cite_span is None:
            continue
        assert citation.full_span.start <= citation.pin_cite_span.start
        assert citation.pin_cite_span.end <= citation.full_span.end
