"""Tests for bounding the scan that reads a citation's court and date.

A missing year costs a check. A year belonging to a different case buys a
confident verdict about the wrong opinion, which is the failure this project
exists to prevent, so these guard the boundary in both directions: it must stop
at an unrelated citation and it must not stop at a parallel one.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text


def _cases(text: str) -> dict[str, FullCaseCitation]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    return {
        " ".join(item.matched_text.split()): item.citation
        for item in document.citations
        if isinstance(item.citation, FullCaseCitation)
    }


def test_a_citation_does_not_take_the_year_of_the_next_one() -> None:
    """The shape from document 009, which read 1994 off the citation after it."""
    cases = _cases(
        "Koulkina, 2009 WL 2103627, at *3. Second, the mailings satisfy the rule. "
        "Spector v. Torenberg, 852 F. Supp. 201, 205 (S.D.N.Y. 1994)."
    )

    assert cases["2009 WL 2103627"].date is None
    assert cases["2009 WL 2103627"].court is None
    assert cases["852 F. Supp. 201"].date.year == "1994"


def test_a_parallel_citation_still_reaches_its_year() -> None:
    """One decision in three reporters, and the date sits after the last of them."""
    cases = _cases("St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968).")

    assert {citation.date.year for citation in cases.values()} == {"1968"}


def test_successive_decisions_do_not_share_a_year() -> None:
    """Brown I and Brown II are two decisions, and only one of them is 1955."""
    cases = _cases("See Brown, 347 U.S. 483, 349 U.S. 294 (1955).")

    assert cases["347 U.S. 483"].date is None
    assert cases["349 U.S. 294"].date.year == "1955"


def test_a_trial_decision_does_not_take_the_year_of_its_appeal() -> None:
    """`aff'd` introduces a different decision, with its own later date."""
    cases = _cases("Doe v. Roe, 100 F.3d 1, aff'd, 200 F.3d 2 (2d Cir. 1999).")

    assert cases["100 F.3d 1"].date is None
    assert cases["200 F.3d 2"].date.year == "1999"


def test_an_ordinary_citation_keeps_its_year_and_court() -> None:
    cases = _cases("Doe v. Megless, 654 F.3d 404, 408 (3d Cir. 2011).")

    assert cases["654 F.3d 404"].date.year == "2011"
    assert cases["654 F.3d 404"].court == "ca3"


def test_a_full_date_is_kept_whole() -> None:
    """58 citations on the bench state one, and they were being cut down to a year."""
    cases = _cases("Doe v. Roe, 2024 WL 4634082, at *3 (D. Ariz. Oct. 31, 2024).")
    date = cases["2024 WL 4634082"].date

    assert (date.year, date.month, date.day) == ("2024", "Oct.", "31")
    assert date.is_exact


def test_a_supreme_court_citation_keeps_the_court_its_reporter_implies() -> None:
    """`scotus` is set from the reporter, not the parenthetical, so it is not the scan's to remove."""
    cases = _cases("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).")

    assert cases["556 U.S. 662"].court == "scotus"


def test_the_span_no_longer_covers_the_following_citation() -> None:
    text = (
        "Koulkina, 2009 WL 2103627, at *3. Second, the mailings satisfy the rule. "
        "Spector v. Torenberg, 852 F. Supp. 201, 205 (S.D.N.Y. 1994)."
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    first = next(item for item in document.citations if item.matched_text == "2009 WL 2103627")

    assert "Spector" not in text[first.span.start : first.span.end]
