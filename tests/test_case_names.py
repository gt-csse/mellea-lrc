"""Tests for locating a citation's case name in the document.

eyecite reports the parties and not where they are written. Rebuilding the name
by joining `plaintiff` and `defendant` produces a string the document may not
contain -- it holds no `v.`, and for a case with no adverse party it holds only
half the name -- so the name is located between the offsets eyecite does give.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.extraction import Relaxation, extract_from_plain_text


def _name(text: str) -> str | None:
    # Eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    citation = document.citations[0]
    span = citation.case_name_span
    return None if span is None else text[span.start : span.end]


def test_the_name_is_the_characters_the_filing_wrote() -> None:
    assert _name("Ashcroft v. Iqbal , 556 U.S. 662, 678 (2009).") == "Ashcroft v. Iqbal"


def test_a_case_with_no_adverse_party_keeps_its_opening_words() -> None:
    """eyecite parses `In re Giftcraft Ltd.` to `defendant='Giftcraft Ltd.'`."""
    assert _name("In re Giftcraft Ltd. , 2025 Bankr. LEXIS 1350 (Bankr. S.D.N.Y. 2025).") == (
        "In re Giftcraft Ltd."
    )
    assert _name("Ex parte Young , 209 U.S. 123, 155 (1908).") == "Ex parte Young"


def test_an_abbreviation_keeps_the_period_that_belongs_to_it() -> None:
    assert _name("Rogers v. Louisville Land Co. , 367 S.W.3d 196, 211 (Tenn. 2012).") == (
        "Rogers v. Louisville Land Co."
    )


def test_a_signal_in_front_of_the_name_is_not_part_of_it() -> None:
    assert _name("See also White v. McBride , 937 S.W.2d 796, 800 (Tenn. 1996).") == ("White v. McBride")
    assert _name("In Day v. Woodworth , 13 How. 363, 371 (1851).") == "Day v. Woodworth"


def test_a_docket_number_before_the_reporter_is_not_part_of_the_name() -> None:
    text = "Turner v. Murphy Oil USA, Inc., No. 05-4206, 2006 WL 1984362, at *1 (E.D. La. 2006)."
    assert _name(text) == "Turner v. Murphy Oil USA, Inc."


def test_a_party_keeps_its_own_comma_and_digits() -> None:
    text = "United States v. Approximately 127,271 Bitcoin , 2025 WL 1234567 (E.D.N.Y. 2025)."
    assert _name(text) == "United States v. Approximately 127,271 Bitcoin"


def test_a_citation_with_no_name_has_no_span() -> None:
    assert _name("Id. at 570.") is None
    assert _name("550 U.S. at 570.") is None
