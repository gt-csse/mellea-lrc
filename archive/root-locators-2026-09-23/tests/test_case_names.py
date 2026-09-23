"""Tests for locating a citation's case name in the document.

eyecite reports the parties and not where they are written. Rebuilding the name
by joining `plaintiff` and `defendant` produces a string the document may not
contain -- it holds no `v.`, and for a case with no adverse party it holds only
half the name -- so the name is located between the offsets eyecite does give.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.model.extraction_metadata import Relaxation
from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text


def _names(text: str) -> list[str | None]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL, with_leaves=True)
    out: list[str | None] = []
    for citation in document.citations:
        span = citation.case_name_span
        out.append(None if span is None else text[span.start : span.end])
    return out


def _name(text: str) -> str | None:
    # Eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL, with_leaves=True)
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
    """Both need the case in full first: a leaf reaches a root or is not grown."""
    full = "Bell Atl. Corp. v. Twombly , 550 U.S. 544 (2007). "

    assert _names(full + "Id. at 570.")[1] is None
    assert _names(full + "The court went on. 550 U.S. at 570.")[1] is None


def test_a_span_that_opens_at_the_versus_takes_the_party_in_front_of_it() -> None:
    """eyecite gives up on a party a quotation dash runs into."""
    text = "must reasonably anticipate being haled into court.' -Calder v. Jones , 465 U.S. 783, 789 (1984)."
    assert _name(text) == "Calder v. Jones"


def test_the_party_is_not_taken_from_the_sentence_before_it() -> None:
    """Only a capitalised word touching the `v.` is the missing party."""
    text = "The court so held. v. Jones , 465 U.S. 783, 789 (1984)."
    assert _name(text) == "v. Jones"


def test_a_period_inside_a_name_is_not_a_sentence_boundary() -> None:
    """`N.C.`, `Atl.` and `Inc.` end a word, and cutting there loses the party."""
    assert _name("Bell Atl. Corp. v. Twombly, 550 U.S. 544, 570 (2007).") == "Bell Atl. Corp. v. Twombly"
    text = "Robinson v. N.C. Farm Bureau Ins. Co. , 86 N.C. App. 44, 46 (1987)."
    assert _name(text) == "Robinson v. N.C. Farm Bureau Ins. Co."


def test_a_heading_in_front_of_the_name_is_not_part_of_it() -> None:
    text = "Case Law: Sedima, S.P.R.L. v. Imrex Co. , 473 U.S. 479 (1985)."
    assert _name(text) == "Sedima, S.P.R.L. v. Imrex Co."


def test_the_second_of_two_citations_written_together_keeps_its_own_name() -> None:
    """eyecite opens the second span back at the first citation's name."""
    text = (
        "Ashcroft v. Iqbal, 556 U.S. 662 (2009), and Bell Atlantic Corp. v. Twombly"
        " , 550 U.S. 544 (2007) both apply."
    )
    assert _names(text) == ["Ashcroft v. Iqbal", "Bell Atlantic Corp. v. Twombly"]


def test_a_parallel_citation_still_reaches_back_to_the_shared_name() -> None:
    """One name, two reporters: the second has no name of its own in between."""
    text = "State v. Peters , 231 Kan. 595, 598, 647 P.2d 1288 (1982)."
    assert _names(text) == ["State v. Peters", "State v. Peters"]


def test_the_number_of_a_list_is_not_part_of_the_name() -> None:
    assert _name("1) In Garrett v. Selby Connor , 425 F.3d 836, 840 (10th Cir. 2005).") == (
        "Garrett v. Selby Connor"
    )
    text = "(ERISA). In Womack v. City of Tulsa , 522 P.3d 508, 511 (Okla. 2022)."
    assert _name(text) == "Womack v. City of Tulsa"


def test_a_name_that_is_only_a_corporate_suffix_reaches_back_to_its_party() -> None:
    """eyecite guesses a short form's antecedent from the token in front of it.

    `Service By Air, Inc., supra` comes back as `Inc.` -- a word in every other
    corporate caption, which identifies none of them. The party is written
    right there, so the name is widened back over it.
    """
    text = (
        "Service By Air, Inc. v. Phoenix Cartage & Air Freight, LLC , 78 F. Supp. 3d 852 "
        "(N.D. Ill. 2015). The officer must have participated personally to pierce the "
        "corporate veil. Service By Air, Inc., supra."
    )
    assert _names(text)[-1] == "Service By Air, Inc."


def test_the_sentence_in_front_of_a_suffix_is_not_part_of_the_party() -> None:
    """The walk back stops at the first word a caption would not hold."""
    text = (
        "Drink Group, Inc. v. Gulfstream Communications, Inc. , 7 F. Supp. 2d 1009 "
        "(N.D. Ill. 1998). That is what the district court decided in Drink Group, "
        "Inc., supra."
    )
    assert _names(text)[-1] == "Drink Group, Inc."


def test_widening_over_a_party_does_not_swallow_the_signal_in_front_of_it() -> None:
    text = "Mars, Inc. v. Curtiss Candy Co. , 8 Ill. App. 3d 338 (1972). See Mars, Inc., supra."
    assert _names(text)[-1] == "Mars, Inc."
