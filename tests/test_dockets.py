"""Tests for reading a docket number as a citation in its own right.

A docket number is the only identifier some filings give, and until it is a
citation kind the pipeline can hold, every reference back to it is stranded:
there is no authority for `Id. ¶ 34` to attach to, so the claim it makes goes
unchecked rather than being checked wrongly.

The hazard on the other side is larger than it looks. A filing states its own
docket number in the caption and in every ECF page stamp -- one document in
false-citation-bench carries twenty identical ones -- and reading those as
citations would invent an authority per page. What separates a citation from a
page stamp is the court written with it, which is why most of these tests are
about declining rather than about finding.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from mellea_lrc.core.citations import DocketCitation, DocketEntry, FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import Document, Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree
from mellea_lrc.serialization.document import (
    deserialize_document,
    serialize_document,
)

INDICTMENT = (
    "See Indictment, United States v. Chen Zhi , No. 1:25-cr-00312-RPK "
    "(E.D.N.Y. filed Oct. 8, 2025). The Indictment alleges that the Prince Group "
    "built a criminal enterprise. Id. ¶¶ 30-31. It further alleges that the "
    "proceeds were laundered. Id. ¶ 34."
)


def _extract(text: str, relaxation: Relaxation = Relaxation.BOUNDED) -> Document:
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=relaxation, with_leaves=True)


def _dockets(text: str, relaxation: Relaxation = Relaxation.BOUNDED) -> list[DocketCitation]:
    return [
        item.stated
        for item in _extract(text, relaxation).citations
        if isinstance(item.stated, DocketCitation)
    ]


# --- What a docket citation is ------------------------------------------------


def test_a_docket_number_with_its_court_is_a_citation() -> None:
    """Both halves are the identifier, and both come back on the citation.

    ``1:25-cr-00312`` exists in every district. Only that number in the Eastern
    District of New York names a case, so a docket citation that reported the
    number alone would not identify anything.
    """
    (docket,) = _dockets(INDICTMENT)

    assert docket.docket_number == "1:25-cr-00312-RPK"
    assert docket.court == "nyed"
    assert docket.court_name == "District Court, E.D. New York"
    assert docket.defendant == "Chen Zhi"


def test_the_spans_point_at_the_docket_and_at_the_whole_citation() -> None:
    """The locator span is the identifier; the full span is the sentence around it.

    Downstream masks what was found by its locator span, so a span that ran to
    the end of the parenthetical would blank out the court and the date as
    though they had been read as part of the number.
    """
    (item,) = [c for c in _extract(INDICTMENT).citations if isinstance(c.stated, DocketCitation)]

    full = INDICTMENT[item.full_span.start : item.full_span.end]

    assert INDICTMENT[item.locator_span.start : item.locator_span.end] == "No. 1:25-cr-00312-RPK"
    assert item.matched_text == "No. 1:25-cr-00312-RPK"
    assert "United States v. Chen Zhi" in full
    assert full.endswith("(E.D.N.Y. filed Oct. 8, 2025)")


def test_a_written_document_entry_refines_the_case_docket() -> None:
    """`Doc. 75` names the filing; the case number still names its docket."""
    text = "See Doc. 75, Case No. 1:25-cv-00312-RPK (E.D.N.Y. Oct. 8, 2025)."

    (record,) = [item for item in _extract(text).citations if isinstance(item.stated, DocketCitation)]

    assert record.stated.docket_number == "1:25-cv-00312-RPK"
    assert record.stated.docket_entry == DocketEntry(
        number="75",
        span=Span(start=text.index("Doc. 75"), end=text.index("Doc. 75") + len("Doc. 75")),
    )
    assert text[record.full_span.start : record.full_span.end].startswith("Doc. 75, Case No.")


def test_a_case_docket_does_not_require_a_document_entry() -> None:
    """A docket-only citation is still complete enough to create a root."""
    (docket,) = _dockets("See Case No. 1:25-cv-00312-RPK (E.D.N.Y. Oct. 8, 2025).")

    assert docket.docket_entry is None


def test_a_docket_and_a_parallel_reporter_locator_are_two_citations() -> None:
    """They point at two databases, and neither substitutes for the other.

    RECAP holds the docket, a reporter corpus holds the locator, and they carry
    different information. Reading only one of them discards the other.
    """
    text = (
        "See Ginena v. Alaska Airlines, Inc. , No. 2:04-CV-01304-RCJ, 2011 WL 4749104, "
        "at *1 (D. Nev. Oct. 6, 2011) (granting a protective order)."
    )

    kinds = {type(c.stated).__name__ for c in _extract(text).citations}

    assert {"DocketCitation", "FullCaseCitation"} <= kinds


@pytest.mark.parametrize(
    "written",
    [
        "CIV 11-0107 JB/KBM",
        r"CIV 16-0318 JB\SCY",
        "13CV04115WHODMR",
        "CV 22-165 MIS/GBW",
        "CV-20-01788-PHX-JJT",
        "CV-2002309-PHX-MTL",
        "CV-06-02903-PHX-JAT",
        "CV 19-7532",
        "CV-15-00077",
        "CIV. A. 08-222-KD-B",
    ],
)
def test_non_cmecf_dockets_are_left_for_site_hunting(written: str) -> None:
    """The raw reader does not grow local docket grammars into roots."""
    assert _dockets(f"Smith v. Jones, No. {written}, 2024 WL 1234567 (D. Ariz. 2024).") == []


def test_a_cmecf_docket_before_a_database_locator_is_read_once() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567, at *4 (D. Ariz. Jan. 1, 2024)."

    dockets = [item for item in _extract(text).citations if isinstance(item.stated, DocketCitation)]

    assert [item.stated.docket_number for item in dockets] == ["1:24-cv-00123"]


def test_a_non_cmecf_docket_is_not_a_raw_root_candidate() -> None:
    assert _dockets("The clerk assigned No. CIV 11-0107 JB/KBM, and briefing followed.") == []


# --- What is not a docket citation --------------------------------------------


def test_a_caption_docket_number_is_kept_as_a_courtless_locator_candidate() -> None:
    """Locator discovery does not depend on resolving a court first."""
    text = (
        "IN THE UNITED STATES DISTRICT COURT\n\nFOR THE DISTRICT OF COLORADO\n\n"
        "Civil Action No. 1:24-cv-00814-PAB-SBP\n\nJAMIE LEE SAUNDERS,\n\nPlaintiff,"
    )

    (docket,) = _dockets(text)
    assert docket.docket_number == "1:24-cv-00814-PAB-SBP"
    assert docket.court is None


def test_an_unlabelled_ecf_page_stamp_is_left_for_site_hunting() -> None:
    text = (
        "COMPLAINT PLAINTIFF DEMANDS A JURY TRIAL ON ALL ISSUES SO TRIABLE - 5 5\n\n"
        "Case 2:25-cv-01295-GMS     Document 1     Filed 04/18/25     Page 6 of 32\n\n"
        "21. After Plaintiff rejected the advances, the retaliation began."
    )

    assert _dockets(text) == []


def test_an_unlabelled_ecf_stamp_is_not_rescued_by_a_nearby_court() -> None:
    text = (
        "See Doe v. Penzato , 2011 WL 1833007, at *3 (N.D. Cal. May 13, 2011); Doe v.\n\n"
        "Case 2:25-cv-01295-GMS     Document 21     Filed 06/12/25     Page 14 of 16\n\n"
        "Megless , 654 F.3d at 408."
    )

    assert _dockets(text) == []


def test_a_court_beyond_a_blank_line_does_not_belong_to_the_docket() -> None:
    """The same block-boundary rule the relaxation levels apply, for the same reason.

    Extraction interleaves a caption into the body, leaving a filing's own
    number a blank line ahead of an unrelated `(10th Cir. 1994)`. What lies
    beyond a paragraph break belongs to something else.
    """
    text = (
        "The admission of evidence lies within the discretion of the trial court. Robinson\n\n"
        "Plaintiff, Case No. 1:22-cv-01129-NYW-SBP\n\n"
        "v. Mo. Pac. R.R. Co ., 16 F.3d 1083, 1086 (10th Cir. 1994)."
    )

    (docket,) = _dockets(text)
    assert docket.docket_number == "1:22-cv-01129-NYW-SBP"
    assert docket.court is None


def test_the_assigned_judge_s_initials_are_not_a_court() -> None:
    """A caption's parenthesis holds the judge, and some initials spell a state.

    Reading the periods loosely is what lets `D.Ariz.` be recognised; the cost
    is that `(SC)` would otherwise resolve to South Carolina and turn a caption
    into a citation.
    """
    text = "UNITED STATES DISTRICT COURT\n\nCase No. 2:23-cv-6188  (SC) SUPERB MOTORS, INC.,"

    (docket,) = _dockets(text)
    assert docket.docket_number == "2:23-cv-6188"
    assert docket.court is None


# --- Damage the converter leaves behind ---------------------------------------


def test_a_canonical_cmecf_number_is_read_as_written() -> None:
    text = "United States v. Bitcoin, No. 1:25-cv-05745-RPK (E.D.N.Y. filed Oct. 14, 2025)."

    (docket,) = _dockets(text)

    assert docket.docket_number == "1:25-cv-05745-RPK"
    assert docket.court == "nyed"


@pytest.mark.parametrize("written", ("1:25cv-05745-RPK", "1:25-cv- 05745-RPK"))
def test_a_damaged_cmecf_number_is_left_for_site_hunting(written: str) -> None:
    assert _dockets(f"United States v. Bitcoin, No. {written} (E.D.N.Y. 2025).") == []


def test_a_number_broken_across_a_line_is_not_read_as_one() -> None:
    """Two numbers on two lines are not one number, and no reading recovers which.

    Unlike a reporter locator, a docket number has no gazetteer to check a
    doubtful reading against, so the tolerant reading has no way to be wrong
    safely.
    """
    assert _dockets("United States v. Chen Zhi , No. 1:25-cr-\n00312 (E.D.N.Y. 2025).") == []


# --- Relaxation has nothing to say about a docket -----------------------------


@pytest.mark.parametrize("relaxation", list(Relaxation))
def test_a_docket_is_read_the_same_at_every_relaxation_level(relaxation: Relaxation) -> None:
    """`Relaxation` governs the separators inside a reporter pattern, and only those.

    A docket citation is not a relaxed reading of anything, so it must not
    appear or disappear with a setting that is about whitespace in reporters.
    """
    (docket,) = _dockets(INDICTMENT, relaxation)

    assert docket.docket_number == "1:25-cr-00312-RPK"


# --- What it is for -----------------------------------------------------------


def test_an_id_chain_attributes_to_the_docket_it_heads() -> None:
    """The reason this exists: fifteen stranded back-references in one filing.

    Every `Id. ¶ N` after the indictment names a paragraph of that indictment.
    With no citation for the docket there is no authority for them to belong
    to, and a pinpoint check on them would be verifying a claim nobody made.
    """
    tree = build_citation_tree(_extract(INDICTMENT))

    (authority,) = tree.roots

    assert isinstance(authority.root.stated, DocketCitation)
    assert authority.pin_cites == ("¶¶ 30-31", "¶ 34")
    assert tree.unattributed == ()


def test_a_damaged_variant_is_not_silently_promoted_to_a_root() -> None:
    text = (
        "See Verified Compl. in Rem ¶ 21, United States v. Approximately 127,271 Bitcoin , "
        "No. 1:25-cv-05745-RPK (E.D.N.Y. filed Oct. 14, 2025). The complaint alleges control. "
        "See also Verified Compl. in Rem ¶¶ 40, 48, United States v. Approximately "
        "127,271 Bitcoin , No. 1:25cv-05745-RPK (E.D.N.Y. filed Oct. 14, 2025)."
    )

    (authority,) = build_citation_tree(_extract(text)).roots

    assert len(authority.occurrences) == 1


def test_two_courts_sharing_a_docket_number_are_two_authorities() -> None:
    """The court is part of the identity, not a label attached after the fact.

    The same number is live in every district at once, so merging them would
    check one case's claims against another case's document.
    """
    text = (
        "See Smith v. Jones , No. 1:19-cv-00362 (M.D.N.C. Jan. 26, 2021); "
        "see also Roe v. Poe , No. 1:19-cv-00362 (D. Nev. Oct. 6, 2011)."
    )

    tree = build_citation_tree(_extract(text))

    assert {a.root.stated.court for a in tree.roots} == {"ncmd", "nvd"}


def test_a_docket_citation_survives_a_serialization_round_trip() -> None:
    """A citation kind the artifact cannot carry is a citation kind nothing can use."""
    document = _extract(INDICTMENT)

    recovered = deserialize_document(serialize_document(document))

    assert recovered.citations == document.citations


def test_a_docket_is_a_full_citation_and_a_reporter_locator_is_still_its_own() -> None:
    """It identifies a case with no antecedent, which is what `full` means here.

    It is not, however, a `FullCaseCitation`: the arms that score reporter
    extraction select on that type, and a docket appearing among them would
    score as a false positive against a bench that deliberately excludes them.
    """
    document = _extract(INDICTMENT)

    (full,) = document.full_citations

    assert isinstance(full.stated, DocketCitation)
    assert not isinstance(full.stated, FullCaseCitation)


def test_a_cmecf_bankruptcy_number_is_a_locator() -> None:
    text = "In re Example, Docket No. 3-02-bk-12345 (Bankr. M.D. Pa. 2002)."

    (citation,) = _dockets(text)

    assert citation.docket_number == "3-02-bk-12345"


@pytest.mark.parametrize("written", ("06-01147", "2010712", "1124201", "26-10769"))
def test_non_cmecf_bankruptcy_numbers_are_left_for_site_hunting(written: str) -> None:
    assert _dockets(f"In re Example, No. {written} (Bankr. S.D.N.Y. 2020).") == []


def test_the_bankruptcy_shape_needs_the_signal_in_front_of_it() -> None:
    """A bare year and sequence is a page range as often as a docket number."""
    text = "The discussion runs from 06-01147 in the appendix (Bankr. S.D.N.Y. 2006)."

    assert not [c for c in _extract(text).citations if isinstance(c.stated, DocketCitation)]


def test_a_docket_number_with_no_office_is_still_a_locator() -> None:
    """Most courts write no office. `No. 22-cv-1231` is the ordinary form."""
    text = (
        "For example, in Doe v. Amazon.com, Inc. , No. 22-cv-1231, 2023 WL 3568691, "
        "at *3 (W.D. Wash. May 19, 2023), the court granted anonymity."
    )

    (citation,) = [c for c in _extract(text).citations if isinstance(c.stated, DocketCitation)]

    assert citation.stated.docket_number == "22-cv-1231"
    assert citation.stated.court_text == "W.D. Wash."


def test_reading_the_docket_keeps_it_out_of_the_case_name() -> None:
    """A docket number nothing reads is one the next citation's name absorbs.

    eyecite searches backward from `2023 WL 3568691` for a case name and stops
    at whatever it does not recognise. With the docket unread the defendant came
    back as `Amazon.com, Inc. , No. 22-cv-1231`.
    """
    text = (
        "For example, in Doe v. Amazon.com, Inc. , No. 22-cv-1231, 2023 WL 3568691, "
        "at *3 (W.D. Wash. May 19, 2023), the court granted anonymity."
    )

    reporter = next(c for c in _extract(text).citations if isinstance(c.stated, FullCaseCitation))

    assert reporter.stated.defendant == "Amazon.com, Inc."


def test_a_non_cmecf_bankruptcy_number_is_not_read_from_a_court_parenthetical() -> None:
    assert _dockets("In re FCI Mkts., No. 21-14743 (CL) (Bankr. S.D. Fla. 2021).") == []


def test_a_reporter_that_is_one_courts_reports_names_that_court() -> None:
    """A filing writing `5 N.C. App. 10 (1969)` names no court and needs none.

    The bridge is between the two databases the project already carries: a
    state's official reports are abbreviated the way its court is, so the
    edition is looked up against courts-db's own `citation_string`.
    """
    text = "In re X , 5 N.C. App. 10 (1969). Doe v. Roe , 556 U.S. 662 (2009). A v. B , 12 N.Y.2d 30 (1963)."
    assert [c.stated.court for c in _extract(text).citations] == ["ncctapp", "scotus", "ny"]


def test_a_reporter_several_courts_publish_in_names_none() -> None:
    """`206 P. 327 (1922)` says nothing about which court decided it.

    A guess would be worse than a gap: the Pacific Reporter carries fifteen
    states, and the filing is the only thing that could say which.
    """
    text = "Rae v. Poe , 206 P. 327 (1922). Coe v. Foe , 192 F.3d 742 (1999)."
    assert [c.stated.court for c in _extract(text).citations] == [None, None]


def test_the_court_the_filing_writes_wins_over_the_reporter_it_cites() -> None:
    text = "United States v. Kim , 5 N.C. App. 10 (4th Cir. 1969)."
    assert [c.stated.court for c in _extract(text).citations] == ["ca4"]


def test_a_docket_entry_number_is_not_a_case_docket() -> None:
    """Neither a local case number nor a docket entry is in the CM/ECF family."""
    text = "Smith, No. 21-11854 (DSJ) [D.I. No. 17] (Bankr. S.D.N.Y. 2021)."

    dockets = [item for item in _extract(text).citations if isinstance(item.stated, DocketCitation)]

    assert dockets == []
