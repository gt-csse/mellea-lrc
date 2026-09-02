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

from mellea_lrc.core.citations import DocketCitation, FullCaseCitation
from mellea_lrc.extraction import ExtractedDocument, Relaxation, extract_from_plain_text
from mellea_lrc.extraction.citation_tree import build_citation_tree
from mellea_lrc.serialization.extracted_document import (
    deserialize_extracted_document,
    serialize_extracted_document,
)

INDICTMENT = (
    "See Indictment, United States v. Chen Zhi , No. 1:25-cr-00312-RPK "
    "(E.D.N.Y. filed Oct. 8, 2025). The Indictment alleges that the Prince Group "
    "built a criminal enterprise. Id. ¶¶ 30-31. It further alleges that the "
    "proceeds were laundered. Id. ¶ 34."
)


def _extract(text: str, relaxation: Relaxation = Relaxation.BOUNDED) -> ExtractedDocument:
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=relaxation)


def _dockets(text: str, relaxation: Relaxation = Relaxation.BOUNDED) -> list[DocketCitation]:
    return [
        item.citation
        for item in _extract(text, relaxation).citations
        if isinstance(item.citation, DocketCitation)
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
    (item,) = [c for c in _extract(INDICTMENT).citations if isinstance(c.citation, DocketCitation)]

    full = INDICTMENT[item.span.start : item.span.end]

    assert INDICTMENT[item.locator_span.start : item.locator_span.end] == "No. 1:25-cr-00312-RPK"
    assert item.matched_text == "No. 1:25-cr-00312-RPK"
    assert "United States v. Chen Zhi" in full
    assert full.endswith("(E.D.N.Y. filed Oct. 8, 2025)")


def test_a_docket_and_a_parallel_reporter_locator_are_two_citations() -> None:
    """They point at two databases, and neither substitutes for the other.

    RECAP holds the docket, a reporter corpus holds the locator, and they carry
    different information. Reading only one of them discards the other.
    """
    text = (
        "See Ginena v. Alaska Airlines, Inc. , No. 2:04-CV-01304-RCJ, 2011 WL 4749104, "
        "at *1 (D. Nev. Oct. 6, 2011) (granting a protective order)."
    )

    kinds = {type(c.citation).__name__ for c in _extract(text).citations}

    assert {"DocketCitation", "FullCaseCitation"} <= kinds


# --- What is not a docket citation --------------------------------------------


def test_a_caption_docket_number_with_no_court_is_declined() -> None:
    """A filing's own number, in its own caption, is not a citation to anything.

    Nothing here says which court, and guessing one from the surrounding page
    would be inventing the half of the identifier that is missing.
    """
    text = (
        "IN THE UNITED STATES DISTRICT COURT\n\nFOR THE DISTRICT OF COLORADO\n\n"
        "Civil Action No. 1:24-cv-00814-PAB-SBP\n\nJAMIE LEE SAUNDERS,\n\nPlaintiff,"
    )

    assert _dockets(text) == []


def test_an_ecf_page_stamp_is_not_a_citation() -> None:
    """Twenty identical stamps in one filing would be twenty invented authorities.

    These are page furniture that preprocessing should have removed and did
    not, so the extractor has to survive them rather than assume they are gone.
    """
    text = (
        "COMPLAINT PLAINTIFF DEMANDS A JURY TRIAL ON ALL ISSUES SO TRIABLE - 5 5\n\n"
        "Case 2:25-cv-01295-GMS     Document 1     Filed 04/18/25     Page 6 of 32\n\n"
        "21. After Plaintiff rejected the advances, the retaliation began."
    )

    assert _dockets(text) == []


def test_a_court_belonging_to_the_citation_before_it_is_not_this_docket_s_court() -> None:
    """Proximity is not attribution, and this is where a nearby-court rule fails.

    Document 022 stamps its own case number forty characters after another
    case's `(N.D. Cal. May 13, 2011)`. A rule that looked either way would read
    the page stamp as a citation and give it the wrong court besides.
    """
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

    assert _dockets(text) == []


def test_the_assigned_judge_s_initials_are_not_a_court() -> None:
    """A caption's parenthesis holds the judge, and some initials spell a state.

    Reading the periods loosely is what lets `D.Ariz.` be recognised; the cost
    is that `(SC)` would otherwise resolve to South Carolina and turn a caption
    into a citation.
    """
    text = "UNITED STATES DISTRICT COURT\n\nCase No. 2:23-cv-6188  (SC) SUPERB MOTORS, INC.,"

    assert _dockets(text) == []


# --- Damage the converter leaves behind ---------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("No. 1:25-cv-05745-RPK", "1:25-cv-05745-RPK"),
        ("No. 1:25cv-05745-RPK", "1:25cv-05745-RPK"),
        ("No. 1:25-cv- 05745-RPK", "1:25-cv- 05745-RPK"),
    ],
)
def test_a_number_damaged_by_the_converter_is_still_read(written: str, expected: str) -> None:
    """Both of these are real: a lost hyphen and a space inside the number.

    The number is reported as the filing wrote it, damage included. A citation
    silently repaired is a citation a reader cannot check against the page.
    """
    text = f"United States v. Approximately 127,271 Bitcoin , {written} (E.D.N.Y. filed Oct. 14, 2025)."

    (docket,) = _dockets(text)

    assert docket.docket_number == expected
    assert docket.court == "nyed"


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

    (authority,) = tree.authorities

    assert isinstance(authority.root.citation, DocketCitation)
    assert authority.pin_cites == ("¶¶ 30-31", "¶ 34")
    assert tree.unattributed == ()


def test_one_docket_written_two_ways_is_one_authority() -> None:
    """A hyphen the converter dropped does not make a second case.

    Document 016 writes the same forfeiture docket both ways, and counting them
    as two authorities would double the lookups and split the claims made about
    one document across two.
    """
    text = (
        "See Verified Compl. in Rem ¶ 21, United States v. Approximately 127,271 Bitcoin , "
        "No. 1:25-cv-05745-RPK (E.D.N.Y. filed Oct. 14, 2025). The complaint alleges control. "
        "See also Verified Compl. in Rem ¶¶ 40, 48, United States v. Approximately "
        "127,271 Bitcoin , No. 1:25cv-05745-RPK (E.D.N.Y. filed Oct. 14, 2025)."
    )

    (authority,) = build_citation_tree(_extract(text)).authorities

    assert len(authority.occurrences) == 2


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

    assert {a.root.citation.court for a in tree.authorities} == {"ncmd", "nvd"}


def test_a_docket_citation_survives_a_serialization_round_trip() -> None:
    """A citation kind the artifact cannot carry is a citation kind nothing can use."""
    document = _extract(INDICTMENT)

    recovered = deserialize_extracted_document(serialize_extracted_document(document))

    assert recovered.citations == document.citations


def test_a_docket_is_a_full_citation_and_a_reporter_locator_is_still_its_own() -> None:
    """It identifies a case with no antecedent, which is what `full` means here.

    It is not, however, a `FullCaseCitation`: the arms that score reporter
    extraction select on that type, and a docket appearing among them would
    score as a false positive against a bench that deliberately excludes them.
    """
    document = _extract(INDICTMENT)

    (full,) = document.full_citations

    assert isinstance(full.citation, DocketCitation)
    assert not isinstance(full.citation, FullCaseCitation)
