"""Tests for finding a case cited in a form that locates nothing.

Every other candidate generator disagrees with the record about a span the
record holds. This one proposes spans the record does not have at all: with no
reporter there is no token, so eyecite produces nothing and a filing full of
these earns "nothing to report" rather than "could not be read".
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.extraction import extract_from_plain_text
from mellea_lrc.extraction.adjudication.candidates import nonconforming_citations
from mellea_lrc.extraction.adjudication.types import CandidateKind


def _proposed(text: str) -> list[str]:
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, source_path="test.txt")
    return [
        text[candidate.span.start : candidate.span.end] for candidate in nonconforming_citations(document)
    ]


def test_a_name_and_a_court_and_year_with_no_reporter_is_proposed() -> None:
    """The shape document 013 lists five authorities in.

    A reader given this cannot reach a decision, and eyecite reads nothing here
    at any relaxation, so without this generator the citation leaves no trace.
    """
    text = (
        "Key among them are: Chugach Natives, Inc. v. Doyon, Ltd. (D. Alaska 1984) - "
        "Clarifies the standards under ANCSA for conveyance of land interests."
    )

    assert _proposed(text) == ["Chugach Natives, Inc. v. Doyon, Ltd. (D. Alaska 1984)"]


def test_the_kind_says_the_citation_conforms_to_no_form() -> None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(
            "See Akiachak Native Community v. U.S. Department of the Interior (D.D.C. 2016).",
            source_path="test.txt",
        )

    kinds = {candidate.kind for candidate in nonconforming_citations(document)}

    assert kinds == {CandidateKind.NONCONFORMING}


def test_a_name_in_front_of_its_own_citation_is_not_proposed() -> None:
    """The ordinary case, and the one that would drown a reviewer."""
    assert _proposed("See Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).") == []


def test_a_name_whose_parties_eyecite_did_not_parse_is_not_proposed() -> None:
    """A docket number between the name and the reporter is still that citation.

    `Rivero v. Bd. of Regents of Univ. of New Mexico , No. CIV 16-0318 JB\\SCY,
    2019 WL 1085179` comes back from eyecite with no plaintiff or defendant, so
    the name is exposed. It is not a case cited without a locator.
    """
    text = (
        "The standard is loose. Rivero v. Bd. of Regents of Univ. of New Mexico , "
        "No. CIV 16-0318 JB\\SCY, 2019 WL 1085179, at *78 (D.N.M. Mar. 7, 2019)."
    )

    assert _proposed(text) == []


def test_a_case_cited_by_docket_number_is_not_proposed() -> None:
    """Bluebook Rule 10.8.1 cites an unreported case this way, and it locates one.

    Document 015 does it eleven times. Nothing is missing from those, and a
    generator that proposed them would be reporting correct citation as a defect.
    """
    text = (
        "See also In re Muscletech Research and Dev. Inc. , No. 06-01147 (JMP) "
        "(Bankr. S.D.N.Y. Jan. 18, 2006) (entering a temporary restraining order)."
    )

    assert _proposed(text) == []


def test_a_textual_short_form_for_a_case_cited_in_full_is_not_proposed() -> None:
    """Rule 10.9 permits the name alone once the case has been given in full."""
    text = (
        "Doe v. Skyline Automobiles Inc., 375 F. Supp. 3d 401 (S.D.N.Y. 2019). "
        "For instance, Doe v. Skyline denied anonymity because the plaintiff had "
        "publicly disclosed her identity."
    )

    assert _proposed(text) == []


def test_a_table_of_authorities_row_is_not_proposed() -> None:
    """A table lists names against the filing's own pages, not the reporter's."""
    text = "| Doe v. Megless , | ……………………………… 6 |\n| Roe v. Bernabei & Wachtel | passim |\n"

    assert _proposed(text) == []


def test_an_in_re_case_named_with_no_locator_is_proposed() -> None:
    """`In re` is the case-name form with no `v.` in it, and cites the same way."""
    text = "A full stay is not required under In re Flint Water Cases, and Plaintiffs suffer."

    assert _proposed(text) == ["In re Flint Water Cases"]
