"""Tests for finding every case named with no locator attached to it.

One bucket, filled and not filtered. eyecite's tokenizer is reporter-driven, so
a name with no volume, reporter, page or docket number produces nothing at any
relaxation, and a filing full of them earns "nothing to report". What each one
turns out to be -- a citation that locates nothing, a short form the filing is
entitled to, the caption naming its own parties -- is a reading, so it is
reported in the note rather than decided by a rule here.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.extraction import extract_from_plain_text
from mellea_lrc.extraction.adjudication.candidates import bare_case_names
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind


def _candidates(text: str) -> list[Candidate]:
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, source_path="test.txt")
    return list(bare_case_names(document))


def _named(text: str) -> list[str]:
    return [text[c.span.start : c.span.end] for c in _candidates(text)]


def test_a_name_and_a_court_and_year_with_no_reporter_is_found() -> None:
    """The shape document 013 lists five authorities in.

    A reader given this cannot reach a decision, and eyecite reads nothing here
    at any relaxation, so without this generator the citation leaves no trace.
    """
    text = (
        "Key among them are: Chugach Natives, Inc. v. Doyon, Ltd. (D. Alaska 1984) - "
        "Clarifies the standards under ANCSA for conveyance of land interests."
    )

    assert _named(text) == ["Chugach Natives, Inc. v. Doyon, Ltd. (D. Alaska 1984)"]


def test_the_kind_describes_the_span_and_does_not_judge_it() -> None:
    text = "See Akiachak Native Community v. U.S. Department of the Interior (D.D.C. 2016)."

    assert {c.kind for c in _candidates(text)} == {CandidateKind.BARE_CASE_NAME}


def test_a_name_in_front_of_its_own_citation_is_not_a_bare_name() -> None:
    """A name with a locator attached is not a name without one."""
    assert _named("See Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).") == []


def test_a_name_whose_parties_eyecite_did_not_parse_is_not_a_bare_name() -> None:
    """A docket number between the name and the reporter is still that citation.

    `Rivero v. Bd. of Regents of Univ. of New Mexico , No. CIV 16-0318 JB\\SCY,
    2019 WL 1085179` comes back from eyecite with no plaintiff or defendant, so
    the name is exposed. The filing did state a locator.
    """
    text = (
        "The standard is loose. Rivero v. Bd. of Regents of Univ. of New Mexico , "
        "No. CIV 16-0318 JB\\SCY, 2019 WL 1085179, at *78 (D.N.M. Mar. 7, 2019)."
    )

    assert _named(text) == []


def test_a_docket_number_is_a_locator() -> None:
    """It identifies a case as surely as a reporter page, and cites an unreported one."""
    text = (
        "See also In re Muscletech Research and Dev. Inc. , No. 06-01147 (JMP) "
        "(Bankr. S.D.N.Y. Jan. 18, 2006) (entering a temporary restraining order)."
    )

    assert _named(text) == []


def test_a_repeat_of_a_case_cited_in_full_is_reported_rather_than_dropped() -> None:
    """Whether a short form is proper is a reading, so the note says what was seen.

    Suppressing it here would be a rule about citation form fitted to one corpus,
    and the same shape is a defect where the filing never gives the case in full.
    """
    text = (
        "Doe v. Skyline Automobiles Inc., 375 F. Supp. 3d 401 (S.D.N.Y. 2019). "
        "For instance, Doe v. Skyline denied anonymity because the plaintiff had "
        "publicly disclosed her identity."
    )
    candidates = _candidates(text)

    assert [text[c.span.start : c.span.end] for c in candidates] == ["Doe v. Skyline"]
    assert "cites this case with a locator elsewhere" in candidates[0].note


def test_a_table_row_is_reported_as_one() -> None:
    text = "| Doe v. Megless , | ……………………………… 6 |\n"
    candidates = _candidates(text)

    assert len(candidates) == 1
    assert "row of a table" in candidates[0].note


def test_an_in_re_case_named_with_no_locator_is_found() -> None:
    """`In re` is the case-name form with no `v.` in it, and cites the same way."""
    text = "A full stay is not required under In re Flint Water Cases, and Plaintiffs suffer."

    assert _named(text) == ["In re Flint Water Cases"]
