"""Tests for the case names the extraction chain did not consume.

Read last, from the document with every citation blanked, so what is left is by
definition what nothing else read. No rule here decides what a name in that
residue means: masking already answers the only question it could ask, which is
whether a locator was read at that position.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.extraction import extract_from_plain_text


def _unread(text: str) -> list[str]:
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, source_path="test.txt")
    return [text[span.start : span.end] for span in document.unread_case_names]


def test_a_case_cited_with_no_locator_is_left_over() -> None:
    """The shape document 013 lists five authorities in.

    eyecite is reporter-driven, so with no volume, reporter, page or docket
    number it produces nothing at any relaxation and the citation leaves no
    trace. Without this the filing earns "nothing to report".
    """
    text = (
        "Key among them are: Chugach Natives, Inc. v. Doyon, Ltd. (D. Alaska 1984) - "
        "Clarifies the standards under ANCSA for conveyance of land interests."
    )

    assert _unread(text) == ["Chugach Natives, Inc. v. Doyon, Ltd."]


def test_a_name_a_citation_covers_is_not_left_over() -> None:
    assert _unread("See Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).") == []


def test_a_name_the_citation_did_not_reach_is_left_over() -> None:
    """The citation is read and half its name is not, which nothing else records.

    Extraction spaces the apostrophe out of `BYJU's`, and eyecite's search for a
    case name stops there: the party comes back as `Alpha, Inc.` and the span
    starts after `In re BYJU ' s`. Reporting the remainder is how that is seen.
    """
    text = (
        "See In re BYJU ' s Alpha, Inc. , 2024 WL 1455586, at *5 "
        "(Bankr. D. Del. Apr. 3, 2024) (granting relief)."
    )

    assert _unread(text) == ["In re BYJU ' s"]


def test_a_short_form_is_left_over_like_anything_else() -> None:
    """Whether a repeat is proper is a reading, and no rule here makes it.

    Suppressing it would be a claim about citation form, and the same shape is a
    defect where the filing never gives the case in full.
    """
    text = (
        "Doe v. Skyline Automobiles Inc., 375 F. Supp. 3d 401 (S.D.N.Y. 2019). "
        "For instance, Doe v. Skyline denied anonymity."
    )

    assert _unread(text) == ["Doe v. Skyline"]


def test_an_in_re_case_with_no_locator_is_left_over() -> None:
    """`In re` is the case-name form with no `v.` in it, and cites the same way."""
    text = "A full stay is not required under In re Flint Water Cases, and Plaintiffs suffer."

    assert _unread(text) == ["In re Flint Water Cases"]


def test_a_document_that_cites_cleanly_leaves_nothing() -> None:
    text = (
        "See Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009); "
        "Bell Atl. Corp. v. Twombly, 550 U.S. 544, 570 (2007)."
    )

    assert _unread(text) == []
