"""Tests for resolving the court a parenthetical names.

eyecite matches courts by one hand-entered spelling per court, with a prefix
fallback when that spelling is not what the filing wrote. Both halves fail on
real briefs, and the fallback fails by answering rather than by declining.
"""

from __future__ import annotations

from mellea_lrc.extraction.courts import resolve_court


def test_the_bluebook_ordinal_resolves() -> None:
    """`3d Cir.` is what the Bluebook prescribes; courts-db stores `3rd Cir.`."""
    assert resolve_court("3d Cir.") == "ca3"


def test_the_other_ordinal_resolves_to_the_same_court() -> None:
    assert resolve_court("3rd Cir.") == "ca3"


def test_a_second_circuit_citation_is_not_the_bankruptcy_panel() -> None:
    """eyecite returns bap2 here, because `2nd Cir. BAP` starts with `2nd Cir.`.

    A different court, returned with nothing to say a guess was made.
    """
    assert resolve_court("2nd Cir.") == "ca2"
    assert resolve_court("2d Cir.") == "ca2"


def test_an_ordinal_broken_by_extraction_still_resolves() -> None:
    """`(2 nd Cir. 2009)` is in this corpus; the space is a converter artefact."""
    assert resolve_court("2 nd Cir.") == "ca2"


def test_an_unambiguous_prefix_is_accepted() -> None:
    """Only one court is spelled `D. Minnesota`, so `D. Minn.` is not a guess."""
    assert resolve_court("D. Minn.") == "mnd"


def test_an_ambiguous_prefix_is_declined() -> None:
    """`Ct. App.` prefixes both Nevada and Indian Territory.

    eyecite returns whichever it happens to reach last. Returning nothing is the
    only honest answer: an unidentified court must not come back plausible.
    """
    assert resolve_court("Ct. App.") is None


def test_a_new_york_department_resolves_to_the_appellate_division() -> None:
    """courts-db holds the four departments as one court and lists no department.

    32 of the courts this corpus states and eyecite drops are of this shape.
    """
    assert resolve_court("2d Dept.") == "nyappdiv"
    assert resolve_court("1st Dep't") == "nyappdiv"
    assert resolve_court("3 rd Dept.") == "nyappdiv"


def test_an_ordinary_court_still_resolves() -> None:
    assert resolve_court("S.D.N.Y.") == "nysd"
    assert resolve_court("D.C. Cir.") == "cadc"


def test_nothing_resolves_to_nothing() -> None:
    assert resolve_court("") is None
    assert resolve_court(None) is None
    assert resolve_court("   ") is None
