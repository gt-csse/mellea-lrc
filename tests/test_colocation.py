"""Tests for grouping citations that occupy the same place in the text.

The distinction under test is between *reporting* co-location and *deciding*
identity. Two citations sitting in one place may be one authority written in
two reporters, or two authorities written side by side, and extraction is not
the layer that can tell. So these tests assert that a group is formed, and
never that its members name the same case.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text


def _extract(text: str):
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=Relaxation.FULL)


def _groups(document) -> list[set[str]]:
    """The matched text of each co-located set."""
    grouped: dict[str, set[str]] = {}
    for citation in document.citations:
        if citation.colocation_id:
            grouped.setdefault(citation.colocation_id, set()).add(" ".join(citation.matched_text.split()))
    return list(grouped.values())


def test_a_parallel_citation_is_one_group() -> None:
    """Three reporters of one case are written as one citation."""
    text = "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968)."

    assert _groups(_extract(text)) == [{"390 U.S. 727", "88 S.Ct. 1323", "20 L.Ed.2d 262"}]


def test_the_group_survives_spans_that_do_not_coincide_exactly() -> None:
    """eyecite gives the members of one parallel cite spans that differ by a character.

    Grouping on equality rather than overlap would leave the first reporter out
    of its own group, which is the reason the rule is written on overlap.
    """
    document = _extract("St. Amant v. Thompson, 390 U.S. 727, 88 S.Ct. 1323 (1968).")
    spans = {(c.full_span.start, c.full_span.end) for c in document.citations}

    assert len(spans) > 1, "this test is pointless if the spans are already identical"
    assert _groups(document) == [{"390 U.S. 727", "88 S.Ct. 1323"}]


def test_two_cases_in_one_reporter_are_never_grouped() -> None:
    """`Brown, 347 U.S. 483, 349 U.S. 294` is Brown I and Brown II.

    One case name, one year parenthetical, coinciding spans, and two decisions.
    Nothing in the text distinguishes it from a parallel cite -- the reporter
    does, because a case has one first page in one reporter.
    """
    assert _groups(_extract("See Brown, 347 U.S. 483, 349 U.S. 294 (1955).")) == []


def test_a_string_cite_is_not_a_group() -> None:
    """A semicolon ends a citation's full span, so the two do not overlap."""
    assert _groups(_extract("Lacey, 693 F.3d 896, 912; Garmon, 828 F.3d 837, 843.")) == []


def test_an_ordinary_citation_has_no_group() -> None:
    """The common case: a citation standing alone carries no id at all."""
    document = _extract("Norton v. Shelby County, 118 U.S. 425, 442 (1886).")

    assert _groups(document) == []
    assert all(c.colocation_id is None for c in document.citations)


def test_a_short_form_is_not_co_located_with_its_authority() -> None:
    """A short form refers to an authority; it is not another name for one.

    Only a citation naming volume, reporter and page can be one identifier
    among several for the same case.
    """
    document = _extract("Iqbal, 556 U.S. 662, 678 (2009). See Iqbal, 556 U.S. at 678.")
    short = [c for c in document.citations if not isinstance(c.citation, FullCaseCitation)]

    assert short, "expected a short form in this text"
    assert all(c.colocation_id is None for c in short)


def test_the_group_id_is_a_citation_id_from_the_group() -> None:
    """So a reader of a serialized document can find the members by it."""
    document = _extract("St. Amant v. Thompson, 390 U.S. 727, 88 S.Ct. 1323 (1968).")
    grouped = [c for c in document.citations if c.colocation_id]
    identifiers = {c.citation_id for c in grouped}

    assert grouped
    assert {c.colocation_id for c in grouped} <= identifiers
