r"""Tests for what each relaxation level will and will not join.

eyecite joins volume, reporter and page with a literal single space, so any
damage to that separator makes the citation vanish rather than parse badly --
and a citation that vanishes is absent from the ledger, so a filing full of
them earns a clean bill instead of an incomplete one.

Damage runs in both directions. Extraction leaves doubled spaces and line
breaks; some documents carry the opposite, a missing space, either from OCR or
from a word processor that wrote `846F.2d746` into its own text layer. One
relaxation covers both.

The bound matters as much as the relaxation, which is why there are three
levels and not a flag. Between reporter and page, whitespace that may be
crossed stops at one newline: a citation can be broken across a line or a page,
but never across a blank line, because on pleading paper what sits beyond a
blank line is the margin line numbers. `FULL` crosses it anyway, deliberately.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import ExtractedDocument, Relaxation, extract_from_plain_text


def _extract(text: str, relaxation: Relaxation) -> ExtractedDocument:
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=relaxation)


def _matched(text: str, relaxation: Relaxation = Relaxation.BOUNDED) -> list[str]:
    return [item.matched_text for item in _extract(text, relaxation).citations]


def _locators(text: str, relaxation: Relaxation = Relaxation.BOUNDED) -> set[str]:
    return {
        f"{c.citation.volume} {c.citation.reporter} {c.citation.page}"
        for c in _extract(text, relaxation).citations
        if isinstance(c.citation, FullCaseCitation)
    }


def _only_full_case(text: str, relaxation: Relaxation = Relaxation.BOUNDED):
    (citation,) = [
        item for item in _extract(text, relaxation).citations if isinstance(item.citation, FullCaseCitation)
    ]
    return citation


# --- What NONE is, and why nothing ships on it -------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The court relied on 846F.2d746 for that proposition.",
        "Doe v. Colgate Univ. , 2016 WL1448829, at *2 (N.D.N.Y. Apr. 12, 2016)",
        "See also White v. McBride , 937\n\nS.W.2d  796,  800  (Tenn.  1996)",
        "Cracker Barrel Old  Country  Store,  Inc.  v.  Epperson ,  284  S.W.3d  303,  312",
    ],
)
def test_unrelaxed_extraction_finds_nothing_in_damaged_text(text: str) -> None:
    """The failure mode worth naming: not a bad parse, no parse at all.

    Nothing downstream reports a problem, because nothing downstream was told a
    citation was there.
    """
    assert _locators(text, Relaxation.NONE) == set()


def test_unrelaxed_extraction_is_eyecite_as_published() -> None:
    text = "Norton v. Shelby County, 118 U.S. 425, 442 (1886)"
    assert _locators(text, Relaxation.NONE) == {"118 U.S. 425"}


# --- What every relaxed level recovers ---------------------------------------


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The court relied on 846F.2d746 for that proposition.", "846F.2d746"),
        ("Reversed at 347U.S.483 (1954).", "347U.S.483"),
        ("See Ashcroft v. Iqbal, 556U.S.662, 678 (2009).", "556U.S.662"),
    ],
)
def test_a_glued_citation_is_recovered(text: str, expected: str, relaxation: Relaxation) -> None:
    """A missing separator is damage, not the absence of a citation."""
    assert expected in _matched(text, relaxation)


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See 550  F.3d  1000 here.", "550  F.3d  1000"),
        ("See 347 U.S.\n483 (1954).", "347 U.S.\n483"),
    ],
)
def test_separator_damage_in_the_other_direction_still_matches(
    text: str, expected: str, relaxation: Relaxation
) -> None:
    """Doubled spaces and a line break are the same defect seen from the other side.

    The matched text keeps the damage. Nothing is collapsed before parsing, so
    what comes back is the source text, not a normalized rendering of it.
    """
    assert expected in _matched(text, relaxation)


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
def test_a_volume_split_from_its_reporter_by_a_page_break_is_recovered(
    relaxation: Relaxation,
) -> None:
    """The volume-to-reporter join crosses a blank line at every relaxed level.

    A break there leaves reporter and page still adjacent on the far side, so
    the page that gets captured is the citation's own. There is no wrong-page
    hazard to trade against.
    """
    text = "See also White v. McBride , 937\n\nS.W.2d  796,  800  (Tenn.  1996)"
    assert "937 S.W.2d 796" in _locators(text, relaxation)


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
def test_an_ordinary_citation_is_unaffected(relaxation: Relaxation) -> None:
    """Relaxation must not change what already worked."""
    text = "See Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009)."
    assert _matched(text, relaxation) == _matched(text, Relaxation.NONE)


# --- Where BOUNDED and FULL part ---------------------------------------------


def test_bounded_never_joins_a_page_across_a_blank_line() -> None:
    """Two newlines are a block boundary, and no page number is read across one.

    Allowing arbitrary whitespace joins `214 F.3d` to the `1` that opens the
    next block -- which on pleading paper is the first margin line number. The
    real citation is `214 F.3d 1058`, so this is not a miss but a wrong page,
    and it sends validation to a different case with a confident verdict about
    it. That is the worst outcome available to an extractor.
    """
    assert _matched("See 214 F.3d\n\n1 The next paragraph begins here.") == []


def test_full_joins_a_page_across_a_blank_line() -> None:
    """The same behaviour, asserted as chosen rather than as a bug.

    It is what reaches a citation whose page landed past a page break, and it
    is why `FULL` is never a default.
    """
    text = "Advanced Textile , 214 F.3d\n\n1\n\n2\n\n3\n\n4"
    assert "214 F.3d 1" in _locators(text, Relaxation.FULL)


# --- Punctuation inside the reporter -----------------------------------------


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See 58  N.Y .2d  916 (1983).", "58  N.Y .2d  916"),
        ("See 58 N .Y.2d 916 (1983).", "58 N .Y.2d 916"),
    ],
)
def test_a_space_before_a_period_inside_the_reporter_is_tolerated(
    text: str, expected: str, relaxation: Relaxation
) -> None:
    r"""eyecite allows whitespace after a period inside a reporter, not before it.

    Its generated pattern is `N\.\s*Y\.\s*2d`, so `N.Y.2d` and `N.Y. 2d` both
    match and `N.Y .2d` matches nothing at all. Extraction puts a space on that
    side as readily as the other, and `58  N.Y .2d  916` is a real citation on
    false-citation-bench -- the last one no tokenizer reached.
    """
    assert _only_full_case(text, relaxation).matched_text == expected


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See 777 F. App ' x 516 (Fed. Cir. 2019).", "777 F. App ' x 516"),
        ("See 777 F. App'x 516 (Fed. Cir. 2019).", "777 F. App'x 516"),
    ],
)
def test_spaces_around_an_apostrophe_in_the_reporter_are_tolerated(
    text: str, expected: str, relaxation: Relaxation
) -> None:
    """eyecite writes `App'x` with the apostrophe tight and allows nothing around it.

    Extraction spaces it out. `777 F. App ' x 516` appears in a real filing,
    printed `777 F. App'x 516` on the page. It is the same defect as the
    period, so it takes the same relaxation rather than a rule of its own.
    """
    assert _only_full_case(text, relaxation).matched_text == expected


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See 206 F. Supp. 3d 1304 (2016).", "206 F. Supp. 3d 1304"),
        ("See 206 F.Supp.3d 1304 (2016).", "206 F.Supp.3d 1304"),
        ("See 58 N.Y.2d 916 (1983).", "58 N.Y.2d 916"),
        ("See 473 F. App'x 160 (2012).", "473 F. App'x 160"),
    ],
)
def test_an_ordinary_reporter_is_unaffected_by_that(text: str, expected: str, relaxation: Relaxation) -> None:
    assert _only_full_case(text, relaxation).matched_text == expected


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
def test_the_case_reporter_closing_period_is_still_required(relaxation: Relaxation) -> None:
    """It is what separates the reporter from the page in `410 U.S. 113`.

    Making it optional would let the reporter run into the number after it.
    """
    assert _locators("Roe v. Wade , 410 U.S 113 (1973)", relaxation) == set()


@pytest.mark.parametrize("relaxation", [Relaxation.BOUNDED, Relaxation.FULL])
def test_reporter_groups_are_not_left_with_absorbed_whitespace(relaxation: Relaxation) -> None:
    r"""Relaxed separators let alternation branches ending in \s* keep a space."""
    text = "United States v. Rucker , 188 Fed. Appx. 772, 778 (10th Cir. 2006)"
    reporters = {
        c.citation.reporter
        for c in _extract(text, relaxation).citations
        if isinstance(c.citation, FullCaseCitation)
    }
    assert reporters
    assert all(r == r.strip() for r in reporters if r)


# --- Spans and provenance ----------------------------------------------------


@pytest.mark.parametrize("relaxation", list(Relaxation))
def test_spans_index_the_text_exactly_as_given(relaxation: Relaxation) -> None:
    """No text is rewritten at any level, so no span needs remapping."""
    text = (
        "The court in Cracker Barrel Old  Country  Store,  Inc.  v.  Epperson ,  "
        "284  S.W.3d  303,  312 (Tenn. 2009) held as much."
    )
    document = _extract(text, relaxation)
    assert document.text == text
    for citation in document.citations:
        assert text[citation.locator_span.start : citation.locator_span.end] == citation.matched_text


@pytest.mark.parametrize("relaxation", list(Relaxation))
def test_the_document_records_which_tokenizer_read_it(relaxation: Relaxation) -> None:
    """Two levels disagree about what is in the text, so the answer is not self-describing."""
    document = _extract("Norton v. Shelby County, 118 U.S. 425 (1886)", relaxation)
    assert document.extraction_metadata.relaxation is relaxation


def test_the_default_is_bounded() -> None:
    """The shipped setting, asserted so a change to it is a deliberate edit."""
    document = extract_from_plain_text("Norton v. Shelby County, 118 U.S. 425 (1886)")
    assert document.extraction_metadata.relaxation is Relaxation.BOUNDED
