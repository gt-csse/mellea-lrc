"""Tests for reading a pin cite through the whitespace extraction leaves.

Losing a pin cite is not a cosmetic failure. It is the page a filing argues
from, so a citation without one carries no checkable claim about what the case
says -- and eyecite does not report the loss, it files the page under `extra`
where nothing looks for it.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text


def _first(text: str, relaxation: Relaxation = Relaxation.BOUNDED):
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=relaxation)
    return next(c for c in document.citations if isinstance(c.citation, FullCaseCitation))


def test_a_doubled_space_before_the_pin_cite_is_read() -> None:
    """`544,  570` is what justified text leaves behind, and it lost the page."""
    citation = _first("Bell Atl. Corp. v. Twombly,  550  U.S.  544,  570  (2007).")

    assert citation.citation.pin_cite == "570"


def test_a_spaced_range_hyphen_is_read() -> None:
    """`998 -1003` is a page range whose hyphen extraction has spaced.

    41 of the 42 pin cites that survived the space widening on
    false-citation-bench were this one shape.
    """
    citation = _first("Doe v. Roe, 899 F.3d 988, 998 -1003 (9th Cir. 2018).")

    # Kept as written, spacing and all. The widening decides what parses, not
    # how it is spelled, and a consumer comparing pages reads the first number.
    assert citation.citation.pin_cite == "998 -1003"


def test_a_range_hyphen_with_spaces_on_both_sides_is_read() -> None:
    citation = _first("Doe v. Roe, 80 F.3d 336, 337 - 38 (9th Cir. 1996).")

    assert citation.citation.pin_cite == "337 - 38"


def test_an_en_dash_range_is_read() -> None:
    """Extraction produces both the hyphen and the dash."""
    citation = _first("Kogan v. Facebook, 334 F.R.D. 393, 403–04 (S.D.N.Y. 2020).")

    assert citation.citation.pin_cite == "403–04"


def test_an_ordinary_pin_cite_is_unchanged() -> None:
    """The widening must not change what already worked."""
    citation = _first("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).")

    assert citation.citation.pin_cite == "678"
    assert citation.citation.extra is None


def test_the_page_does_not_leak_into_extra() -> None:
    """`extra` is where a lost pin cite ends up, so it has to be empty here."""
    citation = _first("Bell Atl. Corp. v. Twombly,  550  U.S.  544,  570  (2007).")

    assert not citation.citation.extra


def test_unrelaxed_extraction_still_loses_it() -> None:
    """NONE is eyecite exactly as published, which is what the baseline means.

    Asserted so that relaxing pin cites cannot quietly become unconditional:
    the evaluation's floor arm has to keep measuring eyecite rather than us.

    The locator is written with single spaces on purpose. Doubling those as
    well would lose the whole citation at NONE, and then this would be testing
    the reporter joins rather than the pin cite.
    """
    citation = _first(
        "Bell Atl. Corp. v. Twombly, 550 U.S. 544,  570 (2007).",
        relaxation=Relaxation.NONE,
    )

    assert citation.citation.pin_cite is None
    assert citation.citation.extra == "570"


def test_the_patch_is_restored_after_extraction() -> None:
    """The widening is a swap of module state, so it must not outlive the call."""
    import eyecite.helpers
    import eyecite.regexes

    before = (eyecite.regexes.PIN_CITE_REGEX, eyecite.helpers.POST_FULL_CITATION_REGEX)
    _first("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).")

    assert (eyecite.regexes.PIN_CITE_REGEX, eyecite.helpers.POST_FULL_CITATION_REGEX) == before
