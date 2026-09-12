"""Tests for reading a pin cite through the whitespace extraction leaves.

Losing a pin cite is not a cosmetic failure. It is the page a filing argues
from, so a citation without one carries no checkable claim about what the case
says -- and eyecite does not report the loss, it files the page under `extra`
where nothing looks for it.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.core.citations import FullCaseCitation, IdCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication.candidates.reporter_sites import SuspectedLocator
from mellea_lrc.extraction.adjudication.promotion import reread_site


def _extract(text: str, relaxation: Relaxation = Relaxation.BOUNDED):
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=relaxation)


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


def test_a_re_read_site_reads_its_pin_cite_as_tolerantly_as_extraction() -> None:
    """The re-read paths widen the reporter rule; they must widen this one too.

    A site reaches `reread_site` because the extractor found nothing there, so
    it is damaged text by definition -- and text damaged enough to lose a
    reporter to capitalisation is text whose pin cite is spaced as well.
    Reading the locator and then dropping its page for a doubled space is the
    one outcome nothing wants: eyecite files the page under `extra`, where
    nothing looks for it, and reports no loss.
    """
    text = "See Doe v. Roe, 33 F.4TH 693,  701  (6th Cir. 2022), for the rule."
    start = text.index("F.4TH")
    site = SuspectedLocator(
        span_start=start,
        span_end=start + len("F.4TH"),
        reporter="F.4TH",
        window=text,
    )

    citation = reread_site(text, site)

    assert citation is not None
    assert citation.citation.pin_cite == "701"
    assert citation.pin_cite_span is not None
    assert text[citation.pin_cite_span.start : citation.pin_cite_span.end] == "701"


def test_a_doubled_space_in_a_pin_cite_does_not_strand_the_citation() -> None:
    """Reading the page is not enough if the check that accepts it counts spaces.

    eyecite tests an `Id.`'s page against the citation it would attach to with
    `(?:at )?(\\d+)`, one literal space, so `Id. at  547` fails the test and the
    attribution is thrown away after the widened patterns read it. And because
    an `Id.` following an unresolved `Id.` is refused by rule, one damaged space
    strands the citation after it as well.
    """
    text = (
        "Bell v. Wolfish, 441 U.S. 520, 547 (1979). Officials are accorded deference. "
        "Id. at  547. Judicial deference is highest here. See id. at 548."
    )

    document = _extract(text)
    by_id = {citation.citation_id: citation for citation in document.citations}
    ids = [c for c in document.citations if isinstance(c.citation, IdCitation)]

    assert len(ids) == 2
    for citation in ids:
        root = by_id[citation.root_id]
        assert text[root.locator_span.start : root.locator_span.end] == "441 U.S. 520"


def test_a_pin_cite_outside_the_case_is_still_refused() -> None:
    """The check is right; only what it reads was damaged."""
    text = "Bell v. Wolfish, 441 U.S. 520, 547 (1979). Something else. Id. at  9999."

    document = _extract(text)
    (id_citation,) = [c for c in document.citations if isinstance(c.citation, IdCitation)]

    assert id_citation.root_id is None
