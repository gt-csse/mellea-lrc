"""Tests for reading a pin cite through the whitespace extraction leaves.

Losing a pin cite is not a cosmetic failure. It is the page a filing argues
from, so a citation without one carries no checkable claim about what the case
says -- and eyecite does not report the loss, it files the page under `extra`
where nothing looks for it.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace

import pytest

from mellea_lrc.extraction.adjudication.candidates.reporter_sites import SuspectedLocator
from mellea_lrc.extraction.adjudication.promotion import reread_site
from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text
from mellea_lrc.extraction.stages import resolve_pin_cites
from mellea_lrc.model.citations import CitationField, FullCaseCitation, IdCitation, ShortCaseCitation
from mellea_lrc.model.extraction_metadata import Relaxation
from mellea_lrc.model.operations import update_field
from mellea_lrc.model.record import Node, Reads


def _extract(text: str, relaxation: Relaxation = Relaxation.BOUNDED):
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=relaxation, with_leaves=True)


def _first(text: str, relaxation: Relaxation = Relaxation.BOUNDED):
    # eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=relaxation)
    return next(c for c in document.citations if isinstance(c.fields, FullCaseCitation))


def test_a_doubled_space_before_the_pin_cite_is_read() -> None:
    """`544,  570` is what justified text leaves behind, and it lost the page."""
    citation = _first("Bell Atl. Corp. v. Twombly,  550  U.S.  544,  570  (2007).")

    assert citation.fields.pin_cite.text == "570"


def test_a_spaced_range_hyphen_is_read() -> None:
    """`998 -1003` is a page range whose hyphen extraction has spaced.

    41 of the 42 pin cites that survived the space widening on
    false-citation-bench were this one shape.
    """
    citation = _first("Doe v. Roe, 899 F.3d 988, 998 -1003 (9th Cir. 2018).")

    # Kept as written, spacing and all. The widening decides what parses, not
    # how it is spelled, and a consumer comparing pages reads the first number.
    assert citation.fields.pin_cite.text == "998 -1003"


def test_a_range_hyphen_with_spaces_on_both_sides_is_read() -> None:
    citation = _first("Doe v. Roe, 80 F.3d 336, 337 - 38 (9th Cir. 1996).")

    assert citation.fields.pin_cite.text == "337 - 38"


def test_an_en_dash_range_is_read() -> None:
    """Extraction produces both the hyphen and the dash."""
    citation = _first("Kogan v. Facebook, 334 F.R.D. 393, 403–04 (S.D.N.Y. 2020).")

    assert citation.fields.pin_cite.text == "403–04"


def test_an_ordinary_pin_cite_is_unchanged() -> None:
    """The widening must not change what already worked."""
    citation = _first("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).")

    assert citation.fields.pin_cite.text == "678"
    assert citation.fields.extra is None


def test_pin_cite_stage_structures_the_written_text_against_final_spans() -> None:
    document = _extract("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).")
    target = next(item for item in document.citations if isinstance(item.fields, FullCaseCitation))
    assert target.field_updates[-1].field is CitationField.PIN_CITE
    raw_record = replace(target)
    update_field(
        raw_record,
        Node("raw_pin_cite", Reads.DOCUMENT, "pin_cite", "test", "unstructured"),
        CitationField.PIN_CITE,
        "678",
        reason="Restore the unstructured written page for this stage test.",
    )
    raw_document = document.evolve(
        citations=tuple(
            raw_record if item.citation_id == target.citation_id else item for item in document.citations
        ),
    )

    resolved = resolve_pin_cites(raw_document)
    citation = next(item for item in resolved.citations if item.citation_id == target.citation_id)

    assert citation.fields.pin_cite.text == "678"
    assert resolved.text[citation.pin_cite_span.start : citation.pin_cite_span.end] == "678"


def test_the_page_does_not_leak_into_extra() -> None:
    """`extra` is where a lost pin cite ends up, so it has to be empty here."""
    citation = _first("Bell Atl. Corp. v. Twombly,  550  U.S.  544,  570  (2007).")

    assert not citation.fields.extra


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

    assert citation.fields.pin_cite is None
    assert citation.fields.extra == "570"


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
    assert citation.fields.pin_cite.text == "701"
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
    ids = [c for c in document.citations if isinstance(c.fields, IdCitation)]

    assert len(ids) == 2
    for citation in ids:
        root = by_id[citation.root_id]
        assert text[root.locator_span.start : root.locator_span.end] == "441 U.S. 520"


def test_a_pin_cite_outside_the_case_is_still_refused() -> None:
    """The check is right; only what it reads was damaged.

    A refused `Id.` is now an `Id.` that is never grown: a leaf reaches its root
    or it does not exist, so "attributed to nothing" and "not there" are the
    same state. See `docs/Extraction.md`, "Roots first, leaves after
    validation".
    """
    text = "Bell v. Wolfish, 441 U.S. 520, 547 (1979). Something else. Id. at  9999."

    document = _extract(text)

    assert not [c for c in document.citations if isinstance(c.fields, IdCitation)]


def test_a_short_forms_page_is_its_pin_cite_when_the_pattern_after_it_fails() -> None:
    """`645 B.R. at 184 (quoting …)` lost its page outright.

    `quoting` and `citing` are citation signals, and one of them after the page
    makes eyecite's post-citation pattern fail: `metadata.pin_cite` comes back
    `None`, the span stops at `at `, and the page is in no field a reader looks
    at. It is still in `groups["page"]`, which is where the working path reads
    it from too.
    """
    document = _extract(
        "In re Andrade Gutierrez Engenharia S.A. , 645 B.R. 175 (Bankr. S.D.N.Y. 2022). "
        "Andrade Gutierrez , 645 B.R. at 184 (quoting H.R. Rep. No. 109-31)."
    )
    citation = next(c for c in document.citations if isinstance(c.fields, ShortCaseCitation))

    assert citation.fields.pin_cite.text == "184"
    assert document.text[citation.locator_span.start : citation.locator_span.end] == "645 B.R. at 184"
    assert document.text[citation.pin_cite_span.start : citation.pin_cite_span.end] == "184"


def test_a_footnote_after_a_page_is_read_with_it() -> None:
    """Rule 3.2(b) writes the page, then the footnote on it, with no comma between.

    eyecite requires a comma before any page after the first, so `570 n.4` read
    as the page alone and the footnote was lost. The widened pattern accepts a
    space in place of that comma when a note label follows it.
    """
    document = _extract("Bell Atl. Corp. v. Twombly , 550 U.S. 544 (2007). Twombly , 550 U.S. at 570 n.4.")
    citation = next(c for c in document.citations if isinstance(c.fields, ShortCaseCitation))

    assert citation.fields.pin_cite.text == "570 n.4"
    assert document.text[citation.pin_cite_span.start : citation.pin_cite_span.end] == "570 n.4"
    pages = citation.pin_cite_pages
    assert [(page.first, page.last, page.kind.value, page.footnote) for page in pages] == [
        (570, 570, "page", "4")
    ]


def test_a_page_after_a_page_still_needs_its_comma() -> None:
    """The widening is scoped to a note label, so damage is not read as a page."""
    document = _extract("Bell Atl. Corp. v. Twombly , 550 U.S. 544 (2007). Twombly , 550 U.S. at 570 2007.")
    citation = next(c for c in document.citations if isinstance(c.fields, ShortCaseCitation))

    assert citation.fields.pin_cite.text == "570"


def test_a_short_forms_range_still_reads_whole_when_nothing_follows_it() -> None:
    """The page is taken only as a fallback, so a parsed range is untouched."""
    document = _extract(
        "Does I thru XXIII v. Advanced Textile Corp. , 214 F.3d 1058 (9th Cir. 2000). "
        "Advanced Textile , 214 F.3d at 1068, 1071 -72."
    )
    citation = next(c for c in document.citations if isinstance(c.fields, ShortCaseCitation))

    assert citation.fields.pin_cite.text == "1068, 1071 -72"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Foman v. Davis , 371 U.S. 178,\n\n182 (1962). There is no", "182"),
        ("Tucker v. Fischbein , 237 F.3d 275,\n\n281 -82 (3d Cir. 2001); x", "281 -82"),
        ("Goel v. Bunge , 820 F.3d 400 (2d Cir. 2016). Id. at 409-\n\n12. If even", "409-\n\n12"),
    ],
)
def test_full_reads_a_pin_cite_across_a_blank_line(text: str, expected: str) -> None:
    """`match_on_tokens` stops at a paragraph token, so widening the pattern is
    not enough on its own. At FULL the scan is tried again with the break
    flattened, and `371 U.S. 178,\\n\\n182 (1962)` keeps its page."""
    citations = _extract(text, Relaxation.FULL).citations
    spans = [c.pin_cite_span for c in citations if c.pin_cite_span]

    assert [text[s.start : s.end] for s in spans] == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Vieux Carre , 875 F.2d 800 (5th Cir. 1989). Id. at 809\n\nThe Murphy Order is not final",
        "accord Marler v. Hiebert , 960 F.Supp. 253, 254\n\n- (D. Kan. 1997) (emphasis added)",
    ],
)
def test_the_wider_scan_never_takes_a_page_away(text: str) -> None:
    """Where the pin cite ends at the break, what follows is the next sentence
    or the margin of pleading paper. The strict scan runs first and its answer
    stands, so reading past the break can add a page and never remove one."""
    at = {
        r: [c.pin_cite_span for c in _extract(text, r).citations if c.pin_cite_span]
        for r in (Relaxation.BOUNDED, Relaxation.FULL)
    }

    assert at[Relaxation.FULL] == at[Relaxation.BOUNDED]
    assert at[Relaxation.FULL]
