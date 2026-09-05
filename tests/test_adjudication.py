"""Tests for the layer that follows the rules: candidates, and promoting one.

The two properties that matter here are that a generator proposes without
deciding, and that promoting an accepted candidate produces an ordinary
citation rather than a hand-built one -- same fields, same spans into the same
document text.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.extraction.adjudication import reread_site
from mellea_lrc.extraction.adjudication.candidates import (
    SiteStage,
    orphan_short_forms,
    suspected_locators,
)
from mellea_lrc.extraction.adjudication.types import CandidateKind
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

_CAPS = (
    "Table of Authorities - Cross & Cross Props. v. Everett Allied Co., 886 F.2d 497 "
    "(2d Cir. 1989) - Dalla-Longa v. Magnetar Capital LLC , 33 F.4TH 693 (2D CIR. 2022) "
    "- Dow Chem. Pac. Ltd. v. Rascator Maritime S.A., 782 F.2d 329 (2d Cir. 1986)."
)


def _extract(text: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=Relaxation.FULL)


def _site(text: str, reporter: str):
    document = _extract(text)
    return next(site for site in suspected_locators(document) if site.reporter == reporter)


def test_a_reporter_in_capitals_is_proposed() -> None:
    """eyecite's reporter extractors are case-sensitive, so this is not a citation."""
    site = _site(_CAPS, "F.4TH")

    assert site.stage is SiteStage.STRICT
    assert site.matched_reporter == "F.4th"
    assert _CAPS[site.span_start : site.span_end] == "F.4TH"


def test_the_reporters_that_were_read_are_not_proposed_again() -> None:
    """`886 F.2d 497` beside it parses, so nothing should be proposed for it."""
    document = _extract(_CAPS)
    proposed = [_CAPS[s.span_start : s.span_end] for s in suspected_locators(document)]

    assert "F.2d" not in proposed


def test_a_site_needing_only_a_re_read_costs_no_call() -> None:
    """Not a hand-built object: the court and date come from eyecite as usual."""
    citation = reread_site(_CAPS, _site(_CAPS, "F.4TH"))

    assert citation is not None
    assert isinstance(citation.citation, FullCaseCitation)
    assert citation.citation.volume == "33"
    assert citation.citation.reporter.canonical == "F.4th"
    assert citation.citation.page == "693"
    assert citation.citation.date.year == "2022"
    assert citation.citation.court == "ca2"


def test_a_re_read_span_indexes_the_original_document() -> None:
    """Offsets add rather than remap, because the window is a slice of the text."""
    citation = reread_site(_CAPS, _site(_CAPS, "F.4TH"))

    assert _CAPS[citation.locator_span.start : citation.locator_span.end] == "33 F.4TH 693"
    assert citation.matched_text == "33 F.4TH 693"


_OPTICAL = "Compare Smith v. Jones, 930 S0. 2d 128, 131 (Fla. 2006) (per curiam)."
_PUNCTUATION = "See Ashcroft v. Iqbal, 556 U,S, 662, 678 (2009) (pleading standard)."


def test_an_optically_damaged_reporter_is_a_strict_site() -> None:
    """`S0.` is one confusable character away from a spelling the gazetteer holds."""
    site = _site(_OPTICAL, "S0. 2d")

    assert site.stage is SiteStage.STRICT
    assert site.canonical_reporter == "So. 2d"


def test_optical_damage_is_not_something_a_re_read_can_forgive() -> None:
    """A widened rule closes gaps between characters; it cannot put one back."""
    assert reread_site(_OPTICAL, _site(_OPTICAL, "S0. 2d")) is None


def test_a_reporter_the_gazetteer_cannot_spell_is_a_fuzzy_site() -> None:
    """`U,S,` matches no spelling, but reduces to one and has numbers either side."""
    site = _site(_PUNCTUATION, "U,S")

    assert site.stage is SiteStage.FUZZY
    assert site.canonical_reporter == "US"


def test_promoting_a_reviewed_locator_repairs_it_and_parses_the_result() -> None:
    """The reviewer's repaired parts stand in for the damage, then eyecite reads it."""
    from mellea_lrc.core.spans import Span
    from mellea_lrc.extraction.adjudication import promote_locator
    from mellea_lrc.extraction.adjudication.review.locator import AdjudicatedLocator

    start = _PUNCTUATION.index("556 U,S, 662")
    citation = promote_locator(
        _PUNCTUATION,
        AdjudicatedLocator(
            span=Span(start=start, end=start + len("556 U,S, 662")),
            text="556 U,S, 662",
            volume="556",
            reporter="U.S.",
            page="662",
            match_method="exact",
        ),
    )

    assert citation is not None
    assert isinstance(citation.citation, FullCaseCitation)
    assert citation.citation.page == "662"
    assert citation.citation.date.year == "2009"
    # The record points at the characters the document holds, never at a repair.
    assert _PUNCTUATION[citation.locator_span.start : citation.locator_span.end] == "556 U,S, 662"
    assert citation.matched_text == "556 U,S, 662"


def test_a_short_form_with_no_full_citation_is_proposed() -> None:
    """Rule 10.9 allows a short form only after the case is given in full."""
    text = "The court disagreed. DCD Programs , 833 F.2d at 186. That principle applies."
    document = _extract(text)
    candidates = list(orphan_short_forms(document))

    assert len(candidates) == 1
    assert candidates[0].kind is CandidateKind.ORPHAN_SHORT_FORM
    assert "no full citation" in candidates[0].note


def test_a_short_form_with_its_full_citation_is_not_proposed() -> None:
    text = (
        "DCD Programs, Inc. v. Leighton, 833 F.2d 183, 186 (9th Cir. 1987). "
        "Later, DCD Programs , 833 F.2d at 187."
    )
    document = _extract(text)

    assert list(orphan_short_forms(document)) == []


def test_a_generator_decides_nothing() -> None:
    """A site carries a span, a window and a reason -- never a verdict."""
    site = _site(_CAPS, "F.4TH")

    assert not hasattr(site, "verdict")
    assert site.window


def test_an_ambiguous_reporter_is_proposed() -> None:
    """Extraction records the ambiguity; this layer asks someone to settle it."""
    from mellea_lrc.extraction.adjudication.candidates import ambiguous_editions

    text = "Marbury v. Madison, 5 Cranch 137 (1803), established judicial review."
    document = _extract(text)
    candidates = list(ambiguous_editions(document))

    assert len(candidates) == 1
    assert candidates[0].kind is CandidateKind.EDITION
    assert "Cranch" in candidates[0].note


def test_an_unambiguous_reporter_is_not_proposed() -> None:
    from mellea_lrc.extraction.adjudication.candidates import ambiguous_editions

    document = _extract("Doe v. Roe, 695 F.Supp.2d 1149 (D. Colo. 2010).")

    assert list(ambiguous_editions(document)) == []
