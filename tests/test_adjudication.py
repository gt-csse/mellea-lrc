"""Tests for the layer that follows the rules: candidates, and promoting one.

The two properties that matter here are that a generator proposes without
deciding, and that promoting an accepted candidate produces an ordinary
citation rather than a hand-built one -- same fields, same spans into the same
document text.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.adjudication import promote
from mellea_lrc.adjudication.candidates import orphan_short_forms, uppercase_reporters
from mellea_lrc.adjudication.types import CandidateKind
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


def test_a_reporter_in_capitals_is_proposed() -> None:
    """eyecite's reporter extractors are case-sensitive, so this is not a citation."""
    document = _extract(_CAPS)
    candidates = list(uppercase_reporters(document))

    assert len(candidates) == 1
    assert candidates[0].kind is CandidateKind.LOCATOR
    assert _CAPS[candidates[0].span.start : candidates[0].span.end] == "33 F.4TH 693"


def test_the_reporters_that_were_read_are_not_proposed_again() -> None:
    """`886 F.2d 497` beside it parses, so nothing should be proposed for it."""
    document = _extract(_CAPS)
    proposed = [_CAPS[c.span.start : c.span.end] for c in uppercase_reporters(document)]

    assert "886 F.2d 497" not in proposed


def test_promoting_a_candidate_reads_it_with_the_real_pipeline() -> None:
    """Not a hand-built object: the court and date come from eyecite as usual."""
    document = _extract(_CAPS)
    candidate = next(iter(uppercase_reporters(document)))
    citation = promote(_CAPS, candidate)

    assert citation is not None
    assert isinstance(citation.citation, FullCaseCitation)
    assert citation.citation.volume == "33"
    assert citation.citation.reporter.canonical == "F.4th"
    assert citation.citation.page == "693"
    assert citation.citation.date.year == "2022"
    assert citation.citation.court == "ca2"


def test_a_promoted_span_indexes_the_original_document() -> None:
    """Offsets add rather than remap, because the window is a slice of the text."""
    document = _extract(_CAPS)
    candidate = next(iter(uppercase_reporters(document)))
    citation = promote(_CAPS, candidate)

    assert _CAPS[citation.locator_span.start : citation.locator_span.end] == "33 F.4TH 693"
    assert citation.matched_text == "33 F.4TH 693"


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
    """A candidate carries a span, a window and a reason -- never a verdict."""
    document = _extract(_CAPS)
    candidate = next(iter(uppercase_reporters(document)))

    assert not hasattr(candidate, "verdict")
    assert candidate.generator == "uppercase_reporters"
    assert candidate.window.start <= candidate.span.start
    assert candidate.window.end >= candidate.span.end
