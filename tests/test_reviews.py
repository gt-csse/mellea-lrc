"""What the review interface promises: opt-in, and a relaxation it assumes.

Every review costs a model call, so nothing runs unless a caller names it --
the opposite default from `preprocessing.Rule`, where every rule is free and all
of them run. And where a relaxation is the reason a review is needed, the review
owns it rather than the caller: asking for `PIN_CITE` is asking for the pin-cite
stop to have been relaxed already, and a document read at a narrower level is
refused rather than answered about.
"""

from __future__ import annotations

import asyncio
import contextlib
import io

import pytest

from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication import DEFAULT_REVIEWS, Review, adjudicate

TEXT = "See Ashcroft v. Iqbal , 556 U.S. 662, 678 (2009)."


def _extract(relaxation: Relaxation):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(TEXT, relaxation=relaxation)


def test_nothing_runs_by_default() -> None:
    """A caller that says nothing gets the deterministic reading and no model."""
    assert DEFAULT_REVIEWS == ()
    document = _extract(Relaxation.FULL)
    before = [record.trace for record in document.citations]

    asyncio.run(adjudicate(document, session=None))

    assert [record.trace for record in document.citations] == before


def test_a_review_names_the_relaxation_it_assumes() -> None:
    assert Review.PIN_CITE.requires is Relaxation.FULL
    assert Review.CASE_NAME.requires is Relaxation.BOUNDED


def test_a_document_read_too_strictly_is_refused() -> None:
    """Answering about text the rules never saw is worse than refusing."""
    document = _extract(Relaxation.BOUNDED)

    with pytest.raises(ValueError, match="assumes full relaxation"):
        asyncio.run(adjudicate(document, [Review.PIN_CITE], session=None))


def test_the_check_runs_before_any_review_does() -> None:
    """A list with one impossible review makes no model call at all, so a run
    either happens as asked or does not happen."""
    document = _extract(Relaxation.BOUNDED)

    with pytest.raises(ValueError, match="assumes full relaxation"):
        asyncio.run(adjudicate(document, [Review.CASE_NAME, Review.PIN_CITE], session=None))

    assert all(not record.trace for record in document.citations)
