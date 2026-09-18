"""Docket-entry attachment is local to an explicitly written case locator."""

from __future__ import annotations

from mellea_lrc.core.citations import DocketCitation, DocketEntry
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import grow_roots, stable
from mellea_lrc.extraction.adjudication.candidates.docket_sites import SuspectedDocket
from mellea_lrc.extraction.adjudication.promotion import promote_docket_locator
from mellea_lrc.extraction.adjudication.review.docket import RecoveredDocketLocator
from mellea_lrc.preprocessing import preprocess


def _docket(text: str) -> DocketCitation:
    document = grow_roots(preprocess(text), rules=stable())
    (record,) = [record for record in document.citations if isinstance(record.source, DocketCitation)]
    return record.source


def test_regular_reader_relaxes_entry_whitespace_inside_the_adjacency_limit() -> None:
    text = "See Doc.\t10-1,   Case  No. 1:25-cv-00312-RPK (E.D.N.Y. 2025)."

    docket = _docket(text)

    assert docket.docket_entry == DocketEntry(
        number="10-1",
        span=Span(text.index("Doc."), text.index("Doc.") + len("Doc.\t10-1")),
    )


def test_regular_reader_does_not_attach_an_entry_beyond_the_adjacency_limit() -> None:
    text = "See Doc. 75" + " " * 13 + "Case No. 1:25-cv-00312-RPK (E.D.N.Y. 2025)."

    assert _docket(text).docket_entry is None


def test_site_promotion_uses_the_same_adjacent_entry_reader() -> None:
    text = "See Doc. 10-1, No. 21-11854 (Bankr. S.D.N.Y. 2021)."
    locator = "No. 21-11854"
    start = text.index(locator)
    site = SuspectedDocket(
        locator_span=Span(start, start + len(locator)),
        locator_text=locator,
        docket_number="21-11854",
        context_span=Span(0, len(text)),
        context=text,
    )
    record = promote_docket_locator(
        text,
        site,
        RecoveredDocketLocator(locator, "21-11854", "A cited bankruptcy docket."),
    )

    assert record.source == DocketCitation(
        span=Span(text.index("Doc."), start + len(locator)),
        locator_span=site.locator_span,
        matched_text=locator,
        docket_number="21-11854",
        docket_entry=DocketEntry("10-1", Span(text.index("Doc."), text.index("Doc.") + len("Doc. 10-1"))),
    )
