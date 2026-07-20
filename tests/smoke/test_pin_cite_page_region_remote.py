"""Real CourtListener PDF sanity tests for reporter-page pin cites."""

# ruff: noqa: INP001

from urllib.request import urlopen

import pytest

from mellea_lrc.assessment.fields.pin_cite import (
    PinCitePageRegionStatus,
    resolve_pin_cite_page_region,
)
from mellea_lrc.core.citations import FullCaseCitation, Reporter
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import ExtractedCitation

pytestmark = pytest.mark.remote_smoke


@pytest.mark.parametrize(
    ("locator", "page", "pin_cite", "pdf_url", "expected_pdf_page", "expected_text"),
    [
        (
            "162 F.R.D. 418",
            "418",
            "422",
            "https://storage.courtlistener.com/harvard_pdf/9046808.pdf",
            5,
            "request for anonymity is effectively moot",
        ),
        (
            "307 F.R.D. 1",
            "1",
            "7",
            "https://storage.courtlistener.com/harvard_pdf/2731684.pdf",
            7,
            "this factor weighs",
        ),
        (
            "5 Cal. App. 5th 1069",
            "1069",
            "1091",
            "https://storage.courtlistener.com/harvard_pdf/4324089.pdf",
            37,
            "render the hearing unfair",
        ),
        (
            "556 U.S. 662",
            "662",
            "678",
            "https://storage.courtlistener.com/harvard_pdf/145875.pdf",
            17,
            "plausible on its face",
        ),
    ],
)
def test_real_courtlistener_reporter_pdf_resolves_exact_page(
    locator: str,
    page: str,
    pin_cite: str,
    pdf_url: str,
    expected_pdf_page: int,
    expected_text: str,
    remote_timeout: float,
) -> None:
    """Resolve real retrieved opinions without searching the whole document."""
    with urlopen(pdf_url, timeout=remote_timeout) as response:  # noqa: S310 - fixed HTTPS fixtures
        opinion_pdf = response.read()
    matched = f"{locator}, {pin_cite}"
    citation = ExtractedCitation(
        citation_id="citation-1",
        citation_span=Span(0, len(matched)),
        matched_locator_text=locator,
        matched_citation_text=matched,
        citation=FullCaseCitation(
            page=page,
            pin_cite=pin_cite,
            reporter=Reporter(edition_short_name=locator.split()[1]),
        ),
    )

    result = resolve_pin_cite_page_region(citation, opinion_pdf)

    assert result.status is PinCitePageRegionStatus.RESOLVED, result.message
    assert result.pdf_page_start == expected_pdf_page
    assert expected_text.lower() in result.text.lower()
