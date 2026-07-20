"""Tests for page-grounded pin-cite resolution."""

# ruff: noqa: D103, INP001

from mellea_lrc.assessment.fields.pin_cite import (
    PinCitePageRegionStatus,
    courtlistener_opinion_pdf_url,
    resolve_pin_cite_page_region,
)
from mellea_lrc.core.citations import FullCaseCitation, Reporter
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationRecord
from mellea_lrc.extraction.types import ExtractedCitation


def _citation(*, page: str = "418", pin_cite: str | None = "422") -> ExtractedCitation:
    matched = f"162 F.R.D. {page}, {pin_cite}" if pin_cite else f"162 F.R.D. {page}"
    return ExtractedCitation(
        citation_id="citation-1",
        citation_span=Span(0, len(matched)),
        matched_locator_text=f"162 F.R.D. {page}",
        matched_citation_text=matched,
        citation=FullCaseCitation(
            volume="162",
            page=page,
            pin_cite=pin_cite,
            reporter=Reporter(edition_short_name="F.R.D."),
        ),
    )


def test_resolves_reporter_pin_by_verified_page_offset() -> None:
    expected_reporter_page = 422
    expected_pdf_page = 5
    pages = tuple(f"{number}\nText from reporter page {number}." for number in range(418, 423))

    result = resolve_pin_cite_page_region(
        _citation(),
        b"pdf",
        extract_pages=lambda _: pages,
    )

    assert result.status is PinCitePageRegionStatus.RESOLVED
    assert result.reporter_page_start == expected_reporter_page
    assert result.reporter_base_pdf_page == 1
    assert result.pdf_page_start == expected_pdf_page
    assert result.pages[0].text == "422\nText from reporter page 422."
    assert result.pages[0].printed_page_label_observed


def test_resolves_abbreviated_reporter_page_range() -> None:
    pages = tuple(f"{number}\nPage {number}." for number in range(460, 465))

    result = resolve_pin_cite_page_region(
        _citation(page="460", pin_cite="463-64"),
        b"pdf",
        extract_pages=lambda _: pages,
    )

    assert result.status is PinCitePageRegionStatus.RESOLVED
    assert (result.reporter_page_start, result.reporter_page_end) == (463, 464)
    assert (result.pdf_page_start, result.pdf_page_end) == (4, 5)
    assert [page.reporter_page_number for page in result.pages] == [463, 464]


def test_rejects_pdf_without_reporter_start_page_anchor() -> None:
    result = resolve_pin_cite_page_region(
        _citation(),
        b"pdf",
        extract_pages=lambda _: ("Cover sheet", "422\nTarget text"),
    )

    assert result.status is PinCitePageRegionStatus.UNVERIFIED_PDF
    assert result.pages == ()


def test_locates_reporter_start_page_after_preceding_reporter_pages() -> None:
    expected_base_pdf_page = 15
    expected_pin_pdf_page = 37
    pages = tuple(f"{number}\nPage {number}." for number in range(1055, 1092))

    result = resolve_pin_cite_page_region(
        _citation(page="1069", pin_cite="1091"),
        b"pdf",
        extract_pages=lambda _: pages,
    )

    assert result.status is PinCitePageRegionStatus.RESOLVED
    assert result.reporter_base_pdf_page == expected_base_pdf_page
    assert result.pdf_page_start == expected_pin_pdf_page
    assert result.pages[0].text == "1091\nPage 1091."


def test_westlaw_star_pin_requires_a_different_page_mapping() -> None:
    result = resolve_pin_cite_page_region(
        _citation(page="461831", pin_cite="at *3"),
        b"pdf",
        extract_pages=lambda _: ("461831\nText",),
    )

    assert result.status is PinCitePageRegionStatus.UNSUPPORTED_PIN_CITE
    assert result.pages == ()


def test_empty_page_text_requests_ocr_fallback() -> None:
    result = resolve_pin_cite_page_region(
        _citation(),
        b"pdf",
        extract_pages=lambda _: ("",),
    )

    assert result.status is PinCitePageRegionStatus.TEXT_UNAVAILABLE
    assert "OCR fallback" in result.message


def test_builds_public_pdf_url_from_retrieved_candidate_metadata() -> None:
    record = CourtListenerCitationRecord(
        extra_data=ExtraData({"filepath_pdf_harvard": "harvard_pdf/9046808.pdf"})
    )

    assert courtlistener_opinion_pdf_url(record) == (
        "https://storage.courtlistener.com/harvard_pdf/9046808.pdf"
    )
