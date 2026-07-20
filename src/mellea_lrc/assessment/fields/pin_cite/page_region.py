"""Resolve reporter pin cites to bounded pages in a retrieved opinion PDF.

The resolver deliberately does no proposition-support inference. Its only job
is to establish a page-faithful evidence region that a later LLM node may read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from mellea_lrc.core.citations import FullCaseCitation

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationRecord
    from mellea_lrc.extraction.types import ExtractedCitation

COURTLISTENER_STORAGE_BASE_URL = "https://storage.courtlistener.com/"
_PIN_PAGE = re.compile(r"(?<![\d*])(\d[\d,]*)(?:\s*[-\u2013\u2014]\s*(\d[\d,]*))?")


class PinCitePageRegionStatus(str, Enum):
    """Outcome of resolving a pin cite against a page-faithful PDF."""

    RESOLVED = "resolved"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED_PIN_CITE = "unsupported_pin_cite"
    UNVERIFIED_PDF = "unverified_pdf"
    OUT_OF_RANGE = "out_of_range"
    TEXT_UNAVAILABLE = "text_unavailable"


@dataclass(frozen=True, slots=True)
class ReporterPage:
    """One reporter page mapped to a one-based physical PDF page."""

    reporter_page_number: int
    pdf_page_number: int
    text: str
    printed_page_label_observed: bool


@dataclass(frozen=True, slots=True)
class PinCitePageRegion:
    """Bounded page evidence for one upstream citation pin."""

    status: PinCitePageRegionStatus
    pin_cite: str | None
    reporter_page_start: int | None = None
    reporter_page_end: int | None = None
    reporter_base_pdf_page: int | None = None
    pdf_page_start: int | None = None
    pdf_page_end: int | None = None
    pages: tuple[ReporterPage, ...] = ()
    mapping_basis: str | None = None
    message: str = ""

    @property
    def text(self) -> str:
        """Join only the resolved pin-cited pages for downstream inference."""
        return "\n\n".join(page.text for page in self.pages)


def courtlistener_opinion_pdf_url(record: CourtListenerCitationRecord) -> str | None:
    """Return the best public CourtListener-hosted opinion PDF URL."""
    extra = record.extra_data.to_dict()
    for field in ("filepath_pdf_harvard", "filepath_pdf_scan"):
        value = extra.get(field)
        if isinstance(value, str) and value.strip():
            return urljoin(COURTLISTENER_STORAGE_BASE_URL, value.lstrip("/"))
    return None


def resolve_pin_cite_page_region(
    citation: ExtractedCitation,
    opinion_pdf: bytes,
    *,
    extract_pages: Callable[[bytes], tuple[str, ...]] | None = None,
) -> PinCitePageRegion:
    """Map a reporter pin cite to exact physical pages in an opinion PDF.

    Harvard reporter PDFs retain one physical PDF page per reporter page, but a
    file may begin before the cited case. We accept page arithmetic only after
    locating a physical page whose first printed line is the citation's reporter
    start-page label. This is a page-header anchor check, not a whole-document
    substring search. PDFs paginated only in a parallel reporter remain
    unverified.
    """
    canonical = citation.citation
    if not isinstance(canonical, FullCaseCitation):
        return _unresolved(
            PinCitePageRegionStatus.NOT_APPLICABLE,
            pin_cite=getattr(canonical, "pin_cite", None),
            message="pin-cite page resolution requires a full case citation",
        )
    pin_cite = canonical.pin_cite
    if not pin_cite:
        return _unresolved(
            PinCitePageRegionStatus.NOT_APPLICABLE,
            pin_cite=None,
            message="citation does not contain a pin cite",
        )
    base_page = _reporter_page_number(canonical.page)
    pin_range = _reporter_pin_range(pin_cite)
    if base_page is None or pin_range is None:
        return _unresolved(
            PinCitePageRegionStatus.UNSUPPORTED_PIN_CITE,
            pin_cite=pin_cite,
            message="pin cite cannot be mapped by reporter-page arithmetic",
        )
    reporter_start, reporter_end = pin_range
    if reporter_start < base_page:
        return _unresolved(
            PinCitePageRegionStatus.UNSUPPORTED_PIN_CITE,
            pin_cite=pin_cite,
            message="pin cite precedes the citation's reporter start page",
        )

    page_extractor = extract_pages or _extract_pdf_pages
    try:
        physical_pages = page_extractor(opinion_pdf)
    except Exception as exc:
        return _unresolved(
            PinCitePageRegionStatus.TEXT_UNAVAILABLE,
            pin_cite=pin_cite,
            message=f"opinion PDF text extraction failed: {type(exc).__name__}: {exc}",
        )
    if not physical_pages or not any(page.strip() for page in physical_pages):
        return _unresolved(
            PinCitePageRegionStatus.TEXT_UNAVAILABLE,
            pin_cite=pin_cite,
            message="opinion PDF has no extractable page text; OCR fallback is required",
        )
    base_pdf_index = next(
        (
            index
            for index, page_text in enumerate(physical_pages)
            if _page_begins_with_label(page_text, base_page)
        ),
        None,
    )
    if base_pdf_index is None:
        return _unresolved(
            PinCitePageRegionStatus.UNVERIFIED_PDF,
            pin_cite=pin_cite,
            message=(
                "no physical PDF page begins with the citation's reporter start page; "
                "the PDF may use parallel-reporter pagination"
            ),
        )

    base_pdf_page = base_pdf_index + 1
    pdf_start = base_pdf_page + reporter_start - base_page
    pdf_end = base_pdf_page + reporter_end - base_page
    if pdf_end > len(physical_pages):
        return _unresolved(
            PinCitePageRegionStatus.OUT_OF_RANGE,
            pin_cite=pin_cite,
            message=(
                f"pin cite maps to PDF pages {pdf_start}-{pdf_end}, but the PDF has "
                f"{len(physical_pages)} pages"
            ),
        )

    pages = tuple(
        ReporterPage(
            reporter_page_number=reporter_page,
            pdf_page_number=base_pdf_page + reporter_page - base_page,
            text=physical_pages[base_pdf_index + reporter_page - base_page],
            printed_page_label_observed=_page_begins_with_label(
                physical_pages[base_pdf_index + reporter_page - base_page], reporter_page
            ),
        )
        for reporter_page in range(reporter_start, reporter_end + 1)
    )
    if any(not page.text.strip() for page in pages):
        return _unresolved(
            PinCitePageRegionStatus.TEXT_UNAVAILABLE,
            pin_cite=pin_cite,
            message="one or more pin-cited PDF pages have no extractable text; OCR fallback is required",
        )
    return PinCitePageRegion(
        status=PinCitePageRegionStatus.RESOLVED,
        pin_cite=pin_cite,
        reporter_page_start=reporter_start,
        reporter_page_end=reporter_end,
        reporter_base_pdf_page=base_pdf_page,
        pdf_page_start=pdf_start,
        pdf_page_end=pdf_end,
        pages=pages,
        mapping_basis="located_reporter_start_page_plus_physical_page_offset",
        message="pin cite resolved to a bounded reporter-page region",
    )


def _extract_pdf_pages(opinion_pdf: bytes) -> tuple[str, ...]:
    from pypdfium2 import PdfDocument  # noqa: PLC0415

    document = PdfDocument(BytesIO(opinion_pdf))
    try:
        return tuple(page.get_textpage().get_text_range() for page in document)
    finally:
        document.close()


def _reporter_page_number(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.replace(",", "").strip()
    return int(normalized) if normalized.isdigit() else None


def _reporter_pin_range(pin_cite: str) -> tuple[int, int] | None:
    if "*" in pin_cite:
        return None
    match = _PIN_PAGE.search(pin_cite)
    if match is None:
        return None
    start_text = match.group(1).replace(",", "")
    end_text = match.group(2)
    start = int(start_text)
    if end_text is None:
        return (start, start)
    normalized_end = end_text.replace(",", "")
    if len(normalized_end) >= len(start_text):
        end = int(normalized_end)
    else:
        magnitude = 10 ** len(normalized_end)
        end = (start // magnitude) * magnitude + int(normalized_end)
        if end < start:
            end += magnitude
    return (start, end) if end >= start else None


def _page_begins_with_label(page_text: str, reporter_page: int) -> bool:
    first_line = next((line.strip() for line in page_text.splitlines() if line.strip()), "")
    return first_line == str(reporter_page)


def _unresolved(
    status: PinCitePageRegionStatus,
    *,
    pin_cite: str | None,
    message: str,
) -> PinCitePageRegion:
    return PinCitePageRegion(status=status, pin_cite=pin_cite, message=message)
