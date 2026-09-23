"""Reclassify recurring ECF filing stamps as page furniture."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

_MARKS = (
    re.compile(r"\bcase\b[^\n]{0,40}?\d[\d:\-]{3,}", re.IGNORECASE),
    re.compile(r"\bdoc(ument|\.)?\s*#?\s*\d", re.IGNORECASE),
    re.compile(r"\bfiled\b[^\n]{0,20}?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", re.IGNORECASE),
    re.compile(r"\bpage\s*(id)?[\s.#:]*\d+(\s+of\s+\d+)?", re.IGNORECASE),
    re.compile(r"\bpageid\b", re.IGNORECASE),
    re.compile(r"\bentry\s+id\b|\bfiling\s+id\b", re.IGNORECASE),
)
_MIN_MARKS = 3
"""How much of a stamp must be present. Three of six, so no single court's
wording is required and no ordinary sentence reaches it."""

_BAND_TOLERANCE = 3.0
"""How far the top and bottom of the band may move between pages."""

_MIN_PAGES = 2
"""Pages a band must appear on to be furniture, when nothing anchors it."""


def looks_like_a_stamp(text: str) -> bool:
    """Whether this text reads like a court's filing stamp."""
    return sum(1 for mark in _MARKS if mark.search(text or "")) >= _MIN_MARKS


def _band(item: Any) -> tuple[float, float] | None:
    provenance = getattr(item, "prov", None)
    if not provenance:
        return None
    box = provenance[0].bbox
    return (box.t, box.b)


def _close(one: tuple[float, float], other: tuple[float, float]) -> bool:
    return all(abs(a - b) <= _BAND_TOLERANCE for a, b in zip(one, other, strict=True))


def docket_stamps(document: DoclingDocument) -> list[Any]:
    """Body items that read like a filing stamp and sit where furniture sits."""
    from docling_core.types.doc.common.content_layer import ContentLayer

    gated: list[tuple[Any, tuple[float, float], int]] = []
    anchors: list[tuple[float, float]] = []
    for item, _ in document.iterate_items(with_groups=False, included_content_layers=set(ContentLayer)):
        band = _band(item)
        if band is None or not looks_like_a_stamp(getattr(item, "text", "") or ""):
            continue
        if item.content_layer is ContentLayer.FURNITURE:
            anchors.append(band)
        else:
            gated.append((item, band, item.prov[0].page_no))

    pages_by_band: dict[tuple[float, float], set[int]] = defaultdict(set)
    for _, band, page in gated:
        for seen in pages_by_band:
            if _close(band, seen):
                pages_by_band[seen].add(page)
                break
        else:
            pages_by_band[band].add(page)

    found = []
    for item, band, _ in gated:
        anchored = any(_close(band, anchor) for anchor in anchors)
        repeated = any(
            _close(band, seen) and len(pages) >= _MIN_PAGES for seen, pages in pages_by_band.items()
        )
        if anchored or repeated:
            found.append(item)
    return found


def reclassify_docket_stamps(document: DoclingDocument) -> None:
    """Move filing stamps left in the body to furniture, in place."""
    from docling_core.types.doc.common.content_layer import ContentLayer

    for item in docket_stamps(document):
        item.content_layer = ContentLayer.FURNITURE
