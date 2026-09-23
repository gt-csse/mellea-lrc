"""Reclassify repeated page headers and footers as furniture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

FURNITURE_LABELS = frozenset({"page_header", "page_footer"})
# Coordinates of repeated furniture agree to well under a point. Three is loose
# enough for rounding and tight enough that body text cannot reach it.
_BOX_TOLERANCE = 3.0


def reclassify_repeated_furniture(document: DoclingDocument) -> None:
    """Move body items sharing a box with a labelled header or footer to furniture, in place."""
    from docling_core.types.doc.common.content_layer import ContentLayer

    for item in repeated_furniture(document):
        item.content_layer = ContentLayer.FURNITURE


def repeated_furniture(document: DoclingDocument) -> list[Any]:
    """Body items printed where this document has a recognised header or footer."""
    known = [
        _box(item)
        for item in document.texts
        if item.label.value in FURNITURE_LABELS and _box(item) is not None
    ]
    if not known:
        return []

    found = []
    for item in document.texts:
        if item.label.value in FURNITURE_LABELS:
            continue
        box = _box(item)
        if box is not None and any(_same_box(box, reference) for reference in known):
            found.append(item)
    return found


def _box(item: Any) -> tuple[float, float, float, float] | None:
    """The item's bounding box as (left, right, top, bottom)."""
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None
    bbox = provenance[0].bbox
    return (bbox.l, bbox.r, bbox.t, bbox.b)


def _same_box(
    box: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> bool:
    """Whether two boxes describe the same place on the page."""
    return all(abs(a - b) <= _BOX_TOLERANCE for a, b in zip(box, reference, strict=True))
