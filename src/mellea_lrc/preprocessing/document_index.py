"""Locate tables of authorities as spans in the exported text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.model.span import Span

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

INDEX_LABEL = "document_index"


def index_table_spans(document: DoclingDocument) -> tuple[Span, ...]:
    """Return the spans of the exported text occupied by index tables.

    Measured by rendering the document with the index in place and again with it
    out, and taking the difference, rather than by searching the output for a
    table's text: an index entry repeats verbatim in the body of a brief -- that
    is what an index is -- and a search would have no way to tell which
    occurrence it had found.

    **One table at a time, and by prefix and suffix.** Removing a single item
    deletes one contiguous region, so the two renderings share a prefix and a
    suffix and the difference is what lies between. Comparing them with a block
    matcher instead invites it to pair fragments of the removed index against
    the body text those fragments also appear in, which is not a hypothetical:
    an index this returns as one span came back from `difflib` as 175
    two-character ones.

    Each table is toggled against the *unmodified* rendering, so every span is in
    the coordinates of the text the caller will hold, and removing one index does
    not shift the next.

    The document is left as it was found.
    """
    from docling_core.types.doc.common.content_layer import ContentLayer

    tables = [table for table in document.tables if table.label.value == INDEX_LABEL]
    if not tables:
        return ()

    with_index = document.export_to_text()
    spans: list[Span] = []
    for table in tables:
        restore = table.content_layer
        try:
            table.content_layer = ContentLayer.FURNITURE
            without = document.export_to_text()
        finally:
            table.content_layer = restore
        span = _deleted_region(with_index, without)
        if span is not None:
            spans.append(span)
    return _without_overlap(sorted(spans, key=lambda span: span.start), with_index)


def _deleted_region(before: str, after: str) -> Span | None:
    """The one contiguous region of `before` that `after` does not have.

    ``None`` when nothing was removed, which means the table contributed no text
    -- an empty index, or one already outside the body layer.

    The suffix is measured only as far back as the prefix reaches. Without that
    bound the walk runs past the deletion and into text the two renderings share
    on both sides of it, and the region comes back empty or inverted.
    """
    if len(after) >= len(before):
        return None

    limit = len(after)
    prefix = 0
    while prefix < limit and before[prefix] == after[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and before[-1 - suffix] == after[-1 - suffix]:
        suffix += 1
    return _trimmed(before, prefix, len(before) - suffix)


def _without_overlap(spans: list[Span], text: str) -> tuple[Span, ...]:
    """Keep consecutive regions apart.

    Two indexes in a row are serialized alike, down to the pipes and the rule
    row, so the characters joining them belong equally to the end of one and the
    start of the next and both walks claim them. They belong to neither.
    """
    kept: list[Span] = []
    for span in spans:
        start = max(span.start, kept[-1].end) if kept else span.start
        trimmed = _trimmed(text, start, span.end)
        if trimmed is not None:
            kept.append(trimmed)
    return tuple(kept)


def _trimmed(text: str, start: int, end: int) -> Span | None:
    """The region without the blank lines that join it to what surrounds it."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end) if start < end else None
