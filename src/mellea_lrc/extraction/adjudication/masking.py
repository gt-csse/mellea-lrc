"""Offset-preserving masking of extracted citations.

Both functions return text of exactly the same length as the input, so every
span into the original document stays valid. That is what lets a second pass
scan the masked text and report offsets that still index the real document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import ExtractedDocument


def mask_locator_spans(document: ExtractedDocument) -> str:
    """Blank every extracted locator, leaving its surrounding citation intact.

    Hides what the extractor identified while keeping case names and
    parentheticals, so a second pass sees the context but not the answer.
    """
    characters = list(document.text)
    for citation in document.citations:
        locator = citation.locator_span
        characters[locator.start : locator.end] = " " * (locator.end - locator.start)
    return "".join(characters)


def mask_full_spans(document: ExtractedDocument) -> str:
    """Blank each extracted citation entirely: name, locator and parenthetical.

    Prefer this for candidate generation. Court abbreviations inside
    parentheticals -- ``(D. Ariz. 2017)`` -- are themselves reporter strings in
    eyecite's gazetteer, and they sit outside the locator span but inside the
    full span. Masking only locators leaves them exposed as candidates; masking
    full spans removes them.
    """
    characters = list(document.text)
    for citation in document.citations:
        span = citation.span
        characters[span.start : span.end] = " " * (span.end - span.start)
    return "".join(characters)
