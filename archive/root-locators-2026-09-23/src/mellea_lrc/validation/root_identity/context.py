"""Target-only local context for root-identity repair.

Identity repair may ask an LLM to re-read one root, but it must not let a
neighbouring citation supply the answer.  This module masks other citations
without changing offsets, while retaining the target root's own written
context.  It is deliberately independent of co-location: a model receives no
group identifiers or parallel-citation concept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea_lrc.model.document import Document
    from mellea_lrc.model.record import CitationRecord


@dataclass(frozen=True, slots=True)
class RootContext:
    """One offset-preserving, target-only view of a document interval."""

    text: str
    start: int
    end: int

    def as_document_text(self, *, document_length: int) -> str:
        """Place this local view back in an otherwise blank document.

        Older, locator-scoped readers accept document text and calculate their
        own local offsets.  This adapter lets those readers use the target-only
        window without changing their coordinate contract.
        """
        if self.start < 0 or self.end < self.start or self.end > document_length:
            msg = "Root context falls outside its source document"
            raise ValueError(msg)
        if len(self.text) != self.end - self.start:
            msg = "Root context text length must equal its source interval"
            raise ValueError(msg)
        return " " * self.start + self.text + " " * (document_length - self.end)


def masked_root_context(
    document: Document,
    root: CitationRecord,
    *,
    before: int = 320,
    after: int = 160,
) -> RootContext:
    """Return a local window where only ``root`` remains readable.

    Other full citation spans are blanked first.  The target citation is then
    restored, and every other locator is blanked again.  The second step keeps
    a parallel locator hidden even where its broad full span overlaps the
    target's shared name or parenthetical.  The returned text has the same
    length as its source interval, so every grounding offset stays usable.
    """
    if root.withdrawn:
        msg = "A withdrawn citation cannot be repaired as a root"
        raise ValueError(msg)
    target = root.locator_span
    start = max(0, target.start - before)
    end = min(len(document.text), target.end + after)
    source = document.text[start:end]
    characters = list(source)

    def blank(span_start: int, span_end: int) -> None:
        left = max(start, span_start)
        right = min(end, span_end)
        if left < right:
            characters[left - start : right - start] = " " * (right - left)

    for other in document.active_citations:
        if other is not root:
            blank(other.full_span.start, other.full_span.end)

    # The target's context may overlap an overly broad span assigned to another
    # citation. Its own evidence must always remain available for re-reading.
    characters[target.start - start : target.end - start] = source[target.start - start : target.end - start]
    for other in document.active_citations:
        if other is not root:
            blank(other.locator_span.start, other.locator_span.end)

    return RootContext(text="".join(characters), start=start, end=end)
