"""Local document context for validation operations."""

from __future__ import annotations

from dataclasses import dataclass

from mellea_lrc.core.spans import Span

DEFAULT_CONTEXT_CHARS = 200


@dataclass(frozen=True, slots=True)
class DocumentTextWindow:
    """Bounded document text retaining the citation anchor."""

    text: str
    span: Span
    anchor_span: Span

    @classmethod
    def around(
        cls,
        document_text: str,
        anchor_span: Span,
        *,
        before_chars: int = DEFAULT_CONTEXT_CHARS,
        after_chars: int = DEFAULT_CONTEXT_CHARS,
    ) -> DocumentTextWindow:
        """Build a bounded text window around a citation."""
        if anchor_span.end > len(document_text):
            msg = f"Span end {anchor_span.end} exceeds text length {len(document_text)}"
            raise ValueError(msg)
        if before_chars < 0 or after_chars < 0:
            msg = "before_chars and after_chars must be non-negative"
            raise ValueError(msg)
        start = max(0, anchor_span.start - before_chars)
        end = min(len(document_text), anchor_span.end + after_chars)
        return cls(
            text=document_text[start:end],
            span=Span(start=start, end=end),
            anchor_span=anchor_span,
        )
