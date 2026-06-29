"""Document text windows used to ground field-level assessment values."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mellea_lrc.core.spans import Span

DEFAULT_CONTEXT_CHARS = 200


def is_text_in_context(value: str, document_context: str) -> bool:
    """Return whether a value occurs after whitespace normalization."""
    return " ".join(value.split()) in " ".join(document_context.split())


@dataclass(frozen=True, slots=True)
class GroundedText:
    """Text copied from a document window with absolute document offsets."""

    text: str
    span: Span


@dataclass(frozen=True, slots=True)
class DocumentTextWindow:
    """Local document text retaining its absolute offset and citation anchor."""

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
        """Build a bounded window around an anchored document span."""
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

    def locate(self, value: str) -> GroundedText | None:
        """Locate a value in the window and return exact text with absolute offsets."""
        if not value.strip():
            return None
        matches = tuple(re.finditer(re.escape(value), self.text))
        if not matches:
            pattern = r"\s+".join(re.escape(piece) for piece in value.split())
            matches = tuple(re.finditer(pattern, self.text))
        if not matches:
            return None
        match = min(
            matches,
            key=lambda item: abs((self.span.start + item.start()) - self.anchor_span.start),
        )
        span = Span(
            start=self.span.start + match.start(),
            end=self.span.start + match.end(),
        )
        return GroundedText(text=match.group(0), span=span)


def get_extended_span_text(
    text: str,
    full_span: Span,
    *,
    before_chars: int = DEFAULT_CONTEXT_CHARS,
    after_chars: int = DEFAULT_CONTEXT_CHARS,
) -> str:
    """Return local text around a document span."""
    return DocumentTextWindow.around(
        text,
        full_span,
        before_chars=before_chars,
        after_chars=after_chars,
    ).text


def find_text_span_near_full_span(
    text: str,
    value: str,
    full_span: Span,
    *,
    before_chars: int = DEFAULT_CONTEXT_CHARS,
    after_chars: int = DEFAULT_CONTEXT_CHARS,
) -> Span | None:
    """Locate a grounded value near a citation span."""
    grounded = DocumentTextWindow.around(
        text,
        full_span,
        before_chars=before_chars,
        after_chars=after_chars,
    ).locate(value)
    return grounded.span if grounded is not None else None
