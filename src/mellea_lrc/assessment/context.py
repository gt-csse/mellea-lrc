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
    """Text copied from a document window with global-document offsets."""

    text: str
    global_span: Span


@dataclass(frozen=True, slots=True)
class DocumentTextWindow:
    """Local document text with explicit global/window offset conversion."""

    text: str
    global_span: Span
    anchor_global_span: Span

    @classmethod
    def around(
        cls,
        document_text: str,
        anchor_global_span: Span,
        *,
        before_chars: int = DEFAULT_CONTEXT_CHARS,
        after_chars: int = DEFAULT_CONTEXT_CHARS,
    ) -> DocumentTextWindow:
        """Build a bounded window around an anchored document span."""
        if anchor_global_span.end > len(document_text):
            msg = f"Span end {anchor_global_span.end} exceeds text length {len(document_text)}"
            raise ValueError(msg)
        if before_chars < 0 or after_chars < 0:
            msg = "before_chars and after_chars must be non-negative"
            raise ValueError(msg)
        global_start = max(0, anchor_global_span.start - before_chars)
        global_end = min(len(document_text), anchor_global_span.end + after_chars)
        return cls(
            text=document_text[global_start:global_end],
            global_span=Span(start=global_start, end=global_end),
            anchor_global_span=anchor_global_span,
        )

    def window_offset_to_global(self, window_offset: int) -> int:
        """Convert a local ``text`` offset to its offset in the full document."""
        if not 0 <= window_offset <= len(self.text):
            msg = f"Window offset {window_offset} is outside 0..{len(self.text)}"
            raise ValueError(msg)
        return self.global_span.start + window_offset

    def global_offset_to_window(self, global_offset: int) -> int:
        """Convert a full-document offset to an offset in local ``text``."""
        if not self.global_span.start <= global_offset <= self.global_span.end:
            msg = f"Global offset {global_offset} is outside window {self.global_span}"
            raise ValueError(msg)
        return global_offset - self.global_span.start

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
            key=lambda item: abs(
                self.window_offset_to_global(item.start()) - self.anchor_global_span.start
            ),
        )
        global_span = Span(
            start=self.window_offset_to_global(match.start()),
            end=self.window_offset_to_global(match.end()),
        )
        return GroundedText(text=match.group(0), global_span=global_span)


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
    return grounded.global_span if grounded is not None else None
