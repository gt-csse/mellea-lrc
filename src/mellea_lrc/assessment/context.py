"""Context helpers for Mellea-assisted assessment."""

from mellea_lrc.core.spans import Span

DEFAULT_CONTEXT_CHARS = 240


def get_extended_span_text(
    text: str,
    full_span: Span,
    *,
    before_chars: int = DEFAULT_CONTEXT_CHARS,
    after_chars: int = DEFAULT_CONTEXT_CHARS,
) -> str:
    """Return text around a full citation span for semantic assessment."""
    if full_span.end > len(text):
        msg = f"Span end {full_span.end} exceeds text length {len(text)}"
        raise ValueError(msg)
    if before_chars < 0 or after_chars < 0:
        msg = "before_chars and after_chars must be non-negative"
        raise ValueError(msg)

    start = max(0, full_span.start - before_chars)
    end = min(len(text), full_span.end + after_chars)
    return text[start:end]
