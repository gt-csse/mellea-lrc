"""Context helpers for Mellea-assisted assessment."""

import re

from mellea_lrc.core.spans import Span

DEFAULT_CONTEXT_CHARS = 200


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


def find_text_span_near_full_span(
    text: str,
    value: str,
    full_span: Span,
    *,
    before_chars: int = DEFAULT_CONTEXT_CHARS,
    after_chars: int = DEFAULT_CONTEXT_CHARS,
) -> Span | None:
    """Find a grounded value near a citation span and return document offsets."""
    if not value.strip():
        return None
    if full_span.end > len(text):
        msg = f"Span end {full_span.end} exceeds text length {len(text)}"
        raise ValueError(msg)
    if before_chars < 0 or after_chars < 0:
        msg = "before_chars and after_chars must be non-negative"
        raise ValueError(msg)

    window_start = max(0, full_span.start - before_chars)
    window_end = min(len(text), full_span.end + after_chars)
    window = text[window_start:window_end]

    literal_matches = tuple(re.finditer(re.escape(value), window))
    if literal_matches:
        match = min(literal_matches, key=lambda item: abs((window_start + item.start()) - full_span.start))
        start = window_start + match.start()
        return Span(start=start, end=start + len(value))

    pattern = _whitespace_flexible_pattern(value)
    matches = tuple(re.finditer(pattern, window))
    if not matches:
        return None
    match = min(matches, key=lambda item: abs((window_start + item.start()) - full_span.start))
    return Span(start=window_start + match.start(), end=window_start + match.end())


def _whitespace_flexible_pattern(value: str) -> str:
    pieces = [re.escape(piece) for piece in value.split()]
    return r"\s+".join(pieces)
