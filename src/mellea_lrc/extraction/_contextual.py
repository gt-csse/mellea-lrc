"""Shared source-only context parsing for native full-citation fields."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mellea_lrc.model.citation_windows import after
from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.fields.court import court_id_if_unique
from mellea_lrc.model.citations.fields.date import FULL_DATE_RE, MONTH_YEAR_RE, YEAR_RE
from mellea_lrc.model.document import Document

_PARENTHETICAL = re.compile(r"\((?P<body>[^()\r\n]{0,100})\)")
_SENTENCE = re.compile(r"\.\s+[A-Z]")
_BREAK = re.compile(r"\n\s*\n|…|\.{2,}")
_DATE_EVENT = re.compile(r"\s+\b(?:filed|decided|issued)\s*$", re.I)
_COURT_LIKE = re.compile(
    r"(?:\d+(?:st|nd|rd|th)\s+Cir\.?|[A-Za-z.]+\s+Cir\.?|"
    r"(?:[ENSW]\.)?D\.\s+[A-Za-z]+\.?|[A-Za-z.]+\s+(?:Court|Ct\.?|DCA|App\.?(?:\s+Div\.)?))",
    re.I,
)


@dataclass(frozen=True, slots=True)
class ContextParenthetical:
    """One complete parenthetical directly following a locator group."""

    body: str
    body_start: int


def parenthetical_after(
    document: Document, citation: FullCitationVariant, limit: int
) -> ContextParenthetical | None:
    """Return the first complete post-group parenthetical before new prose.

    A missing or truncated parenthetical is not evidence that a field was not
    stated, so callers receive ``None`` and leave the field unread.
    """
    text, offset = after(document, citation, limit)
    for match in _PARENTHETICAL.finditer(text):
        if _SENTENCE.search(text[: match.start()]) or _BREAK.search(text[: match.start()]):
            return None
        return ContextParenthetical(match.group("body"), offset + match.start("body"))
    return None


def date_spans(text: str) -> list[tuple[int, int]]:
    """Read non-overlapping dates, preferring a complete date over its year."""
    spans: list[tuple[int, int]] = []
    for pattern in (FULL_DATE_RE, MONTH_YEAR_RE, YEAR_RE):
        for match in pattern.finditer(text):
            candidate = match.start(), match.end()
            if not any(candidate[0] < end and start < candidate[1] for start, end in spans):
                spans.append(candidate)
    return spans


def supported_date_span(context: ContextParenthetical) -> tuple[int, int] | None:
    """Return a terminal citation date, never an explanatory year in prose."""
    spans = date_spans(context.body)
    if len(spans) != 1:
        return None
    start, end = spans[0]
    if context.body[end:].strip(" ,;"):
        return None
    prefix = context.body[:start].strip(" ,;")
    if not prefix:
        return start, end
    if event := _DATE_EVENT.search(prefix):
        prefix = prefix[: event.start()].strip(" ,;")
    return (start, end) if looks_like_court_label(prefix) else None


def looks_like_court_label(text: str) -> bool:
    """Recognize citation-shaped court text without claiming it normalizes."""
    return court_id_if_unique(text) is not None or _COURT_LIKE.fullmatch(text.strip()) is not None
