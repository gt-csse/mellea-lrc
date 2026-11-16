"""Read the pinpoint immediately following each full locator."""

import re
from bisect import bisect_left

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction.resolve_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.pin_cite import PIN_PREFIX

SUBSTAGE = "grow_roots.read_root_fields.resolve_pin_cites"

_BARE_NOTE = re.compile(r"^\s*,?\s*(?:at\s+)?(?P<pin>(?:n{1,2}\.|fn\.?)\s*\d+)", re.I)
_COURT_ORDINAL = re.compile(r"(?:st|nd|rd|th|d)\b\s+(?:Cir\.|Dept\.|Dist\.)", re.I)
_PARAGRAPH_BREAK = re.compile(r"\r?\n[^\S\r\n]*\r?\n")
_PARALLEL_REPORTER = re.compile(
    r"\s+(?:[A-Z][A-Za-z0-9.]*\.[A-Za-z0-9.]*|[A-Z]{2,})(?:\s+[A-Za-z0-9.]+){0,3}\s+\d+\b"
)
_PIN_CONTINUATION = re.compile(
    r"(?:[A-Za-z0-9]+"
    r"|[^\S\r\n]*[-–]\s*(?:[A-Za-z0-9*¶]+)?"
    r"|\s*,\s*(?:\d[\dA-Za-z]*|\*\d+|¶+\s*\d+|n{1,2}\.\s*\w+|fn\.?\s*\w+)"
    r"|[^\S\r\n]+(?:n{1,2}\.|fn\b\.?)\s*[^\s;(),]*"
    r"|[^\S\r\n]+(?:and|or|through|&)\s*(?:\d+|n{1,2}\.\s*\w+|fn\.?\s*\w+)"
    r"|[^\S\r\n]*&\s*(?:n{1,2}\.|fn\.?)\s*\w+)",
    re.I,
)


def _read_pin_cite(region: str, start: int) -> Span | None:
    match = PIN_PREFIX.match(region) or _BARE_NOTE.match(region)
    if match is None:
        return None
    finish = match.end("pin")
    if _COURT_ORDINAL.match(region, finish) or region[finish : finish + 1] == ":":
        return None
    pin_text = match.group("pin")
    if "," in pin_text and _PARALLEL_REPORTER.match(region, finish):
        finish = match.start("pin") + pin_text.rfind(",")
    elif continuation := _PIN_CONTINUATION.match(region, finish):
        # Keep malformed continuations in the source quote for later review.
        finish = len(region[: continuation.end()].rstrip())
    return Span(start + match.start("pin"), start + finish)


def resolve_pin_cites(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read each locator's own pin, including within a parallel citation group."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if COLOCATIONS not in document.substage_runs:
        raise ValueError("Resolve colocations before reading pin cites")
    limit = (rules or stable()).post_locator_window
    citations = document.full_locators
    starts = [citation.locator_span.start for citation in citations]
    source = document.text
    for citation in citations:
        if citation.pin_cite is not None:
            continue
        start = citation.locator_span.end
        next_index = bisect_left(starts, start)
        boundary = starts[next_index] if next_index < len(starts) else len(source)
        stop = min(boundary, start + limit)
        region = source[start:stop]
        if paragraph_break := _PARAGRAPH_BREAK.search(region):
            region = region[: paragraph_break.start()]
            boundary = stop = start + len(region)
        span = _read_pin_cite(region, start)
        if span is not None:
            if stop < boundary and not region[span.end - start :].strip():
                continue
            updated = citation.record(SUBSTAGE).with_pin_cite(source, span)
        elif (stop == boundary and not region.strip(" \t\r\n,")) or region.lstrip().startswith(
            ("(", ".", ";", ")")
        ):
            updated = citation.record(SUBSTAGE).with_pin_cite_not_stated()
        else:
            continue
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
