"""Read pinpoint references adjacent to full locator sites."""

from __future__ import annotations

import re

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction._support.context_windows import require_structure
from mellea_lrc.extraction._support.context_windows import site as locator_site
from mellea_lrc.model.citations.fields.pin_cite import PIN_PREFIX
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "pin_cites"

_BARE_NOTE = re.compile(r"^\s*,?\s*(?:at\s+)?(?P<pin>(?:n{1,2}\.|fn\.?)\s*\d+)", re.I)
_COURT_ORDINAL = re.compile(r"(?:st|nd|rd|th|d)\b\s+(?:Cir\.|Dept\.|Dist\.)", re.I)
# A comma-plus-number may start a parallel citation's volume. Its following
# reporter-like token and page distinguish it from another pinpoint target.
_PARALLEL_REPORTER = re.compile(
    r"\s+(?:[A-Z][A-Za-z0-9.]*\.[A-Za-z0-9.]*|[A-Z]{2,})"
    r"(?:\s+[A-Za-z0-9.]+){0,3}\s+\d+\b"
)
_PIN_CONTINUATION = re.compile(
    r"(?:[A-Za-z0-9]+"
    r"|[^\S\r\n]*[-–]\s*(?:[A-Za-z0-9*¶]+)?"
    r"|\s*,\s*(?:\d[\dA-Za-z]*|\*\d+|¶+\s*\d+|n{1,2}\.\s*\w+|fn\.?\s*\w+)"
    r"|[^\S\r\n]+(?:n{1,2}\.|fn\.?)\s*[^\s;(),]+"
    r"|[^\S\r\n]+(?:and|&)\s*(?:\d+|n{1,2}\.\s*\w+|fn\.?\s*\w+)"
    r"|[^\S\r\n]*&\s*(?:n{1,2}\.\s*\w+|fn\.?\s*\w+))",
    re.I,
)


def resolve_pin_cites(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an adjacent pin, retaining malformed continuations for later review."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        site = locator_site(citation)
        next_start = min(
            (
                locator_site(item).start
                for item in document.full_locators
                if locator_site(item).start >= site.end
            ),
            default=len(document.text),
        )
        region = document.text[site.end : min(next_start, site.end + config.post_locator_window)]
        match = PIN_PREFIX.match(region) or _BARE_NOTE.match(region)
        if match is None:
            continue
        end = match.end("pin")
        if _COURT_ORDINAL.match(region, end):
            continue
        if region[end : end + 1] == ":":
            # A colon immediately after digits is ordinarily prose numbering,
            # not a pin cite after the locator.
            continue
        pin_text = match.group("pin")
        parallel_reporter = "," in pin_text and _PARALLEL_REPORTER.match(region, end)
        if parallel_reporter:
            end = match.start("pin") + pin_text.rfind(",")
        elif continuation := _PIN_CONTINUATION.match(region, end):
            # Keep the unread token, without swallowing the following prose.
            end = len(region[: continuation.end()].rstrip())
        span = Span(site.end + match.start("pin"), site.end + end)
        document = document.replace_citation(citation.record(STAGE).with_pin_cite(document.text, span))
    return document.complete(STAGE)
