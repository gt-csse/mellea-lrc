"""Read pinpoint references adjacent to full locator sites."""

import re

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig, FieldSource
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.pin_cite import PIN_PREFIX
from mellea_lrc.parsing.span import Span

SUBSTAGE = "grow_roots.read_root_fields.resolve_pin_cites"

_BARE_NOTE = re.compile(r"^\s*,?\s*(?:at\s+)?(?P<pin>(?:n{1,2}\.|fn\.?)\s*\d+)", re.I)
_COURT_ORDINAL = re.compile(r"(?:st|nd|rd|th|d)\b\s+(?:Cir\.|Dept\.|Dist\.)", re.I)
# The following reporter/page distinguishes a parallel volume from a pin list.
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


def read_pin_cite(source: str, start: int, end: int) -> Span | None:
    """Read an adjacent full-citation pin, retaining malformed continuations."""
    if not isinstance(source, str) or type(start) is not int or type(end) is not int:
        raise ValueError("A pinpoint suffix needs source text and integer offsets")
    if not 0 <= start <= end <= len(source):
        raise ValueError("Pinpoint suffix is outside the source")
    region = source[start:end]
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
        finish = len(region[: continuation.end()].rstrip())
    return Span(start + match.start("pin"), start + finish)


def resolve_pin_cites(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Read each locator's own pin, independently of shared group fields."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    require_structure(document)
    fields = field_extraction or FieldExtractionConfig.from_env()
    config = rules or stable()
    citations = document.full_locators
    if fields.pin_cite is FieldSource.ML:
        spans = [
            fields.validate_prediction(citation.get_ml_field_prediction()).pin_cite for citation in citations
        ]
    else:
        spans = []
        for citation in citations:
            site = citation.locator_span
            next_start = min(
                (item.locator_span.start for item in citations if item.locator_span.start >= site.end),
                default=len(document.text),
            )
            spans.append(
                read_pin_cite(document.text, site.end, min(next_start, site.end + config.post_locator_window))
            )
    for citation, span in zip(citations, spans, strict=True):
        updated = citation.record(SUBSTAGE)
        if span is not None:
            updated = updated.with_pin_cite(document.text, span)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
