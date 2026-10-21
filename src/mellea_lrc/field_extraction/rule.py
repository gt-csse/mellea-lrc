"""Reusable citation-field span rules, without document or stage state."""

from __future__ import annotations

import re

from mellea_lrc.field_extraction._case_names import read_case_name, read_short_case_name
from mellea_lrc.field_extraction.models import FieldPrediction, LocatorContext, validate_contexts
from mellea_lrc.parsing.dates import FULL_DATE_RE, YEAR_RE
from mellea_lrc.parsing.pin_cite import AT_JOIN, PIN_PREFIX, pin_after
from mellea_lrc.parsing.span import Span

_PAREN = re.compile(r"\((?P<body>[^()\r\n]{0,100})\)")
_DATE_EVENT = re.compile(r"\s+\b(?:filed|decided|issued)\b\s*$", re.I)
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


def read_court_and_date(after: str, start: int) -> tuple[Span | None, Span | None]:
    """Read one dated parenthetical from an already bounded suffix."""
    if not isinstance(after, str) or type(start) is not int or start < 0:
        raise ValueError("A court/date suffix needs text and a nonnegative source offset")
    for parenthetical in _PAREN.finditer(after):
        body = parenthetical.group("body")
        if not YEAR_RE.search(body):
            continue
        if re.search(r"\.\s+[A-Z]", after[: parenthetical.start()]):
            return None, None
        date = FULL_DATE_RE.search(body) or YEAR_RE.search(body)
        if date is None:
            raise ValueError("Dated parenthetical did not contain a readable year")
        body_start = start + parenthetical.start("body")
        date_span = Span(body_start + date.start(), body_start + date.end())
        court_region = body[: date.start()]
        if event := _DATE_EVENT.search(court_region):
            court_region = court_region[: event.start()]
        written = court_region.strip(" ,;")
        if not written:
            return None, date_span
        stripped = len(court_region) - len(court_region.lstrip(" ,;"))
        return Span(body_start + stripped, body_start + stripped + len(written)), date_span
    return None, None


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


class RuleFieldReader:
    """Predict all four source spans once, before field normalization stages."""

    def predict(self, context: LocatorContext) -> FieldPrediction:
        """Read exactly the supplied context, without constructing another one."""
        return self.predict_many((context,))[0]

    def _predict(self, context: LocatorContext) -> FieldPrediction:
        source = context.text
        offset = context.source_span.start
        prefix = context.name_before
        before = source[prefix.start - offset : prefix.end - offset]
        if context.kind == "short_reporter":
            region = context.pin_region
            start, end = region.start - offset, region.end - offset
            marker = re.search(rf"\b{AT_JOIN}", source[start:end], re.I)
            pin = pin_after(source, start + marker.end(), end) if marker is not None else None
            prediction = FieldPrediction(
                context_span=context.source_span,
                case_name=read_short_case_name(before, context.name_before.start, context.known_names),
                pin_cite=Span(offset + pin[0], offset + pin[1]) if pin is not None else None,
            )
        else:
            suffix = context.court_date_after
            court, date = read_court_and_date(
                source[suffix.start - offset : suffix.end - offset], suffix.start
            )
            pin = read_pin_cite(source, context.pin_region.start - offset, context.pin_region.end - offset)
            locator = context.locator
            prediction = FieldPrediction(
                context_span=context.source_span,
                case_name=read_case_name(
                    before,
                    context.name_before.start,
                    source[locator.start - offset : locator.end - offset]
                    if context.kind == "reporter"
                    else None,
                ),
                court=court,
                date=date,
                pin_cite=Span(offset + pin.start, offset + pin.end) if pin is not None else None,
            )
        prediction.validate_context(context)
        return prediction

    def predict_many(self, contexts: tuple[LocatorContext, ...]) -> tuple[FieldPrediction, ...]:
        """Return one prediction for each already prepared locator context."""
        validate_contexts(contexts)
        return tuple(self._predict(context) for context in contexts)
