"""Reusable citation-field span rules, without document or stage state."""

from __future__ import annotations

import re

from mellea_lrc.field_extraction._case_names import read_case_name, read_short_case_name
from mellea_lrc.field_extraction._source_contexts import source_contexts
from mellea_lrc.field_extraction.models import FieldContext, FieldPrediction, validate_reader_inputs
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

    def __init__(
        self,
        *,
        case_name_window: int = 160,
        post_locator_window: int = 120,
        colocation_max_meaningful_gap: int = 5,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (case_name_window, post_locator_window)):
            raise ValueError("Rule field windows must be positive integers")
        if type(colocation_max_meaningful_gap) is not int or colocation_max_meaningful_gap < 0:
            raise ValueError("Colocation gap must be a nonnegative integer")
        self.case_name_window = case_name_window
        self.post_locator_window = post_locator_window
        self.colocation_max_meaningful_gap = colocation_max_meaningful_gap

    def predict(self, source: str, locator: Span, *, context: FieldContext | None = None) -> FieldPrediction:
        """Read the same anchored source input accepted by the ML reader."""
        return self.predict_many(source, (locator,), contexts=(context,) if context is not None else None)[0]

    def _predict(self, source: str, locator: Span, context: FieldContext) -> FieldPrediction:
        before = source[context.name_before.start : context.name_before.end]
        if context.kind == "short_reporter":
            region = context.pin_region
            marker = re.search(rf"\b{AT_JOIN}", source[region.start : region.end], re.I)
            pin = pin_after(source, region.start + marker.end(), region.end) if marker is not None else None
            prediction = FieldPrediction(
                context_span=Span(0, len(source)),
                case_name=read_short_case_name(before, context.name_before.start, context.known_names),
                pin_cite=Span(*pin) if pin is not None else None,
            )
        else:
            suffix = context.court_date_after
            court, date = read_court_and_date(source[suffix.start : suffix.end], suffix.start)
            prediction = FieldPrediction(
                context_span=Span(0, len(source)),
                case_name=read_case_name(
                    before,
                    context.name_before.start,
                    source[locator.start : locator.end] if context.kind == "reporter" else None,
                ),
                court=court,
                date=date,
                pin_cite=read_pin_cite(source, context.pin_region.start, context.pin_region.end),
            )
        prediction.validate_source(source, locator)
        return prediction

    def predict_many(
        self,
        source: str,
        locators: tuple[Span, ...],
        *,
        contexts: tuple[FieldContext, ...] | None = None,
    ) -> tuple[FieldPrediction, ...]:
        """Prepare source boundaries once and return one prediction per anchor."""
        validate_reader_inputs(source, locators, contexts)
        if not locators:
            return ()
        prepared = (
            contexts
            if contexts is not None
            else source_contexts(
                source,
                locators,
                case_name_window=self.case_name_window,
                post_locator_window=self.post_locator_window,
                colocation_max_meaningful_gap=self.colocation_max_meaningful_gap,
            )
        )
        validate_reader_inputs(source, locators, prepared)
        return tuple(
            self._predict(source, locator, context)
            for locator, context in zip(locators, prepared, strict=True)
        )
