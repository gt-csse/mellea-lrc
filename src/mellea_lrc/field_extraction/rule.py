"""Reusable citation-field span rules, without document or stage state."""

from __future__ import annotations

import re

from mellea_lrc.field_extraction._case_names import read_case_name
from mellea_lrc.field_extraction.models import FieldPrediction
from mellea_lrc.matching.literal_to_regex import fuzzy_literal
from mellea_lrc.parsing.dates import FULL_DATE_RE, YEAR_RE
from mellea_lrc.parsing.pin_cite import AT_JOIN, PIN_PREFIX, pin_after
from mellea_lrc.parsing.reporters import (
    FullReporterReading,
    ShortReporterReading,
    full_reporter_readings,
    short_reporter_readings,
)
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


def read_short_case_name(before: str, start: int, known_names: tuple[str, ...] = ()) -> Span | None:
    """Read a short name adjacent to a locator in a caller-bounded prefix."""
    if not isinstance(before, str) or type(start) is not int or start < 0:
        raise ValueError("A short-name prefix needs text and a nonnegative source offset")
    known: list[Span] = []
    for name in known_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Known case names must be nonempty strings")
        expression = rf"(?<!\w){fuzzy_literal(name, whitespace=True, newline=True)}\s*[,(]\s*$"
        if match := re.search(expression, before, re.I):
            finish = len(before.rstrip()) - 1
            finish = len(before[:finish].rstrip())
            known.append(Span(start + match.start(), start + finish))
    if known:
        return min(known, key=lambda part: part.start)
    match = re.search(r"(?P<name>[A-Z][\w.'’&/-]*(?:\s+[\w.'’&/-]+){0,12})\s*[,(]\s*$", before)
    if match is None:
        return None
    name_start = match.start("name")
    signal = re.match(r"(?:See(?:\s+also)?|But\s+see|Cf\.|Accord|Compare)\s+", match.group("name"), re.I)
    if signal:
        name_start += signal.end()
    return Span(start + name_start, start + match.end("name"))


class RuleFieldReader:
    """Predict all four source spans from the same text and locator as ML."""

    def __init__(self, *, case_name_window: int = 160, post_locator_window: int = 120) -> None:
        if any(type(value) is not int or value < 1 for value in (case_name_window, post_locator_window)):
            raise ValueError("Rule field windows must be positive integers")
        self.case_name_window = case_name_window
        self.post_locator_window = post_locator_window

    def predict(self, source: str, locator: Span) -> FieldPrediction:
        """Predict spans indexed to the unchanged source supplied by the caller."""
        return self.predict_many(source, (locator,))[0]

    def _predict(
        self,
        source: str,
        locator: Span,
        full: tuple[FullReporterReading, ...],
        short: tuple[ShortReporterReading, ...],
    ) -> FieldPrediction:
        if not isinstance(source, str) or not isinstance(locator, Span):
            raise TypeError("Rule field prediction needs source text and a Span locator")
        if (
            not 0 <= locator.start < locator.end <= len(source)
            or not source[locator.start : locator.end].strip()
        ):
            raise ValueError("Target locator is outside the source, empty, or blank")
        full_target = next(
            (reading for reading in full if reading.span == (locator.start, locator.end)), None
        )
        short_target = next(
            (reading for reading in short if reading.span == (locator.start, locator.end)), None
        )
        previous_end = max(
            (reading.span[1] for reading in (*full, *short) if reading.span[1] <= locator.start), default=0
        )
        following_start = min(
            (reading.span[0] for reading in (*full, *short) if reading.span[0] >= locator.end),
            default=len(source),
        )
        start = max(previous_end, locator.start - self.case_name_window)
        stop = min(following_start, locator.end + self.post_locator_window)
        before = source[start : locator.start]
        if short_target is not None:
            marker = re.search(rf"\b{AT_JOIN}", source[locator.start : locator.end], re.I)
            pin = pin_after(source, locator.start + marker.end(), locator.end) if marker is not None else None
            prediction = FieldPrediction(
                context_span=Span(0, len(source)),
                case_name=read_short_case_name(before, start),
                pin_cite=Span(*pin) if pin is not None else None,
            )
        else:
            court, date = read_court_and_date(source[locator.end : stop], locator.end)
            prediction = FieldPrediction(
                context_span=Span(0, len(source)),
                case_name=read_case_name(
                    before, start, source[locator.start : locator.end] if full_target is not None else None
                ),
                court=court,
                date=date,
                pin_cite=read_pin_cite(source, locator.end, stop),
            )
        prediction.validate_source(source, locator)
        return prediction

    def predict_many(self, source: str, locators: tuple[Span, ...]) -> tuple[FieldPrediction, ...]:
        """Return one prediction for each supplied locator, preserving order."""
        if not isinstance(source, str):
            raise TypeError("Rule field prediction source must be a string")
        if not isinstance(locators, tuple) or any(not isinstance(locator, Span) for locator in locators):
            raise TypeError("Rule field locators must be a tuple of Spans")
        if not locators:
            return ()
        for locator in locators:
            if (
                not 0 <= locator.start < locator.end <= len(source)
                or not source[locator.start : locator.end].strip()
            ):
                raise ValueError("Target locator is outside the source, empty, or blank")
        full = full_reporter_readings(source)
        short = short_reporter_readings(source)
        return tuple(self._predict(source, locator, full, short) for locator in locators)
