"""Full and short reporter source readings with the shared eyecite tokenizer.

Eyecite generates reporter patterns with literal single-space joins. Widening
those joins in its tokenizer lets it read PDF whitespace on the original text,
so its spans and component groups remain exact source evidence. The same
reader is used on a whole document and on an isolated locator quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import ahocorasick
from eyecite import get_citations
from eyecite.models import FullCaseCitation, ShortCaseCitation, TokenExtractor
from eyecite.tokenizers import EXTRACTORS, AhocorasickTokenizer

from mellea_lrc.parsing.pin_cite import pin_after

_ANY_WHITESPACE = r"\s*"
_REPORTER_GROUP = re.compile(r"\(\?P<reporter>((?:[^()\\]|\\.)*)\)")
_TIGHT_PUNCTUATION = re.compile(r"\\\.|['’]")


@dataclass(frozen=True)
class FullReporterReading:
    """A full-case eyecite result indexed to unchanged source text."""

    span: tuple[int, int]
    citation: FullCaseCitation


def _relax_reporter_punctuation(match: re.Match[str]) -> str:
    body = _TIGHT_PUNCTUATION.sub(lambda part: rf"\s*{part.group()}\s*", match.group(1))
    return f"(?P<reporter>{body})"


def _relax_reporter_pattern(pattern: str) -> str:
    """Widen eyecite's generated joins without replacing its reporter list."""
    # A negative lookbehind keeps eyecite's reporter group from swallowing the
    # separator after it. Short citations have a separate reporter/page join.
    for written, relaxed in (
        (r") (?P<reporter>", rf"){_ANY_WHITESPACE}(?P<reporter>"),
        (
            r"),? at\s?(p(\.|age)?)? (?P<page>",
            rf")(?<!\s),?{_ANY_WHITESPACE}at{_ANY_WHITESPACE}(p(\.|age)?)?{_ANY_WHITESPACE}(?P<page>",
        ),
        (r"),? (?P<page>", rf")(?<!\s),?{_ANY_WHITESPACE}(?P<page>"),
    ):
        pattern = pattern.replace(written, relaxed)
    return _REPORTER_GROUP.sub(_relax_reporter_punctuation, pattern)


class _ReporterTokenizer(AhocorasickTokenizer):
    """Prefilter the replacement extractors, in deterministic source order."""

    def __post_init__(self) -> None:
        self.unfiltered_extractors = {extractor for extractor in self.extractors if not extractor.strings}
        self.case_sensitive_filter = self._filter(case_sensitive=True)
        self.case_insensitive_filter = self._filter(case_sensitive=False)
        self._order = {id(extractor): index for index, extractor in enumerate(self.extractors)}

    def get_extractors(self, text: str) -> list[TokenExtractor]:
        found = super().get_extractors(text)
        return sorted(found, key=lambda extractor: self._order.get(id(extractor), len(self._order)))

    def _filter(self, *, case_sensitive: bool) -> ahocorasick.Automaton:
        pairs = [
            (string.replace(" ", "") if case_sensitive else string.replace(" ", "").lower(), extractor)
            for extractor in self.extractors
            if extractor.strings and bool(extractor.flags & re.I) is not case_sensitive
            for string in extractor.strings
        ]
        return self.make_ahocorasick_filter(pairs)


@lru_cache(maxsize=1)
def _reporter_tokenizer() -> _ReporterTokenizer:
    return _ReporterTokenizer(
        extractors=[
            TokenExtractor(
                regex=_relax_reporter_pattern(extractor.regex),
                constructor=extractor.constructor,
                extra=extractor.extra,
                flags=extractor.flags,
                strings=extractor.strings,
            )
            for extractor in EXTRACTORS
        ]
    )


def full_reporter_readings(source: str) -> tuple[FullReporterReading, ...]:
    """Read full reporter locators with exact source spans."""
    return tuple(
        FullReporterReading(span=citation.span(), citation=citation)
        for citation in get_citations(source, tokenizer=_reporter_tokenizer())
        if isinstance(citation, FullCaseCitation)
    )


@dataclass(frozen=True, slots=True)
class ShortReporterReading:
    """A short-case eyecite result indexed to unchanged source text."""

    span: tuple[int, int]
    citation: ShortCaseCitation


def short_reporter_readings(source: str) -> tuple[ShortReporterReading, ...]:
    """Read short reporter sites, retaining their entire written pinpoint."""
    readings: list[ShortReporterReading] = []
    citations = sorted(
        get_citations(source, tokenizer=_reporter_tokenizer()), key=lambda citation: citation.span()[0]
    )
    next_start = len(source)
    # Discover every base span first, then extend pins back to front. The
    # next independently recognized authority bounds the pin grammar, so a
    # comma-separated reporter volume cannot become a page-list item. Keep
    # full/noncase/Id/supra events as boundaries even though this substage only
    # creates short reporters. Creation still happens in source order.
    for citation in reversed(citations):
        if isinstance(citation, ShortCaseCitation):
            start, end = citation.span_with_pincite()
            end = min(end, next_start)
            marker = re.search(r"\bat\s+", source[start:end], re.I)
            if marker and (pin := pin_after(source, start + marker.end(), min(end + 100, next_start))):
                end = pin[1]
            readings.append(ShortReporterReading(span=(start, end), citation=citation))
        next_start = citation.span()[0]
    return tuple(reversed(readings))
