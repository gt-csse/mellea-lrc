"""One eyecite reporter reader for discovery and field normalization.

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

_ANY_WHITESPACE = r"\s*"
_REPORTER_GROUP = re.compile(r"\(\?P<reporter>((?:[^()\\]|\\.)*)\)")
_TIGHT_PUNCTUATION = re.compile(r"\\\.|['’]")


@dataclass(frozen=True)
class ReporterReading:
    """An eyecite result whose span indexes the unchanged source string."""

    span: tuple[int, int]
    citation: FullCaseCitation | ShortCaseCitation


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


def reporter_readings(source: str) -> tuple[ReporterReading, ...]:
    """Return full and short reporter readings with their eyecite kind intact."""
    return tuple(
        ReporterReading(
            # A short citation's `span()` can end at "at "; its pinpoint is
            # part of the short site and belongs in the source-grounded quote.
            span=(
                citation.span() if isinstance(citation, FullCaseCitation) else citation.span_with_pincite()
            ),
            citation=citation,
        )
        for citation in get_citations(source, tokenizer=_reporter_tokenizer())
        if isinstance(citation, (FullCaseCitation, ShortCaseCitation))
    )


def full_reporter_readings(source: str) -> tuple[ReporterReading, ...]:
    """Return only eyecite full-case readings, preserving the shared matcher."""
    return tuple(
        reading for reading in reporter_readings(source) if isinstance(reading.citation, FullCaseCitation)
    )


def short_reporter_readings(source: str) -> tuple[ReporterReading, ...]:
    """Return only eyecite short-case readings, preserving the shared matcher."""
    return tuple(
        reading for reading in reporter_readings(source) if isinstance(reading.citation, ShortCaseCitation)
    )
