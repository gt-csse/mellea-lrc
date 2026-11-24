"""Full and short reporter source readings for native locator normalization.

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

from mellea_lrc.parsing.pin_cite import PIN_PREFIX

_ANY_WHITESPACE = r"\s*"
_REPORTER_GROUP = re.compile(r"\(\?P<reporter>((?:[^()\\]|\\.)*)\)")
_TIGHT_PUNCTUATION = re.compile(r"\\\.|['’]")


@dataclass(frozen=True)
class FullReporterReading:
    """A full-case eyecite result indexed to unchanged source text."""

    span: tuple[int, int]
    citation: FullCaseCitation


@dataclass(frozen=True, slots=True)
class ShortReporterReading:
    """A short-case eyecite result indexed to unchanged source text."""

    span: tuple[int, int]
    citation: ShortCaseCitation


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


def short_reporter_readings(source: str) -> tuple[ShortReporterReading, ...]:
    """Read short reporter sites without taking the next citation's text."""
    citations = sorted(
        get_citations(source, tokenizer=_reporter_tokenizer()), key=lambda citation: citation.span()[0]
    )
    readings: list[ShortReporterReading] = []
    next_start = len(source)
    for citation in reversed(citations):
        if isinstance(citation, ShortCaseCitation):
            start, end = citation.span_with_pincite()
            end = min(end, next_start)
            # Eyecite's token ends at the first page. The shared pinpoint
            # grammar can extend it through a written range, list, or note.
            if page := citation.groups.get("page"):
                token_end = citation.token.end
                pin_start = token_end - len(page)
                stop = min(end + 100, next_start)
                if source[pin_start:token_end] == page and (pin := PIN_PREFIX.match(source[pin_start:stop])):
                    end = max(end, pin_start + pin.end("pin"))
            readings.append(ShortReporterReading(span=(start, end), citation=citation))
        # Full cases, statutes, Id., and supra also bound a preceding site.
        next_start = citation.span()[0]
    return tuple(reversed(readings))
