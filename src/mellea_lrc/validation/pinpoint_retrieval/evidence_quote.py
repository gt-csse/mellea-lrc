"""Deterministically ground a proposed evidence quote in source text."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from mellea_lrc.core.spans import Span
from mellea_lrc.validation.types import EvidenceQuoteMatchMethod

MIN_FUZZY_TOKENS = 5
MIN_FUZZY_SCORE = 0.9
_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
_NEGATIONS = frozenset({"cannot", "never", "no", "not", "without"})
_PUNCTUATION_EQUIVALENTS = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{MINUS SIGN}": "-",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedEvidenceQuote:
    """One uniquely grounded source span for a proposed quote."""

    span: Span
    text: str
    method: EvidenceQuoteMatchMethod
    score: float


def resolve_evidence_quote(source: str, proposed_quote: str) -> ResolvedEvidenceQuote | None:
    """Resolve a quote uniquely, allowing only tightly bounded surface drift."""
    quote = proposed_quote.strip()
    if not quote:
        return None

    exact_starts = _substring_starts(source, quote)
    if len(exact_starts) == 1:
        start = exact_starts[0]
        return ResolvedEvidenceQuote(
            span=Span(start, start + len(quote)),
            text=source[start : start + len(quote)],
            method=EvidenceQuoteMatchMethod.EXACT,
            score=1.0,
        )

    normalized_source, source_map = _normalized_text_with_map(source)
    normalized_quote, _ = _normalized_text_with_map(quote)
    normalized_starts = _substring_starts(normalized_source, normalized_quote)
    if normalized_quote and len(normalized_starts) == 1:
        normalized_start = normalized_starts[0]
        start = source_map[normalized_start]
        end = source_map[normalized_start + len(normalized_quote) - 1] + 1
        return ResolvedEvidenceQuote(
            span=Span(start, end),
            text=source[start:end],
            method=EvidenceQuoteMatchMethod.NORMALIZED,
            score=1.0,
        )

    return _resolve_fuzzy(source, quote)


def _resolve_fuzzy(source: str, quote: str) -> ResolvedEvidenceQuote | None:
    source_tokens = _tokens(source)
    quote_tokens = tuple(match.group().casefold() for match in _WORD.finditer(_normalize(quote)))
    token_count = len(quote_tokens)
    if token_count < MIN_FUZZY_TOKENS or len(source_tokens) < token_count:
        return None

    candidates: list[tuple[float, int]] = []
    for index in range(len(source_tokens) - token_count + 1):
        source_window = tuple(token for token, _start, _end in source_tokens[index : index + token_count])
        if _critical_tokens(source_window) != _critical_tokens(quote_tokens):
            continue
        candidates.append(
            (
                SequenceMatcher(
                    None,
                    quote_tokens,
                    source_window,
                    autojunk=False,
                ).ratio(),
                index,
            )
        )
    if not candidates:
        return None
    # WARNING: For now, ties select the first highest-scoring source window.
    # Repeated or similarly worded passages can therefore attach the span to the
    # wrong occurrence. Future ambiguity handling must compare distinct passages
    # rather than treating overlapping windows around one passage as competitors.
    best_score, best_index = max(candidates, key=lambda candidate: candidate[0])
    if best_score < MIN_FUZZY_SCORE:
        return None

    start = source_tokens[best_index][1]
    end = source_tokens[best_index + token_count - 1][2]
    while end < len(source) and not source[end].isalnum() and not source[end].isspace():
        end += 1
    return ResolvedEvidenceQuote(
        span=Span(start, end),
        text=source[start:end],
        method=EvidenceQuoteMatchMethod.FUZZY,
        score=round(best_score, 3),
    )


def _tokens(value: str) -> tuple[tuple[str, int, int], ...]:
    normalized, index_map = _normalized_text_with_map(value)
    return tuple(
        (
            match.group(),
            index_map[match.start()],
            index_map[match.end() - 1] + 1,
        )
        for match in _WORD.finditer(normalized)
    )


def _critical_tokens(tokens: tuple[str, ...]) -> Counter[str]:
    """Preserve negation and numbers across any fuzzy quote recovery."""
    return Counter(token for token in tokens if token in _NEGATIONS or token.isdigit())


def _normalized_text_with_map(value: str) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    index_map: list[int] = []
    for source_index, source_character in enumerate(value):
        normalized = _normalize(source_character)
        for character in normalized:
            if character.isspace():
                if characters and characters[-1] != " ":
                    characters.append(" ")
                    index_map.append(source_index)
                continue
            characters.append(character)
            index_map.append(source_index)
    if characters and characters[-1] == " ":
        characters.pop()
        index_map.pop()
    return "".join(characters), tuple(index_map)


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).translate(_PUNCTUATION_EQUIVALENTS).casefold()


def _substring_starts(value: str, substring: str) -> list[int]:
    if not substring:
        return []
    starts: list[int] = []
    start = 0
    while (found := value.find(substring, start)) >= 0:
        starts.append(found)
        start = found + 1
    return starts
