"""Ground model-proposed text in the source document, deterministically.

The extractor asks a model where the citations are; the model answers with strings.
Those strings are lookup keys, never evidence: every span this module returns is sliced from the source document,
so a located citation is verbatim document text by construction.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

from mellea_lrc.core.spans import Span
from mellea_lrc.experimental.llm_only_extraction.locators import parse_locator

MIN_FUZZY_TOKENS = 3
MIN_FUZZY_SCORE = 0.9

_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
_PUNCTUATION_EQUIVALENTS = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{MINUS SIGN}": "-",
        "\N{NO-BREAK SPACE}": " ",
    }
)


class MatchMethod(str, Enum):
    """How a model-proposed string was grounded in the source document."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"


@dataclass(frozen=True, slots=True)
class LocatedText:
    """One grounded span, with the evidence for trusting it."""

    span: Span
    text: str
    method: MatchMethod
    score: float


def locate_text(
    source: str,
    proposal: str,
    *,
    start: int = 0,
    allow_fuzzy: bool = True,
) -> LocatedText | None:
    """Ground `proposal` in `source` at or after `start`, or return None.

    Repeated text resolves to its first occurrence at or after `start`. Passing
    the previous match's end as `start` walks successive occurrences, which is
    how a brief that cites one case several times gets distinct spans.
    """
    proposal = proposal.strip()
    if not proposal or start < 0 or start > len(source):
        return None

    exact_start = source.find(proposal, start)
    if exact_start >= 0:
        return LocatedText(
            span=Span(exact_start, exact_start + len(proposal)),
            text=source[exact_start : exact_start + len(proposal)],
            method=MatchMethod.EXACT,
            score=1.0,
        )

    located = _locate_normalized(source, proposal, start)
    if located is not None:
        return located
    if not allow_fuzzy:
        return None
    return _locate_fuzzy(source, proposal, start)


def locate_locator(
    source: str,
    proposal: str,
    *,
    start: int = 0,
    allow_fuzzy: bool = True,
) -> LocatedText | None:
    """Ground the locator inside `proposal`, rejecting proposals with none.

    The shape gate runs first: a proposal that does not parse as a locator is
    refused before any span search, so a truncated ``320 U.S. *.`` can never
    attach itself to the ``320 U.S.`` prefix of a real citation. The grounded
    source text must then parse to the *same* locator, which stops a match from
    silently landing on a neighbouring volume or page.
    """
    proposed = parse_locator(proposal)
    if proposed is None:
        return None

    located = locate_text(source, proposed.text, start=start, allow_fuzzy=allow_fuzzy)
    if located is None:
        return None

    grounded = parse_locator(located.text)
    if grounded is None or grounded.key != proposed.key:
        return None
    return located


def _locate_normalized(source: str, proposal: str, start: int) -> LocatedText | None:
    """Match through Unicode, case, punctuation, and whitespace folding."""
    normalized_source, index_map = _normalized_with_map(source)
    normalized_proposal, _ = _normalized_with_map(proposal)
    if not normalized_proposal:
        return None

    search_from = _normalized_offset(index_map, start)
    found = normalized_source.find(normalized_proposal, search_from)
    if found < 0:
        return None

    span_start = index_map[found]
    span_end = index_map[found + len(normalized_proposal) - 1] + 1
    return LocatedText(
        span=Span(span_start, span_end),
        text=source[span_start:span_end],
        method=MatchMethod.NORMALIZED,
        score=1.0,
    )


def _locate_fuzzy(source: str, proposal: str, start: int) -> LocatedText | None:
    """Match on token windows, gated by a score floor and exact digit equality."""
    source_tokens = _tokens(source)
    proposal_tokens = tuple(match.group() for match in _WORD.finditer(_normalize(proposal)))
    token_count = len(proposal_tokens)
    if token_count < MIN_FUZZY_TOKENS or len(source_tokens) < token_count:
        return None

    proposal_digits = _digits(proposal_tokens)
    best_score = 0.0
    best_index = -1
    for index in range(len(source_tokens) - token_count + 1):
        if source_tokens[index][1] < start:
            continue
        window = source_tokens[index : index + token_count]
        window_tokens = tuple(token for token, _start, _end in window)
        if _digits(window_tokens) != proposal_digits:
            continue
        score = SequenceMatcher(None, proposal_tokens, window_tokens, autojunk=False).ratio()
        if score > best_score:
            best_score, best_index = score, index

    if best_index < 0 or best_score < MIN_FUZZY_SCORE:
        return None

    span_start = source_tokens[best_index][1]
    span_end = source_tokens[best_index + token_count - 1][2]
    return LocatedText(
        span=Span(span_start, span_end),
        text=source[span_start:span_end],
        method=MatchMethod.FUZZY,
        score=round(best_score, 3),
    )


def _digits(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return the digit runs in order.

    Order matters as much as membership: a volume and a page that swapped
    places name a different case, and comparing multisets would let that pass.
    """
    return tuple(token for token in tokens if token.isdigit())


def _tokens(value: str) -> tuple[tuple[str, int, int], ...]:
    """Return each word token with its start and end offset in the original."""
    normalized, index_map = _normalized_with_map(value)
    return tuple(
        (match.group(), index_map[match.start()], index_map[match.end() - 1] + 1)
        for match in _WORD.finditer(normalized)
    )


def _normalized_with_map(value: str) -> tuple[str, tuple[int, ...]]:
    """Normalize `value`, keeping each output character's source offset."""
    characters: list[str] = []
    index_map: list[int] = []
    for source_index, source_character in enumerate(value):
        for character in _normalize(source_character):
            if character.isspace():
                if characters and characters[-1] != " ":
                    characters.append(" ")
                    index_map.append(source_index)
                continue
            characters.append(character)
            index_map.append(source_index)
    while characters and characters[-1] == " ":
        characters.pop()
        index_map.pop()
    return "".join(characters), tuple(index_map)


def _normalized_offset(index_map: tuple[int, ...], start: int) -> int:
    """Return the first normalized index whose source offset is at or after `start`."""
    if start <= 0:
        return 0
    for normalized_index, source_index in enumerate(index_map):
        if source_index >= start:
            return normalized_index
    return len(index_map)


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).translate(_PUNCTUATION_EQUIVALENTS).casefold()
