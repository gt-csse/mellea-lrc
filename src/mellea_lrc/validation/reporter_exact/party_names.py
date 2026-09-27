"""Conservative, rule-based party comparison for reporter citation lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache, lru_cache

from reporters_db import CASE_NAME_ABBREVIATIONS, STATE_ABBREVIATIONS

from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameKind

_TOKEN = re.compile(r"&|[A-Za-z0-9]+(?:[.'’][A-Za-z0-9]+)*\.?", re.ASCII)
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_IN_RE = re.compile(r"^\s*in\s+re\b", re.IGNORECASE)
_EX_PARTE = re.compile(r"^\s*ex\s+parte\b", re.IGNORECASE)


@dataclass(frozen=True)
class PartyNameMatch:
    """Whether the written parties occur separately in a retrieved full name."""

    plaintiff_present: bool | None
    defendant_present: bool | None
    subject_present: bool | None
    qualifies: bool


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.group().replace("’", "'").casefold() for token in _TOKEN.finditer(text))


def _literal(token: str) -> str:
    # Retain '&' as its own literal; its Bluebook expansion is supplied by reporters-db.
    return token if token == "&" else _NON_ALNUM.sub("", token)


@lru_cache(maxsize=1)
def _abbreviations() -> dict[str, tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]]:
    """Index database spellings without treating full words as synonyms."""
    indexed: dict[str, set[tuple[tuple[str, ...], tuple[str, ...]]]] = {}
    entries = [
        *CASE_NAME_ABBREVIATIONS.items(),
        *((key, [value]) for key, value in STATE_ABBREVIATIONS.items()),
    ]
    for abbreviation, expansions in entries:
        key = _tokens(abbreviation)
        if not key:
            continue
        for expansion in expansions:
            words = tuple(_literal(token) for token in _tokens(expansion))
            if words and words != tuple(_literal(token) for token in key):
                indexed.setdefault(key[0], set()).add((key, words))
    return {first: tuple(sorted(options)) for first, options in indexed.items()}


def _options(tokens: tuple[str, ...], offset: int) -> tuple[tuple[int, tuple[str, ...]], ...]:
    token = tokens[offset]
    options = {(offset + 1, (_literal(token),))}
    for key, expanded in _abbreviations().get(token, ()):
        if tokens[offset : offset + len(key)] == key:
            options.add((offset + len(key), expanded))
    return tuple(sorted(options))


def _phrase_spans(phrase: str, full_name: str) -> frozenset[tuple[int, int]]:
    """Find whole-token spans, expanding only abbreviations actually written."""
    source = _tokens(phrase)
    target = _tokens(full_name)
    if not source or not target:
        return frozenset()

    @cache
    def ends(
        source_offset: int,
        target_offset: int,
        source_words: tuple[str, ...],
        target_words: tuple[str, ...],
    ) -> frozenset[int]:
        if source_offset == len(source) and not source_words:
            return frozenset((target_offset,)) if not target_words else frozenset()
        if not source_words:
            return frozenset().union(
                *(
                    ends(next_offset, target_offset, words, target_words)
                    for next_offset, words in _options(source, source_offset)
                )
            )
        if not target_words:
            if target_offset == len(target):
                return frozenset()
            return frozenset().union(
                *(
                    ends(source_offset, next_offset, source_words, words)
                    for next_offset, words in _options(target, target_offset)
                )
            )
        if source_words[0] != target_words[0]:
            return frozenset()
        return ends(source_offset, target_offset, source_words[1:], target_words[1:])

    return frozenset(
        (start, end) for start in range(len(target)) for end in ends(0, start, (), ()) if end > start
    )


def compare_case_names(source: CaseName, retrieved_full_name: str) -> PartyNameMatch:
    """Require both adversarial parties, or the subject in the same caption form.

    A database abbreviation may expand to multiple words. Each candidate expansion
    is checked against words actually written on the other side. Full words are
    never collapsed merely because they share a possible abbreviation.
    """
    if source.kind is CaseNameKind.ADVERSARIAL:
        plaintiff_spans = _phrase_spans(source.plaintiff or "", retrieved_full_name)
        defendant_spans = _phrase_spans(source.defendant or "", retrieved_full_name)
        distinct = any(
            plaintiff_end <= defendant_start or defendant_end <= plaintiff_start
            for plaintiff_start, plaintiff_end in plaintiff_spans
            for defendant_start, defendant_end in defendant_spans
        )
        return PartyNameMatch(
            plaintiff_present=bool(plaintiff_spans),
            defendant_present=bool(defendant_spans),
            subject_present=None,
            qualifies=distinct,
        )

    prefix = _IN_RE if source.kind is CaseNameKind.IN_RE else _EX_PARTE
    if not (match := prefix.match(retrieved_full_name)):
        return PartyNameMatch(None, None, False, False)
    subject_present = bool(_phrase_spans(source.subject or "", retrieved_full_name[match.end() :]))
    return PartyNameMatch(None, None, subject_present, subject_present)
