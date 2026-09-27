"""Small, reusable literal matching policies for noisy document text."""

from __future__ import annotations

import re


def fuzzy_literal(value: str, *, whitespace: bool = False, newline: bool = False) -> str:
    """Return a regex for a literal, optionally relaxing whitespace joins.

    With ``whitespace=False`` this is exactly ``re.escape(value)``. Relaxation
    permits zero or more spaces at written whitespace joins and around literal
    punctuation; it does not change any letter or digit. Newlines remain
    excluded unless explicitly requested.
    """
    if not whitespace:
        return re.escape(value)
    separator = r"\s*" if newline else r"[^\S\r\n]*"
    tokens = tuple(re.finditer(r"\w+|[^\w\s]+", value))
    if not tokens:
        return separator
    pieces: list[str] = []
    for index, token in enumerate(tokens):
        pieces.append(re.escape(token.group()))
        if index == len(tokens) - 1:
            if value[token.end() :]:
                pieces.append(separator)
            continue
        following = tokens[index + 1]
        if (
            value[token.end() : following.start()]
            or not token.group().isalnum()
            or not following.group().isalnum()
        ):
            pieces.append(separator)
    return "".join(pieces)
