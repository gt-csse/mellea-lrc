"""Deterministic normalization of parsed citation text."""

from __future__ import annotations

import re

from mellea_lrc.model.citations.pin_cite import PinCiteKind, PinCiteTarget, PinCiteValue

_PAGE_PIN = re.compile(r"(?P<star>\*)?(?P<first>\d+)(?:[-–](?P<last>\d+))?\Z")


def normalize_pin_cite(quote: str) -> PinCiteValue:
    """Normalize a parsed page/star-page pin or raise on failure."""
    match = _PAGE_PIN.fullmatch(quote)
    if match is None:
        raise ValueError(f"Cannot normalize pin cite: {quote!r}")
    first_text = match.group("first")
    last_text = match.group("last")
    first = int(first_text)
    last = first
    if last_text is not None:
        last = int(last_text)
        if len(last_text) < len(first_text):
            place = 10 ** len(last_text)
            last += first // place * place
            if last < first:
                last += place
    return (
        PinCiteTarget(
            first=first,
            last=last,
            kind=PinCiteKind.STAR if match.group("star") else PinCiteKind.PAGE,
        ),
    )
