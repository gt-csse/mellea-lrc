r"""Read eyecite's patterns as widened for a block, and put them back after.

Every widening this project applies to eyecite has the same two parts: a table
saying **what eyecite writes and what to read instead**, and a swap of the
module-level names that hold the composed result, for the duration of one call.
Both were written once inside the pin-cite reader. They are here so the next
widening is a row in a table rather than another copy of the machinery.

## Why a swap and not an argument

eyecite composes these patterns at import time. `POST_FULL_CITATION_REGEX` is an
f-string interpolating `PIN_CITE_REGEX`, and `helpers.py` imports the composed
result **by value**, so there is no seam to pass a variant through -- unlike the
reporter extractors, which :class:`~mellea_lrc.extraction.reading.relaxation.Relaxation`
rebuilds and hands to a tokenizer.

So the names are swapped for the duration of a block and restored in a `finally`.
Two consequences worth stating plainly:

*   Nothing is patched unless a caller asks, so eyecite as published is always
    one argument away. That is what the evaluation's baseline arm means.
*   The swap mutates module state, so a *concurrent* extraction in another
    thread would see the widened patterns while they are in effect. Extraction
    is synchronous and the window is one call, but it is a global and should be
    read as one.

## Why a table

A widening is a claim about a difference between what eyecite expects and what a
converted PDF holds, and every one of them so far is a literal substring of the
pattern. Writing them as data keeps the claim and its reason together, makes the
order they apply in visible -- one widening rewrites the spelling another one
matches on -- and means a new one cannot be added without saying why.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@dataclass(frozen=True, slots=True)
class Widening:
    """One difference between what eyecite writes and what a document holds."""

    written: str
    """Eyecite's own spelling, as a literal substring of the pattern."""

    read_as: str
    """What to read instead."""

    why: str
    """The damage or the convention that makes the widening necessary."""


def widen(pattern: str, widenings: Sequence[Widening]) -> str:
    """Apply every widening to a pattern, in the order they are given.

    The order is the caller's to fix and it matters: a widening written in
    eyecite's spelling has to run before one that rewrites that spelling.
    """
    for widening in widenings:
        pattern = pattern.replace(widening.written, widening.read_as)
    return pattern


@contextlib.contextmanager
def patched(patterns: dict[tuple[Any, str], str], **swaps: tuple[Any, str, Any]) -> Iterator[None]:
    """Swap module-level names for the block, and restore them after.

    `patterns` maps a `(module, name)` pair to the string to put there. `swaps`
    takes anything else that has to change with them -- a function eyecite uses
    to accept or reject what the widened pattern read -- as
    `(module, name, replacement)`.
    """
    originals = {where: getattr(where[0], where[1]) for where in patterns}
    originals |= {(module, name): getattr(module, name) for module, name, _ in swaps.values()}
    for (module, name), widened in patterns.items():
        setattr(module, name, widened)
    for module, name, replacement in swaps.values():
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)
