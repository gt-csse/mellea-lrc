r"""Relax the whitespace eyecite requires inside a pin cite.

The reporter joins were relaxed because eyecite writes a literal single space
between volume, reporter and page, and PDF extraction leaves several. Its
pin-cite pattern has the same defect twice, and the same cause.

**Separators.** `PIN_CITE_REGEX` writes its separators as `\ ?`, one optional
literal space. `544,  570` does not parse, so the page is filed under
`metadata.extra` instead and the claim the filing makes about page 570 becomes
invisible.

**The range hyphen.** A page range is `\d+(?:-\d+)?`, hyphen against the digits.
Extraction spaces it -- `998 -1003`, `337 - 38`, `189 - 90` -- and the whole pin
cite is lost the same way.

Over the 26 documents of `false-citation-bench`, the two together take citations
carrying a bare page in `extra` from **68 to 1**, and pin cites from 387 to 463.
Nothing else moves: no citation kind changes count, and every locator span is
identical. The one that remains is `928 F.3d 652, 657 n.1`, a page followed by a
footnote, which is a different shape rather than a whitespace problem.

Both widenings are horizontal only, as the reporter joins are. A doubled or
tabbed separator matches; a paragraph break does not. Doubled spaces are the
defect observed, and this project's history is that the bounded form was right
and the unbounded one bought errors.

## Why this is applied by patching, and only around one call

eyecite composes these patterns at import time. `POST_FULL_CITATION_REGEX` is an
f-string interpolating `PIN_CITE_REGEX`, and `helpers.py` imports the composed
result by value, so there is no seam to pass a variant through -- unlike the
reporter extractors, which `Relaxation` rebuilds and hands to a tokenizer.

So the patterns are swapped for the duration of a single extraction and restored
afterwards. Two consequences worth stating plainly:

*   `Relaxation.NONE` is left alone, so it remains eyecite exactly as published.
    That is what the evaluation baseline means by the name.
*   The swap mutates module state, so a *concurrent* extraction in another
    thread would see the relaxed patterns while it is in effect. Extraction is
    synchronous and the window is one call, but it is a global and should be
    read as one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import eyecite.helpers
import eyecite.regexes

_HORIZONTAL_OPTIONAL = r"[^\S\r\n]*"
_HORIZONTAL_REQUIRED = r"[^\S\r\n]+"
# Horizontal space either side of the hyphen, and an en dash beside it, because
# extraction produces both.
_RANGE_HYPHEN = r"[^\S\r\n]*[-–][^\S\r\n]*"


def relax(pattern: str) -> str:
    """Widen a pin-cite pattern's literal spaces and range hyphens."""
    widened = pattern.replace(r"\ ?", _HORIZONTAL_OPTIONAL).replace("\\ ", _HORIZONTAL_REQUIRED)
    widened = widened.replace(r"(?:-\d+(?::\d+)?)?", rf"(?:{_RANGE_HYPHEN}\d+(?::\d+)?)?")
    return widened.replace(r"(?:-\d+)?", rf"(?:{_RANGE_HYPHEN}\d+)?")


# Every pattern that embeds `PIN_CITE_REGEX` at import time. Widening the
# constant alone reaches only what reads it at call time, which is references.
_BAKED = (
    "POST_FULL_CITATION_REGEX",
    "POST_SHORT_CITATION_REGEX",
    "POST_LAW_CITATION_REGEX",
    "POST_JOURNAL_CITATION_REGEX",
)


@contextlib.contextmanager
def relaxed_pin_cites() -> Iterator[None]:
    """Read pin cites tolerantly for the duration of the block.

    Several names have to be swapped, and finding that out is the point of this
    module. `reference_pin_cite_re` reads `PIN_CITE_REGEX` when it is called, so
    that global reaches reference citations. The four `POST_*` patterns are
    f-strings that interpolated the same constant **at import time**, so the
    strict version is already baked into each and the global does nothing for
    them; `helpers.py` then imports the composed results by value, so those
    bindings need patching too.

    All four, not just the full-citation one. A short form writes its page the
    same way -- `556 U.S. at 678` -- and breaks on the same doubled space, and
    an `Id.` takes its pin cite through the short-citation pattern. Widening
    only the full path leaves `645  B.R.  at  181` and `Id. at  547` unread,
    which is 15 pin cites on the bench and its own citation entirely where the
    damage falls between the reporter and the `at`.
    """
    pin = eyecite.regexes.PIN_CITE_REGEX
    baked = {name: getattr(eyecite.helpers, name) for name in _BAKED}
    eyecite.regexes.PIN_CITE_REGEX = relax(pin)
    for name, pattern in baked.items():
        widened = relax(pattern)
        setattr(eyecite.regexes, name, widened)
        setattr(eyecite.helpers, name, widened)
    try:
        yield
    finally:
        eyecite.regexes.PIN_CITE_REGEX = pin
        for name, pattern in baked.items():
            setattr(eyecite.regexes, name, pattern)
            setattr(eyecite.helpers, name, pattern)
