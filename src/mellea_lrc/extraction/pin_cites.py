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


@contextlib.contextmanager
def relaxed_pin_cites() -> Iterator[None]:
    """Read pin cites tolerantly for the duration of the block.

    Both names have to be swapped. `reference_pin_cite_re` reads
    `PIN_CITE_REGEX` when it is called, so that global reaches reference
    citations; `POST_FULL_CITATION_REGEX` already contains the strict version,
    baked in when it was composed, so it has to be widened separately. Patching
    only the first changes nothing about pin cites and looks exactly like a
    null result.
    """
    pin = eyecite.regexes.PIN_CITE_REGEX
    post = eyecite.helpers.POST_FULL_CITATION_REGEX
    eyecite.regexes.PIN_CITE_REGEX = relax(pin)
    eyecite.regexes.POST_FULL_CITATION_REGEX = relax(post)
    eyecite.helpers.POST_FULL_CITATION_REGEX = relax(post)
    try:
        yield
    finally:
        eyecite.regexes.PIN_CITE_REGEX = pin
        eyecite.regexes.POST_FULL_CITATION_REGEX = post
        eyecite.helpers.POST_FULL_CITATION_REGEX = post
