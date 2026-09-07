r"""Read a pin cite through the whitespace extraction leaves, and record the page.

Two jobs, because a pin cite arrives damaged in two different ways: the pattern
that finds it is too strict about spaces, and the string it hands back spells
the same page four ways depending on which citation kind it came from.

# I. Relaxing the pattern

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

# II. Stripping the connector

eyecite spells the same claim four ways, and all four are what it intends::

    550 U.S. 544, 570      FullCaseCitation    pin_cite='570'
    556 U.S. at 678        ShortCaseCitation   pin_cite='678'
    Id. at 547             IdCitation          pin_cite='at 547'
    Caraway , at 1301      ReferenceCitation   pin_cite=',  at  1301'

The first two are bare because the words before the page belong to something
else: a full citation's `, ` is punctuation the post-citation pattern consumes,
and a short form's `at` is part of the *locator* regex -- eyecite reads the page
straight out of `groups["page"]` and passes it back through `extract_pin_cite`
as a prefix. `Id.` and supra keep their `at` because they have no locator page
for it to belong to; that pairing is documented in eyecite's README.

The reference's leading comma is not intended. Both `ReferenceCitation`
construction sites in `find.py` build the object from `match.groupdict()`
directly, and `clean_pin_cite` -- which every path through `helpers.py` calls,
and which is only `pin_cite.strip(", ")` -- is not imported in that file at all.
The type arrived in 2.6.5 (January 2025) on a new extractor, four years after
the pin-cite handling it did not reuse, and `Bar at 7` and `Bar , at 9` in one
sentence still come back as `at 7` and `, at 9` on current `main`.

None of that survives to a scored column. The question a pin cite is scored on
is *which page*, and a connector that varies by citation kind is noise in an
equality test. So a leading comma, `at`, `p.`, `pp.` or `pg.` is removed and
every kind reads the same.

What is **not** removed is a label: `¶`, `§`, `*`, `n.`, `note` and `fn.` stay
where they are written, because they change what is being pointed at. `n. 1` is
not page 1 and `*3` is not page 3, so stripping them would not normalise a
spelling, it would assert something the document does not say.

The direction is forced rather than chosen. Adding a connector is impossible:
`550 U.S. 544, 570` contains no `at` anywhere, so there are no characters for a
span to cover. Removing one is always available.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator

import eyecite.helpers
import eyecite.regexes

# A comma, an `at`, or a page abbreviation standing in front of the page. Only
# what joins the pin cite to the citation -- `¶`, `§`, `*` and `n.` are labels
# that change the page's meaning and are left alone.
# The lookahead is what keeps `pt. 3` intact: without it `p` matches and the
# strip leaves `t. 3`, which reads as a page and is not one.
_CONNECTOR = re.compile(
    r"^[,\s]*(?:at[^\S\r\n]+)?(?:(?:pp?|pg)\.?(?=[\s\d]))?[^\S\r\n]*",
    re.I,
)


def strip_connector(pin_cite: str | None) -> str | None:
    """The page a pin cite states, without the words joining it to the citation.

    ``None`` when there is no pin cite, and also when nothing survives the strip
    -- a pin cite that is only a connector states no page.
    """
    if pin_cite is None:
        return None
    stripped = _CONNECTOR.sub("", pin_cite).strip(", \t")
    return stripped or None


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
