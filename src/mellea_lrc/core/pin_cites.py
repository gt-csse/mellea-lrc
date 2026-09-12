r"""Which pages a pin cite claims.

A filing writes the same claim several ways. `247-48` is two pages, `588-90` is
three, `1068, 1071 -72` is one page and then two more, and `*3` is a Westlaw
page rather than a printed one. A checker that compares pin cites as strings
cannot see that `247-48` and `247 - 248` are the same claim, and one that reads
the first number only cannot see the second page at all.

So a pin cite is read into the pages it names, beside the text it was written
as. The text stays verbatim, damage included -- `1053 -54`, `*2 -3` with the
space extraction left inside it -- because normalizing in place would hide what
the document says. This is the second reading, not a replacement.

## The shorthand

The Bluebook lets a range drop the digits its endpoints share, so `588-90` ends
at 590 and `1053 -54` at 1054. The rule here is the mechanical one: when the
second number is shorter than the first, it replaces that many digits at the
end. A comma separates places that are not continuous, so each comma-separated
part is its own range.

## The label stays

`*3` is not page 3 and `¶ 26` is not page 26, so the label is read rather than
stripped: a star page is Westlaw's or LEXIS's, and a paragraph is a court's own
numbering in the public-domain format several states use. Extraction's
connector strip already removes what merely joins the page to the citation --
a comma, `at`, `p.` -- and leaves these alone.

## What is not read

A pin cite holding anything beyond pages, labels and the punctuation between
them is returned as one `NONCONFORMING` element and nothing else.
`657 n.1` is a page and a footnote on it; which number is the page claim is a
reading, and guessing it here would put an answer in the ground truth's own
shape. The text is on the citation for whoever makes that reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class PinCiteKind(str, Enum):
    """What a pin cite's numbers point at."""

    PAGE = "page"
    STAR = "star"
    """A Westlaw or LEXIS page, written `*3`."""
    PARAGRAPH = "paragraph"
    """A court's own paragraph number, written `¶ 26`.

    Several states cite decisions in public-domain format, where the court
    numbers its paragraphs and those numbers, not a reporter's pages, are what
    a pinpoint names. A pleading numbers its allegations the same way, so the
    notation alone does not say which is being cited.
    """
    NONCONFORMING = "nonconforming"
    """The pin cite states something these kinds do not cover."""


@dataclass(frozen=True, slots=True)
class PinCitePages:
    """A continuous run of pages a pin cite claims.

    `first` and `last` are equal for a single page and set for every kind but
    `NONCONFORMING`, which carries no numbers because none were read.
    """

    first: int | None
    last: int | None
    kind: PinCiteKind

    def __post_init__(self) -> None:
        numbered = self.first is not None and self.last is not None
        if self.kind is PinCiteKind.NONCONFORMING:
            if numbered:
                msg = "A nonconforming pin cite states no pages"
                raise ValueError(msg)
            return
        if not numbered:
            msg = f"A {self.kind.value} pin cite states a first and a last page"
            raise ValueError(msg)
        if self.last < self.first:  # type: ignore[operator]
            msg = f"A pin cite cannot end at {self.last} and begin at {self.first}"
            raise ValueError(msg)


# Pages, the labels in front of them, the commas between them and the hyphens
# inside them. A pin cite made of anything else is not read.
_CONFORMING = re.compile(r"[\d*¶,\s-]+")
_PART = re.compile(r"(?P<label>\*|¶¶|¶)?\s*(?P<first>\d+)(?:\s*[-–]\s*\*?(?P<last>\d+))?")
_LABELS = {"*": PinCiteKind.STAR, "¶": PinCiteKind.PARAGRAPH, "¶¶": PinCiteKind.PARAGRAPH}


def read_pin_cite(pin_cite: str | None) -> tuple[PinCitePages, ...]:
    """The pages a pin cite claims, or `()` when it states none."""
    if not pin_cite:
        return ()
    text = re.sub(r"\s+", " ", pin_cite).strip()
    if not _CONFORMING.fullmatch(text):
        return (PinCitePages(first=None, last=None, kind=PinCiteKind.NONCONFORMING),)
    pages, label = [], None
    for part in text.split(","):
        found = _PART.search(part)
        if not found:
            continue
        # A label written once governs what follows it: `*3 -4` and
        # `¶¶ 26, 28` each state their kind at the front.
        label = found.group("label") or label
        first, last = found.group("first"), found.group("last")
        pages.append(
            PinCitePages(
                first=int(first),
                last=_expand(first, last),
                kind=_LABELS.get(label or "", PinCiteKind.PAGE),
            )
        )
    return tuple(pages)


def _expand(first: str, last: str | None) -> int:
    """The page a range ends at, with the Bluebook's dropped digits restored."""
    if last is None:
        return int(first)
    if len(last) < len(first):
        return int(first[: len(first) - len(last)] + last)
    return int(last)
