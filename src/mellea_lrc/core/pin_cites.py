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

## A footnote is on a page

`570 n.10` is the Bluebook's form for citing a footnote, and Rule 3.2(b) fixes
which number is which: the page the footnote appears on, then `n.` and the
footnote's own number, closed up, with `nn.` for several. So the page claim is
570 and the footnote says where on it -- the kind stays `PAGE` and the footnote
number rides beside it, because a reader cutting pages wants 570 and a reader
deciding whether a passage is in the body or a note wants the rest.

The marker is orthogonal to the kind. `*3 n.1` is a star page with a footnote
on it, and reading a footnote as a kind of its own would make every consumer
ask "is this a page?" and get the wrong answer.

## What is not read

A pin cite holding anything beyond pages, labels, footnote markers and the
punctuation between them is returned as one `UNREAD` element carrying `note`,
which says what stopped the read. **`UNREAD` is a statement about this reader,
not about the filing**: `slip op. at 3` is a perfectly proper citation that
nothing here knows how to turn into reporter pages. The word `nonconforming`
is deliberately not used, because `nonconforming_citation` is a dataset unit
meaning a case the filing argues from and never cites, which is a defect.
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
    UNREAD = "unread"
    """Nothing here could turn this pin cite into pages.

    A statement about this reader. The pin cite may be perfectly proper --
    `slip op. at 3` names a page of a slip opinion, which has no reporter
    pagination at all -- or it may be damage the converter left, `749-50`
    arriving as `74950`. Which it is, `note` says.
    """


@dataclass(frozen=True, slots=True)
class PinCitePages:
    """A continuous run of pages a pin cite claims.

    `first` and `last` are equal for a single page and set for every kind but
    `UNREAD`, which carries no numbers because none were read.
    """

    first: int | None
    last: int | None
    kind: PinCiteKind
    footnote: str | None = None
    """The footnote on these pages, as written: `10` for `570 n.10`.

    `None` when the pin cite names no footnote, which is most of them. Set
    beside any kind, since `*3 n.1` is a footnote on a star page.
    """
    note: str | None = None
    """Why nothing could be read, and set only for `UNREAD`.

    Says what stopped the read rather than that something did, so a reader can
    tell a proper citation this reader does not cover from converter damage
    without fetching the document.
    """

    def __post_init__(self) -> None:
        numbered = self.first is not None and self.last is not None
        if self.kind is PinCiteKind.UNREAD:
            if numbered:
                msg = "An unread pin cite states no pages"
                raise ValueError(msg)
            if not self.note:
                msg = "An unread pin cite must say what stopped the read"
                raise ValueError(msg)
            return
        if self.note:
            msg = f"A {self.kind.value} pin cite was read, so it carries no note"
            raise ValueError(msg)
        if not numbered:
            msg = f"A {self.kind.value} pin cite states a first and a last page"
            raise ValueError(msg)
        if self.last < self.first:  # type: ignore[operator]
            msg = f"A pin cite cannot end at {self.last} and begin at {self.first}"
            raise ValueError(msg)


# Pages, the labels in front of them, the footnote markers after them, the
# commas between them and the hyphens inside them. Anything else is not read.
_READABLE = re.compile(r"[\d*¶,\s-]+(?:n{1,2}\.\s*[\d,\s-]+)?[\d*¶,\s-]*")
_PART = re.compile(
    r"(?P<label>\*|¶¶|¶)?\s*(?P<first>\d+)(?:\s*[-–]\s*\*?(?P<last>\d+))?"
    # Rule 3.2(b): the page, then `n.` and the footnote's own number, `nn.` for
    # several. The page is the claim; the footnote says where on it.
    r"(?:\s*n{1,2}\.\s*(?P<footnote>\d+(?:\s*[-–,]\s*\d+)*))?"
)
_LABELS = {"*": PinCiteKind.STAR, "¶": PinCiteKind.PARAGRAPH, "¶¶": PinCiteKind.PARAGRAPH}
#: A run of characters that is none of a page, a label or the punctuation
#: between them, which is what an unread pin cite is quoted as holding.
_STRANGE = re.compile(r"[^\d*¶,\s-]+")


def read_pin_cite(pin_cite: str | None) -> tuple[PinCitePages, ...]:
    """The pages a pin cite claims, or `()` when it states none."""
    if not pin_cite:
        return ()
    text = re.sub(r"\s+", " ", pin_cite).strip()
    if not _READABLE.fullmatch(text):
        return (PinCitePages(first=None, last=None, kind=PinCiteKind.UNREAD, note=_why_unread(text)),)
    pages, label = [], None
    for part in _parts(text):
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
                footnote=_collapsed(found.group("footnote")),
            )
        )
    return tuple(pages) or (
        PinCitePages(first=None, last=None, kind=PinCiteKind.UNREAD, note="it states no number"),
    )


def _parts(text: str) -> list[str]:
    """The comma-separated places, with a footnote's own commas left inside it.

    `570 nn.10, 12` is one page and two footnotes on it, not two places.
    """
    parts, current = [], ""
    for piece in text.split(","):
        if current and re.search(r"n{1,2}\.\s*[\d\s-]*$", current):
            current += "," + piece
            continue
        if current:
            parts.append(current)
        current = piece
    if current:
        parts.append(current)
    return parts


def _collapsed(footnote: str | None) -> str | None:
    """A footnote marker with the spacing closed up: `10 - 12` is `10-12`."""
    if not footnote:
        return None
    return re.sub(r"\s*([-–])\s*", r"\1", re.sub(r"\s*,\s*", ", ", footnote.strip()))


def _why_unread(text: str) -> str:
    """What the pin cite holds that stopped it being read."""
    strange = " ".join(_STRANGE.findall(text))
    if strange:
        return f"it holds {strange!r}, which is not a page, a label or a footnote marker"
    return "its numbers and markers are not in an order this reads"


def _expand(first: str, last: str | None) -> int:
    """The page a range ends at, with the Bluebook's dropped digits restored."""
    if last is None:
        return int(first)
    if len(last) < len(first):
        return int(first[: len(first) - len(last)] + last)
    return int(last)
