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
identical.

**The footnote.** `928 F.3d 652, 657 n.1` was the one pin cite the two
widenings above left unread, and it is a different shape rather than a
whitespace problem: eyecite understands `n.` as a label, but only in front of a
page that follows a **comma**, and Rule 3.2(b) writes no comma between a page
and the footnote on it. So the separator accepts a horizontal space in place of
that comma when a note label follows it, and nothing else -- `544, 570 2007`
must not read 2007 as a second page. This is what takes `550 n. 16`, `246 n.13`,
`1117 n.4` and `850 n.10` from unread to a page with a footnote beside it.

At `Relaxation.BOUNDED` both widenings are horizontal only, as the reporter
joins are: a doubled or tabbed separator matches and a paragraph break does not.
At `FULL` they are any whitespace, which is what that level already means for
the volume, the reporter and the page.

**The blank line needs more than the pattern.** `add_post_citation` reads the
pin cite with `match_on_tokens`, which stops appending at a `ParagraphToken`, so
the widest pattern in the world is matched against text that ends before the
page begins. At `FULL` the scan is run a second time with the break flattened --
and only as a fallback, where the strict scan found no pin cite at all. Reading
past the break unconditionally costs as much as it buys: `Id. at 809\n\nThe
Murphy Order` and `960 F.Supp. 253, 254\n\n- (D. Kan. 1997)` both end their
page at the break, and continuing loses a page that was read. As a fallback it
adds six pin cites across the three annotated sets and removes none.

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
from dataclasses import replace

import eyecite.helpers
import eyecite.regexes
import eyecite.resolve
from eyecite.tokenizers import ParagraphToken

from mellea_lrc.extraction.reading.eyecite_patterns import Widening, patched, widen
from mellea_lrc.model.extraction_metadata import Relaxation

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
# At `Relaxation.FULL` the page may be anywhere after the separator, blank line
# included, which is the same latitude the locator already takes at that level:
# `371 U.S. 178,\n\n182 (1962)` is one citation on the page and the pin cite is
# the only part of it this project was still losing.
_ANY_OPTIONAL = r"\s*"
_ANY_REQUIRED = r"\s+"
_RANGE_HYPHEN_ANY = r"\s*[-–]\s*"

#: What this project reads differently from eyecite, in the order it applies.
#: Every one is a literal substring of eyecite's own pattern, and the order
#: matters: the footnote and the end of text are written in eyecite's spelling
#: of a space, which the two widenings after them rewrite.
PIN_CITE_WIDENINGS = (
    Widening(
        written="(?:,\\ ?",
        read_as="(?:(?:,|\\ (?=(?:&\\ )?(?:note|nn?\\.|fn?\\.)))\\ ?",
        why=(
            "A page after the first needs a comma in front of it, and Bluebook "
            "Rule 3.2(b) writes none between a page and the footnote on it: "
            "`570 n.10`. A space is accepted in its place when a note label "
            "follows, and only then, because a comma is what separates two "
            "pages and `544, 570 2007` must not read 2007 as a second one."
        ),
    ),
    Widening(
        written="$            # end of text",
        read_as="[^\\S\\r\\n]*$   # end of text, or the space a cut left",
        why=(
            "A pin cite must be followed by punctuation, a paren or the end of "
            "the text. `extract_pin_cite` matches with `strings_only`, so the "
            "text stops at the first token that is not a string -- and "
            "`(quoting` is a stop word, hence a token. `Id. at 71 (quoting ...)` "
            "is matched against `' at 71 '`, and the space the cut left is all "
            "that stands between the page and the end."
        ),
    ),
    Widening(
        written=r"\ ?",
        read_as=_HORIZONTAL_OPTIONAL,
        why="An optional literal space, where extraction leaves several or none.",
    ),
    Widening(
        written="\\ ",
        read_as=_HORIZONTAL_REQUIRED,
        why="A required literal space, where extraction leaves several.",
    ),
    Widening(
        written=r"(?:-\d+(?::\d+)?)?",
        read_as=rf"(?:{_RANGE_HYPHEN}\d+(?::\d+)?)?",
        why="A page:paragraph range's hyphen, which extraction spaces.",
    ),
    Widening(
        written=r"(?:-\d+)?",
        read_as=rf"(?:{_RANGE_HYPHEN}\d+)?",
        why="A page range's hyphen: `998 -1003`, `337 - 38`, `189 - 90`.",
    ),
)


def widenings_for(relaxation: Relaxation) -> tuple[Widening, ...]:
    """`PIN_CITE_WIDENINGS`, with the separators the relaxation level allows.

    `BOUNDED` keeps the horizontal forms: a doubled or tabbed space matches and
    a paragraph break does not. `FULL` lets every one of them be any whitespace,
    which is what `FULL` already means for the volume, the reporter and the
    page, and the reason it exists -- a citation broken across a blank line is
    still one citation.
    """
    if relaxation is not Relaxation.FULL:
        return PIN_CITE_WIDENINGS
    wider = {
        _HORIZONTAL_OPTIONAL: _ANY_OPTIONAL,
        _HORIZONTAL_REQUIRED: _ANY_REQUIRED,
    }
    return tuple(
        replace(
            step,
            read_as=wider.get(step.read_as, step.read_as).replace(_RANGE_HYPHEN, _RANGE_HYPHEN_ANY),
        )
        for step in PIN_CITE_WIDENINGS
    )


def relax(pattern: str, relaxation: Relaxation = Relaxation.BOUNDED) -> str:
    """Widen a pin-cite pattern by every reading the relaxation level allows."""
    return widen(pattern, widenings_for(relaxation))


_BAKED = (
    "POST_FULL_CITATION_REGEX",
    "POST_SHORT_CITATION_REGEX",
    "POST_LAW_CITATION_REGEX",
    "POST_JOURNAL_CITATION_REGEX",
)


_HORIZONTAL_RUN = re.compile(r"[^\S\r\n]+")
# What a filing writes between a citation and the page it claims, and nothing
# else: whitespace, the comma, and the connector.
_SEPARATOR_ONLY = re.compile(r"[\s,]*(?:at\b[\s,]*)?(?:pp?\.[\s,]*)?")


def _tolerant_check(original):
    r"""eyecite's pin-cite check, asked about a pin cite with its spaces collapsed.

    Reading a pin cite through doubled spaces is only half of it. `Id. at  547`
    parses once the patterns are widened, and then resolution throws it away:
    `_has_invalid_pin_cite` tests the string with `(?:at )?(\d+)`, one literal
    space, so the doubled one fails to match and the id cite is called invalid.

    Because an `Id.` that follows an unresolved `Id.` is refused by rule, one
    damaged space strands the citation after it as well. Document 026 loses two
    that way, both returning to `Bell v. Wolfish, 441 U.S. 520` under a sentence
    that names it.

    The check itself is right; what it reads is damaged. So it is asked about
    the same pin cite with horizontal runs collapsed, and the citation keeps the
    characters the filing wrote.
    """

    def check(full_cite, id_cite) -> bool:
        pin = getattr(id_cite.metadata, "pin_cite", None)
        if not pin:
            return original(full_cite, id_cite)
        collapsed = _HORIZONTAL_RUN.sub(" ", pin).strip()
        if collapsed == pin:
            return original(full_cite, id_cite)
        id_cite.metadata.pin_cite = collapsed
        try:
            return original(full_cite, id_cite)
        finally:
            id_cite.metadata.pin_cite = pin

    return check


def _across_paragraphs(original):
    """eyecite's forward token scan, asked again across a blank line.

    Widening the pattern is not enough on its own. `add_post_citation` reads the
    pin cite with `match_on_tokens`, which builds the text to match by appending
    tokens and **stops at a `ParagraphToken`** -- so where the converter put a
    blank line between the page and the pin cite, the pattern is matched against
    text that ends before the pin cite starts, however wide the pattern is.
    `371 U.S. 178,\n\n182 (1962)` is one citation on the page, and at
    `Relaxation.FULL` a blank line is not a boundary.

    **The second scan is a fallback, never a replacement.** Reading past the
    break unconditionally costs as much as it buys: where the pin cite has
    already ended at the break, what follows is the next sentence or the margin
    of pleading paper -- `Id. at 809\n\nThe Murphy Order`, `960 F.Supp. 253,
    254\n\n- (D. Kan. 1997)` -- and continuing turns a page that was read into
    one that is not. So the strict scan runs first and its answer stands
    wherever it found a pin cite; the wider one is tried only where it found
    none, which can add a page and can never take one away.

    Backward scans are left alone. They are how a case name is found, and a name
    reaching back over a paragraph break would cross into the sentence before.
    """

    def scan(words, start_index, regex, prefix="", strings_only=False, forward=True, **kwargs):
        found = original(words, start_index, regex, prefix, strings_only, forward, **kwargs)
        if not forward or "?P<pin_cite>" not in regex:
            return found
        if found is not None and found.groupdict().get("pin_cite"):
            return found
        flattened = [str(word) if isinstance(word, ParagraphToken) else word for word in words]
        if len(flattened) == len(words) and all(a is b for a, b in zip(flattened, words)):
            return found
        wider = original(flattened, start_index, regex, prefix, strings_only, forward, **kwargs)
        if wider is None or not wider.groupdict().get("pin_cite"):
            return found
        # Only a separator may stand between a citation and its page. Reading
        # past a blank line otherwise reaches into the next paragraph and calls
        # whatever number starts it a pin cite -- `( Id. ¶ 10). As a part of her
        # role, for over two\n\n1 Defendant's request` ends on a footnote
        # marker -- and the span that comes back is not where the citation's
        # kind puts its pin cite, which is a citation that cannot be pointed at.
        if not _SEPARATOR_ONLY.fullmatch(wider.group(0)[: wider.start("pin_cite")]):
            return found
        return wider

    return scan


@contextlib.contextmanager
def relaxed_pin_cites(relaxation: Relaxation = Relaxation.BOUNDED) -> Iterator[None]:
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

    Resolution is swapped too, for the reason `_tolerant_check` gives: reading
    the pin cite is not enough if the check that accepts it counts spaces.
    """
    patterns = {
        (eyecite.regexes, "PIN_CITE_REGEX"): relax(eyecite.regexes.PIN_CITE_REGEX, relaxation),
    }
    if relaxation is Relaxation.FULL:
        patterns[(eyecite.helpers, "match_on_tokens")] = _across_paragraphs(eyecite.helpers.match_on_tokens)
    for name in _BAKED:
        widened = relax(getattr(eyecite.helpers, name), relaxation)
        patterns[(eyecite.regexes, name)] = widened
        patterns[(eyecite.helpers, name)] = widened
    with patched(
        patterns,
        pin_cite_check=(
            eyecite.resolve,
            "_has_invalid_pin_cite",
            _tolerant_check(eyecite.resolve._has_invalid_pin_cite),
        ),
    ):
        yield
