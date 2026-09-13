r"""Which root a leaf points at, decided from `stated`.

A leaf -- a short form, an `Id.`, a `supra`, a bare name -- means whichever root
it points at, and nothing in the characters at it says which. eyecite decides
that while parsing, from the parse: it matches a short form's antecedent guess
against the party names *it* read. So does every reading that follows.

**This decides it from `stated` instead**, and that is the whole reason the
leaves are grown in a second pass. `source` is what the rules read and never
changes; `stated` is the citation as it is now, after a reader has corrected a
name it truncated and after validation has checked that name against the
authority. A root eyecite named `Cnty.` is a root no short form can find; the
same root named `Huri v. Office of the Chief Judge of the Cir. Ct. of Cook
Cnty.` is one that `Huri , 804 F.3d at 833` reaches. Attaching against the
parse throws that away, so nothing here reads `source`.

## What each kind is matched by

``ShortCaseCitation``
    The volume and reporter it states, against the roots that state the same.
    One candidate is the answer. Several -- a case reported at `477 U.S. 242`
    and another at `477 U.S. 317`, both cited -- are narrowed by the name it
    writes, and then by the page: of the roots that begin at or before the page
    claimed, the last one holds it.

``SupraCitation`` and ``ReferenceCitation``
    The name alone, since neither states an identifier.

``IdCitation``
    Position. `Id.` means the authority of the citation before it, so it takes
    the root of the nearest citation that precedes it -- which may be another
    leaf, already attached.

## What is not decided here

Nothing is guessed. A leaf that matches no root, or more than one and cannot be
narrowed, gets `None` and is not grown: `CitationRecord` refuses a leaf without
a root, so an undecided leaf is a leaf that does not exist rather than one
attached to the wrong case.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import CitationKind, citation_kind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.core.citations import CanonicalCitation
    from mellea_lrc.core.record import CitationRecord

_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")
_NOT_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
# Words in every other case name, which identify none of them.
_EMPTY = frozenset(
    {
        "v", "vs", "in", "re", "the", "of", "and", "et", "al", "ex", "rel",
        "inc", "llc", "co", "corp", "ltd", "lp", "llp", "pc", "pllc", "no",
        "company", "ass", "assn", "dept", "bd", "comm", "cnty", "dist",
    }
)
#: How far past a root's first page a pin cite may fall and still be its page.
#: eyecite's own bound, reused so the two do not disagree.
MAX_PAGES = 150


def _words(name: str | None) -> set[str]:
    return {
        word.lower().rstrip("'’")
        for word in _WORD.findall(name or "")
        if word.lower().rstrip("'’") not in _EMPTY and len(word) > 2
    }


def _names(citation: CanonicalCitation) -> set[str]:
    """Every significant word the citation is written under.

    The whole name is preferred over the parsed parties, and they are read only
    when there is no name. eyecite fills `plaintiff` and `defendant` from the
    words in front of the citation, which are often the words of the citation
    before it: the `Bell Atl. Corp. v. Twombly , 550 U.S. 544` two sentences
    after `Ashcroft v. Iqbal` is parsed with `defendant='Iqbal'`. Reading the
    parties alongside the name would let `Iqbal , supra` reach Twombly.
    """
    name = citation.case_name
    text = getattr(name, "text", None)
    parts = (
        [text]
        if text
        else [
            getattr(name, "plaintiff", None),
            getattr(name, "defendant", None),
            getattr(citation, "plaintiff", None),
            getattr(citation, "defendant", None),
        ]
    )
    parts.append(getattr(citation, "antecedent", None))
    return _words(" ".join(part for part in parts if part))


def _reporter(citation: CanonicalCitation) -> str:
    reporter = getattr(citation, "reporter", None)
    written = getattr(reporter, "canonical", None) or str(reporter or "")
    return _NOT_ALPHANUMERIC.sub("", written.lower())


def _page(citation: CanonicalCitation) -> int | None:
    page = getattr(citation, "page", None)
    return int(page) if page and str(page).isdigit() else None


def _claimed(citation: CanonicalCitation) -> int | None:
    """The first page a leaf claims, which is what tells two volumes apart."""
    pin = getattr(citation, "pin_cite", None)
    for pages in getattr(pin, "pages", ()) or ():
        if pages.first is not None:
            return pages.first
    return _page(citation)


def _by_locator(leaf: CanonicalCitation, roots: Sequence[CitationRecord]) -> list[CitationRecord]:
    volume, reporter = getattr(leaf, "volume", None), _reporter(leaf)
    if not volume or not reporter:
        return []
    return [
        root
        for root in roots
        if getattr(root.stated, "volume", None) == volume and _reporter(root.stated) == reporter
    ]


def _holding_the_page(leaf: CanonicalCitation, candidates: Sequence[CitationRecord]):
    """Of the roots in this volume, the last one that begins at or before the page.

    `477 U.S. at 254` is written in a filing that cites both `Anderson v.
    Liberty Lobby , 477 U.S. 242` and `Celotex , 477 U.S. 317`, and volume and
    reporter alone cannot tell them apart. Page 254 is inside the first and
    before the second, so it is the first's.
    """
    page = _claimed(leaf)
    if page is None:
        return None
    reachable = [
        root
        for root in candidates
        if (start := _page(root.stated)) is not None and start <= page <= start + MAX_PAGES
    ]
    if not reachable:
        return None
    return max(reachable, key=lambda root: _page(root.stated) or 0)


def _by_name(leaf: CanonicalCitation, roots: Sequence[CitationRecord]) -> list[CitationRecord]:
    wanted = _names(leaf)
    if not wanted:
        return []
    return [root for root in roots if wanted & _names(root.stated)]


def root_for(
    leaf: CanonicalCitation,
    roots: Sequence[CitationRecord],
    *,
    before: Sequence[CitationRecord] = (),
) -> str | None:
    """The `citation_id` of the root this leaf points at, or `None`.

    `roots` is every root in the document -- order is not a test, because a
    filing may write a short form before the citation it shortens and a reader
    can still reach the case. `before` is the citations already attached that
    precede this one, in the order they are written, which is what an `Id.`
    means by "the one before".
    """
    kind = citation_kind(leaf)

    if kind is CitationKind.ID:
        for earlier in reversed(before):
            if earlier.root_id:
                root = next((r for r in roots if r.citation_id == earlier.root_id), None)
                # An `Id.` claiming a page its antecedent cannot hold is not
                # that antecedent's, and this reader does not guess whose it is.
                page, start = _claimed(leaf), _page(root.stated) if root else None
                if root is None or page is None or start is None:
                    return earlier.root_id
                return earlier.root_id if start <= page <= start + MAX_PAGES else None
        return None

    if kind in {CitationKind.SUPRA, CitationKind.REFERENCE}:
        named = _by_name(leaf, roots)
        return named[0].citation_id if len(named) == 1 else None

    candidates = _by_locator(leaf, roots)
    if len(candidates) == 1:
        return candidates[0].citation_id
    if not candidates:
        return None
    named = _by_name(leaf, candidates)
    if len(named) == 1:
        return named[0].citation_id
    holding = _holding_the_page(leaf, named or candidates)
    return holding.citation_id if holding is not None else None
