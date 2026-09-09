"""Group a document's citations by the root they all refer to.

A **root** is an identifier the filing states -- a claim about which case it
means, which a lookup can check. It is not an **authority**: an authority is the
identity a root is established to have, and establishing it is validation's
work, not this module's.

A filing does not state a root once. It cites it in full, then returns to
it as `Id. at 570`, `550 U.S. at 563`, or by party name -- and each of those
return visits usually names a **different page** and attaches a **different
proposition** to it. Validating only the full citations therefore checks one
claim per root and skips every other claim made about it, which is the
larger part of what a brief actually asserts.

The tree makes that structure explicit. Every citation resolves, transitively,
to the citation that stated its identifier; that identifier is looked up once
and the work is shared; and each occurrence keeps its own pin cite, so it
is its own checkable claim about its own page rather than an alias of the first
one.

Two properties are what make this worth having rather than merely tidy:

- **Identity is resolved once per root, not once per occurrence.** Ten
  references to one case cost one lookup.
- **Pinpoint claims multiply.** A case cited at pages 563, 570 and 578 is
  three distinct claims about three distinct pages, each verifiable on its own
  and each capable of being wrong on its own.

Resolution here is eyecite's, followed transitively: `Id.` may point at a short
form that points at the full citation. Nothing is invented -- a citation eyecite
could not attribute is reported rather than guessed at, because attaching a
claim to the wrong root would check it against the wrong page.

What is *reported* matters as much as what is resolved, and two failures that
look alike in a count mean opposite things:

- **out of scope** -- positive evidence that the citation names something other
  than a case: it is a statute or a journal article, it is a span eyecite could
  not parse as a citation at all, or it is an `id.` that resolved to one of
  those. There is no case root for these to belong to, and grouping them
  under one would be wrong. On false-citation-bench this is 252 of 917, and
  every one is correct behaviour.
- **unattributed** -- no such evidence. Either a case citation that could not
  be traced to the full citation introducing it, or a reference that needed an
  antecedent and reached none, so its kind is unknown. This is the number that
  measures the tree, and on the same corpus it is 2.

Only positive evidence sends a citation out of scope, which is the same rule
the rest of the project applies to absence: not knowing what something refers
to is not evidence that it refers to a statute.

Read individually, the 2 are:

- one `ShortCaseCitation`, and it is real -- `Rosenblatt v. Baer, 383 U.S. at
  85`, quoted inside another case's parenthetical and never given in full
  anywhere in the document. There is no antecedent, so declining is right.
- one `Id. ¶¶26-28`, pointing into the opposing party's motions rather than
  into any case. There is no root for it because it is not citing a case.

A root may be stated as a docket number as well as a reporter locator, and that is what the second figure measures. Without it the same
corpus strands 17 rather than 2, fifteen of them one chain of `Id. ¶ N` in a
declaration, every one pointing at a paragraph of an indictment cited as
`No. 1:25-cr-00312-RPK (E.D.N.Y.)`. An earlier reading of this file guessed
those were references into the filing's own numbered allegations. They were
not; they were a root the extractor had no type for.

Reporting all of this as one figure would read as a 28% failure rate for what
is, in case citations, one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import (
    CitationKind,
    DocketCitation,
    FullCaseCitation,
    ShortCaseCitation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument

# A chain longer than this is a resolution loop or a pathology, not a brief.
MAX_RESOLUTION_DEPTH = 24

# Citations that carry no identity of their own and mean whatever they point at.
# One of these that resolves to nothing is a resolution failure, not a statute.
_REFERRING_KINDS = frozenset({CitationKind.ID, CitationKind.SUPRA, CitationKind.REFERENCE})

# What can stand at the head of a chain of references. A docket citation is
# here for the same reason a full case citation is: it names one case, once,
# and every later `Id.` is another claim about that same document.
_ROOT_KINDS = (FullCaseCitation, DocketCitation)

# Citations that name a case, whether or not they name it completely. These are
# never out of scope: what a short form or a docket points at is a case.
_CASE_KINDS = (FullCaseCitation, ShortCaseCitation, DocketCitation)


@dataclass(frozen=True, slots=True)
class CitationOccurrence:
    """One place a document refers to a root, with the page it names."""

    citation: ExtractedCitation
    depth: int
    pin_cite: str | None

    @property
    def citation_id(self) -> str:
        """The extracted citation's identifier."""
        return self.citation.citation_id

    @property
    def is_root(self) -> bool:
        """Whether this occurrence is the citation that stated the identifier."""
        return self.depth == 0


@dataclass(frozen=True, slots=True)
class Root:
    """One root -- an identifier the filing states -- and every place it returns to it."""

    root: ExtractedCitation
    occurrences: tuple[CitationOccurrence, ...]

    @property
    def root_id(self) -> str:
        """The id of the citation that stated this identifier first."""
        return self.root.citation_id

    @property
    def pin_cites(self) -> tuple[str, ...]:
        """Every distinct pin cite claimed under this root, in order of appearance.

        This is the count that matters for verification: it is how many separate
        claims about separate pages the document makes about one case.
        """
        seen: dict[str, None] = {}
        for occurrence in self.occurrences:
            if occurrence.pin_cite:
                seen.setdefault(occurrence.pin_cite, None)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class CitationTree:
    """Every root a document states, and what could not be attributed to one."""

    roots: tuple[Root, ...]
    unattributed: tuple[ExtractedCitation, ...]
    """References with no positive evidence of being out of scope, and no root.

    A case citation whose full form was never found, or a reference that needed
    an antecedent and reached none. The second kind has an unknown citation
    type, which is why it is here rather than in `out_of_scope`.
    """
    out_of_scope: tuple[ExtractedCitation, ...] = ()
    """Citations that are not to a case, so no case root could hold them."""

    @property
    def occurrence_count(self) -> int:
        """How many citation occurrences were attributed to a root."""
        return sum(len(root.occurrences) for root in self.roots)

    @property
    def pinpoint_claim_count(self) -> int:
        """How many distinct page-level claims the document makes.

        Larger than the number of roots whenever a brief returns to a case for a
        second proposition, which is most of the time.
        """
        return sum(len(root.pin_cites) for root in self.roots)


def assign_roots(citations: Sequence[ExtractedCitation]) -> tuple[ExtractedCitation, ...]:
    """Return the citations, each carrying the root it refers to.

    The same resolution `build_citation_tree` performs, written onto the
    citations so the answer survives serialization and so a consumer need not
    rebuild the chain. A citation the resolution cannot place keeps `None`,
    which is the tree's answer too.
    """
    by_id = {item.citation_id: item for item in citations}
    assigned: dict[str, str | None] = {}
    for item in citations:
        root_id, _ = _resolve_root(item, by_id)
        root = by_id.get(root_id) if root_id else None
        placed = root is not None and isinstance(root.citation, _ROOT_KINDS)
        assigned[item.citation_id] = root_id if placed else None
    return tuple(
        replace(item, root_id=assigned[item.citation_id]) if assigned[item.citation_id] else item
        for item in citations
    )


def build_citation_tree(document: ExtractedDocument) -> CitationTree:
    """Group a document's citations under the authorities they refer to."""
    by_id = {item.citation_id: item for item in document.citations}
    roots: dict[str, list[CitationOccurrence]] = {}
    unattributed: list[ExtractedCitation] = []
    out_of_scope: list[ExtractedCitation] = []

    for item in document.citations:
        root_id, depth = _resolve_root(item, by_id)
        root = by_id.get(root_id) if root_id else None
        if root is not None and isinstance(root.citation, _ROOT_KINDS):
            roots.setdefault(root_id or "", []).append(
                CitationOccurrence(citation=item, depth=depth, pin_cite=_pin_cite(item))
            )
        elif _is_out_of_scope(item, root):
            out_of_scope.append(item)
        else:
            unattributed.append(item)

    grouped = tuple(
        Root(root=by_id[root_id], occurrences=tuple(occurrences)) for root_id, occurrences in roots.items()
    )
    return CitationTree(
        roots=grouped,
        unattributed=tuple(unattributed),
        out_of_scope=tuple(out_of_scope),
    )


def _is_out_of_scope(item: ExtractedCitation, root: ExtractedCitation | None) -> bool:
    """Whether there is positive evidence this citation names something other than a case.

    Only positive evidence counts, and it comes from exactly two places: the
    citation is itself of a non-case kind, or it resolved to one. A statute is
    a statute on its own evidence; an `id.` that resolved to a statute is a
    statute by what it stands for.

    An `id.` that resolved to *nothing* is neither. It carries no reporter, so
    it cannot be typed on its own, and there is no antecedent to type it by --
    which is a resolution failure, and every failure of that kind belongs in
    `unattributed` where it will be counted. Calling it out of scope would
    assert that it did not name a case, which is precisely what is unknown.
    """
    if isinstance(item.citation, _CASE_KINDS):
        return False
    if not _needs_an_antecedent(item):
        return True
    reached_something_else = root is not None and root.citation_id != item.citation_id
    if not reached_something_else:
        return False
    return not isinstance(root.citation, _CASE_KINDS)


def _needs_an_antecedent(item: ExtractedCitation) -> bool:
    """Whether this citation carries no identity of its own and must resolve to one."""
    return item.citation.kind in _REFERRING_KINDS


def _resolve_root(
    item: ExtractedCitation,
    by_id: Mapping[str, ExtractedCitation],
) -> tuple[str | None, int]:
    """Follow `resolves_to` to the citation that stated the identifier.

    `Id.` often points at a short form rather than at the full citation, so the
    chain is walked rather than read once. A cycle or an implausible chain
    yields no root, which sends the citation to `unattributed` -- guessing would
    attach a page claim to the wrong case.
    """
    current = item
    seen = {current.citation_id}
    for depth in range(MAX_RESOLUTION_DEPTH):
        target = current.resolves_to
        if target is None:
            return (current.citation_id, depth)
        if target in seen:
            return (None, depth)
        following = by_id.get(target)
        if following is None:
            return (None, depth)
        seen.add(target)
        current = following
    return (None, MAX_RESOLUTION_DEPTH)


def _pin_cite(item: ExtractedCitation) -> str | None:
    pin = getattr(item.citation, "pin_cite", None)
    return str(pin) if pin else None


def summarize(trees: Sequence[CitationTree]) -> dict[str, int]:
    """Count what a set of documents' trees contain, for reporting."""
    return {
        "roots": sum(len(tree.roots) for tree in trees),
        "occurrences": sum(tree.occurrence_count for tree in trees),
        "pinpoint_claims": sum(tree.pinpoint_claim_count for tree in trees),
        "unattributed": sum(len(tree.unattributed) for tree in trees),
        "out_of_scope": sum(len(tree.out_of_scope) for tree in trees),
    }
