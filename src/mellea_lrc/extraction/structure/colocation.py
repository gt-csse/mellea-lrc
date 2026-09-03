"""Group citations that occupy the same place in the text.

A filing citing an authority in parallel writes one citation and several
identifiers for it::

    St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262

eyecite extracts three full citations there, one per reporter, and links them
to nothing. Downstream every count that is per-authority is then wrong: this
corpus reports about 4% more authorities than it has, and a claim about the
case attaches to whichever reporter happened to come last.

**This reports co-location. It does not decide identity.** Citations whose full
spans coincide are grouped and given a shared id; whether they name one case is
a question for validation, which can resolve each against CourtListener and
compare the opinion cluster. That division matters, because co-location alone
cannot settle it:

    See Brown, 347 U.S. 483, 349 U.S. 294 (1955).

has one case name, one year parenthetical and identical spans, and is two
decisions. A rule deciding identity here would merge them; a rule reporting
candidacy hands both to a layer that can tell.

One refusal is applied, because it needs no lookup and cannot be wrong: **two
citations sharing a reporter are two cases**, since a case has one first page
in one reporter. That is what separates Brown I from Brown II, and it is
deliberately the only judgement made here.

Measured over the 26 documents of `false-citation-bench`: **14 groups covering
30 of 694 full citations**, every one a genuine parallel citation -- a state
reporter beside its regional reporter, or the three Supreme Court reporters
together. Thirteen of the fourteen are in two filings, because citing the
official and regional reporter together is a jurisdiction's house style rather
than a property of briefs in general, so the rate here says little about the
rate elsewhere.

The locator output is unaffected: adding this leaves the run artifact
byte-identical. Nothing about which citations are found, or where, changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.extraction.types import ExtractedCitation

# Only a citation that names an authority by volume, reporter and page can be
# one of several identifiers for it. A short form or an `id.` is a reference to
# an authority, not another name for one.
_COLOCATABLE = frozenset({"FullCaseCitation", "FullLawCitation", "FullJournalCitation"})


def _reporter(citation: ExtractedCitation) -> str:
    """The citation's reporter, normalised so spacing does not split a group."""
    return "".join(str(getattr(citation.citation, "reporter", "") or "").split()).lower()


# Parallel citations share a full span to within a character: eyecite yields
# 11-78, 11-78 and 12-78 for one sentence. Requiring exact equality would leave
# the third out of its own group; allowing mere overlap admits a different
# failure, because a long full span swallows the citation after it. On this
# corpus overlap grouped `501 U.S. 32` with `28 U.S.C. § 1927` and `869 F.2d
# 688` -- a case, a statute and another case, 239 characters apart at the start
# and sharing only an end.
_SPAN_SLACK = 2


def _co_located(left: ExtractedCitation, right: ExtractedCitation) -> bool:
    """Whether two citations occupy the same span, to within a character or two."""
    return (
        abs(left.span.start - right.span.start) <= _SPAN_SLACK
        and abs(left.span.end - right.span.end) <= _SPAN_SLACK
    )


def colocation_groups(citations: Sequence[ExtractedCitation]) -> list[list[ExtractedCitation]]:
    """Return each set of two or more citations occupying the same place.

    A group is built by overlap and then rejected if any reporter appears twice
    in it, so a group is always a set of distinct identifiers for what may be
    one authority.
    """
    eligible = [c for c in citations if type(c.citation).__name__ in _COLOCATABLE]
    ordered = sorted(eligible, key=lambda c: (c.span.start, c.span.end))

    groups: list[list[ExtractedCitation]] = []
    for citation in ordered:
        if groups and any(_co_located(citation, member) for member in groups[-1]):
            groups[-1].append(citation)
        else:
            groups.append([citation])

    return [
        group
        for group in groups
        if len(group) > 1
        # Distinct reporters: a case has one first page in one reporter, so a
        # repeat means two authorities, not two names for one.
        and len({_reporter(member) for member in group}) == len(group)
        # One kind: a statute is not another name for a case, however close it
        # sits. Overlap grouped the two before this test existed.
        and len({type(member.citation).__name__ for member in group}) == 1
    ]


def assign_colocation(citations: Sequence[ExtractedCitation]) -> tuple[ExtractedCitation, ...]:
    """Return the citations with a shared `colocation_id` on each co-located set.

    The id is the citation id of the group's first member, which makes it stable
    against re-running and readable when a serialized document is inspected by
    hand. A citation in no group keeps `None`, which is the common case.
    """
    from dataclasses import replace

    assigned: dict[str, str] = {}
    for group in colocation_groups(citations):
        identifier = group[0].citation_id
        for member in group:
            assigned[member.citation_id] = identifier

    if not assigned:
        return tuple(citations)
    return tuple(
        replace(citation, colocation_id=assigned[citation.citation_id])
        if citation.citation_id in assigned
        else citation
        for citation in citations
    )
