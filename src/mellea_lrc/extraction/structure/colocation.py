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

A second refusal is about the text rather than about the cases. In a table of
authorities eyecite gives every entry a full span running to the end of the
table, so two entries coincide by span while the page shows them on different
lines::

    Donovan v. City of Dallas , 377 U.S. 408 (1964)……… 6  Gucci America , 768 F.3d 122

Those are two cases, and nothing about the identifiers says so. What says so is
what lies between them: a leader, a page number and another case name. So the
locators of a co-located set must have nothing between them that begins another
citation -- no leader dots, no `v.`. In a real parallel citation the locators
are separated by a comma, a pin cite, a judge's initials or a short
parenthetical, never by more than that.

Measured over the 26 documents of `false-citation-bench`: **31 groups covering
64 citations**, every one a genuine parallel citation. Sixteen pair a docket
number with the reporter or database page written beside it; the rest are a
state reporter beside its regional reporter, or the three Supreme Court
reporters together, and are concentrated in two filings, because citing the
official and regional reporter together is a jurisdiction's house style rather
than a property of briefs in general.

The locator output is unaffected: this reads the spans extraction produced and
writes an id onto each citation. Nothing about which citations are found, or
where, changes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.extraction.types import ExtractedCitation

# What kind of thing a citation names. Only a citation that names an authority
# outright can be one of several identifiers for it -- a short form or an `id.`
# is a reference to an authority, not another name for one.
#
# A docket citation names a case, which is why it sits with the reporter ones.
# `In re Iovate Health Scis. Int'l Inc. , No. 25-11958 (MG), 2025 Bankr. LEXIS
# 2284` is one authority written twice, and seven citations on this corpus are
# that shape; without this each was counted as two authorities.
_NAMES = {
    "FullCaseCitation": "case",
    "DocketCitation": "case",
    "FullLawCitation": "law",
    "FullJournalCitation": "journal",
}


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


# What separates one citation from the next: the leader dots of an index, or the
# `v.` of another case name. Between two identifiers for one case there is a
# comma, a pin cite, a judge's initials or a short parenthetical, and nothing of
# this kind.
_ANOTHER_CITATION = re.compile(r"…|\.{2,}|\bvs?\.")


def _co_located(text: str, left: ExtractedCitation, right: ExtractedCitation) -> bool:
    """Whether two citations occupy the same span, to within a character or two."""
    if (
        abs(left.full_span.start - right.full_span.start) > _SPAN_SLACK
        or abs(left.full_span.end - right.full_span.end) > _SPAN_SLACK
    ):
        return False
    first, second = sorted((left, right), key=lambda c: c.locator_span.start)
    between = text[first.locator_span.end : second.locator_span.start]
    return not _ANOTHER_CITATION.search(between)


def colocation_groups(text: str, citations: Sequence[ExtractedCitation]) -> list[list[ExtractedCitation]]:
    """Return each set of two or more citations occupying the same place.

    A group is built by overlap and then rejected if any reporter appears twice
    in it, so a group is always a set of distinct identifiers for what may be
    one authority.
    """
    eligible = [c for c in citations if type(c.citation).__name__ in _NAMES]
    ordered = sorted(eligible, key=lambda c: (c.full_span.start, c.full_span.end))

    groups: list[list[ExtractedCitation]] = []
    for citation in ordered:
        if groups and any(_co_located(text, citation, member) for member in groups[-1]):
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
        # One kind of thing named: a statute is not another name for a case,
        # however close it sits, and overlap grouped the two before this test
        # existed. A docket and a reporter page *are* two names for one case, so
        # the test is on what is named rather than on the citation's type.
        and len({_NAMES[type(member.citation).__name__] for member in group}) == 1
    ]


def assign_colocation(text: str, citations: Sequence[ExtractedCitation]) -> tuple[ExtractedCitation, ...]:
    """Return the citations with a shared `colocation_id` on each co-located set.

    The id is the citation id of the group's first member, which makes it stable
    against re-running and readable when a serialized document is inspected by
    hand. A citation in no group keeps `None`, which is the common case.
    """
    from dataclasses import replace

    assigned: dict[str, str] = {}
    for group in colocation_groups(text, citations):
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
