"""Shared source-span grouping for full and short reporter stage writers."""

from __future__ import annotations

import re
from collections.abc import Sequence

from mellea_lrc.model.colocation import Colocation
from mellea_lrc.model.span import Span

_SEPARATE_CITATION = re.compile(r"…|\.{2,}|\bvs?\.|\n\s*\n|\.\s+[A-Z]", re.I)


def colocation_readings(
    source: str,
    sites: Sequence[tuple[str, Span, str | None]],
    maximum_gap: int,
    *,
    barriers: Sequence[Span] = (),
) -> tuple[Colocation, ...]:
    """Group adjacent sites, never crossing a barrier or repeating an edition.

    Each site supplies its citation ID, source span and normalized reporter
    edition (None for a docket or failed normalization). This reader neither
    updates citations nor makes identity or root-assignment decisions.
    """
    groups: list[list[tuple[str, Span, str | None]]] = []
    for site in sorted(sites, key=lambda item: item[1].start):
        if groups:
            previous = groups[-1][-1]
            between = source[previous[1].end : site[1].start]
            duplicate = site[2] is not None and any(member[2] == site[2] for member in groups[-1])
            interrupted = any(previous[1].end <= span.start < site[1].start for span in barriers)
            if (
                not duplicate
                and not interrupted
                and not _SEPARATE_CITATION.search(between)
                and sum(character.isalnum() for character in between) <= maximum_gap
            ):
                groups[-1].append(site)
                continue
        groups.append([site])
    return tuple(
        Colocation(id=f"colocation:{group[0][0]}", citation_ids=tuple(site[0] for site in group))
        for group in groups
        if len(group) > 1
    )
