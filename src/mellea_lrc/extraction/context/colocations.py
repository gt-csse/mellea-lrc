"""Convert source-only colocation groups into document colocation records."""

from __future__ import annotations

from collections.abc import Sequence

from mellea_lrc.model.colocation import Colocation
from mellea_lrc.parsing.colocations import colocation_groups
from mellea_lrc.parsing.span import Span


def colocation_readings(
    source: str,
    sites: Sequence[tuple[str, Span, str | None]],
    maximum_gap: int,
    *,
    barriers: Sequence[Span] = (),
) -> tuple[Colocation, ...]:
    """Use the same grouping grammar as the standalone field reader."""
    return tuple(
        Colocation(id=f"colocation:{group[0]}", citation_ids=group)
        for group in colocation_groups(source, sites, maximum_gap, barriers=barriers)
    )
