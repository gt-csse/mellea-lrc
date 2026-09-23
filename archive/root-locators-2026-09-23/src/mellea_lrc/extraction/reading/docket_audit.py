"""Audit docket candidates after grouping, independently of writing courts.

A docket is admitted when its group is followed by an explicit court or
contains a reporter/database locator. The latter can be checked by citation
lookup even when no court is stated. Other candidates are withdrawn with an
explanation; their records and raw locator spans remain available.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.extraction.reading.post_citation import docket_court, docket_parenthetical_context
from mellea_lrc.model.citations import DocketCitation, FullCaseCitation
from mellea_lrc.model.operations import observe_citation, withdraw_citation
from mellea_lrc.model.record import WITHDRAWN, Node, Reads

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.model.record import CitationRecord

MADE_BY = "mellea_lrc.extraction.reading.docket_audit"


def audit_docket_citations(text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]:
    """Record an admission or withdrawal without populating any court fields."""
    audited: list[CitationRecord] = []
    for item in citations:
        if not isinstance(item.fields, DocketCitation) or item.withdrawn:
            audited.append(item)
            continue
        court = docket_court(text, item, citations)
        parenthetical = docket_parenthetical_context(text, item, citations)
        reporters = tuple(
            other.citation_id
            for other in citations
            if item.colocation_id
            and other.colocation_id == item.colocation_id
            and not other.withdrawn
            and isinstance(other.fields, FullCaseCitation)
        )
        if court is not None:
            reason = "explicit_court"
            message = "An explicit court follows this docket's locator group."
        elif reporters:
            reason = "colocated_reporter"
            message = "A colocated reporter or database locator supports citation lookup."
        elif parenthetical:
            reason = "citation_parenthetical"
            message = "A date or U.S. parenthetical supports this courtless docket citation."
        else:
            reason = "no_citation_context"
            message = "No explicit court or colocated reporter supports this docket candidate."

        admitted = court is not None or bool(reporters) or parenthetical
        node = Node(
            node_id=f"{item.citation_id}:docket_audit:{reason}",
            reads=Reads.DOCUMENT,
            stage="extraction",
            made_by=MADE_BY,
            outcome="accepted" if admitted else WITHDRAWN,
            message=message,
            details={
                "reason": reason,
                "supporting_citation_ids": list(reporters),
                "court_span": (
                    {"start": court.span_start, "end": court.span_end} if court is not None else None
                ),
            },
        )
        updated = replace(item)
        if admitted:
            observe_citation(updated, node)
        else:
            withdraw_citation(updated, node)
        audited.append(updated)
    return tuple(audited)
