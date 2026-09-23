"""An opt-in, iterative docket-locator hunting plugin.

This module is intentionally outside ``grow_roots``. A caller can place it
before validation, after validation, or omit it, without changing deterministic
extraction. When it admits a locator, it immediately writes that locator so the
next candidate sees the updated locator mask. Co-location is a separate, single
projection after hunting ends. It never runs the docket audit: model admission
and a court/context audit are separate operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.extraction.adjudication.candidates.docket_sites import SuspectedDocket, suspected_dockets
from mellea_lrc.extraction.adjudication.promotion import promote_docket_locator
from mellea_lrc.extraction.adjudication.review.docket import (
    RecoveredDocketLocator,
    adjudicate_docket,
)
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind, SiteReview
from mellea_lrc.extraction.locator_stages import DOCKET_SITE_STAGE
from mellea_lrc.extraction.rules import ExtractionRules
from mellea_lrc.model.findings import Finding, FindingKind
from mellea_lrc.model.operations import observe_citation
from mellea_lrc.model.record import Node, Reads
from mellea_lrc.serialization import serialize_site_review

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.model.document import Document

STAGE = DOCKET_SITE_STAGE
MADE_BY = "mellea_lrc.extraction.adjudication.review.docket"
DECLINED = "declined"


def _candidate(site: SuspectedDocket) -> Candidate:
    return Candidate(
        generator="suspected_dockets",
        kind=CandidateKind.DOCKET,
        span=site.locator_span,
        window=site.context_span,
        note="Unread opaque docket candidate; an explicit label may be absent beside a reporter locator.",
    )


def _node(
    document: Document,
    site: SuspectedDocket,
    review: SiteReview[RecoveredDocketLocator],
    *,
    outcome: str,
) -> Node:
    return Node(
        node_id=_next_node_id(document, site),
        reads=Reads.DOCUMENT,
        stage=STAGE,
        made_by=MADE_BY,
        outcome=outcome,
        message=review.reason or None,
        details=serialize_site_review(_candidate(site), review),
    )


def _next_node_id(document: Document, site: SuspectedDocket) -> str:
    """Retain each review when a declined site is reviewed again later."""
    base = f"docket_site:{site.locator_span.start}-{site.locator_span.end}"
    held = {node.node_id for node in document.nodes}
    held.update(node.node_id for record in document.citations for node in record.trace)
    if base not in held:
        return base
    attempt = 2
    while f"{base}:{attempt}" in held:
        attempt += 1
    return f"{base}:{attempt}"


def _after(document: Document) -> tuple[str, ...]:
    return document.passes if STAGE in document.passes else (*document.passes, STAGE)


def apply_docket_site_review(
    document: Document,
    site: SuspectedDocket,
    review: SiteReview[RecoveredDocketLocator],
    *,
    rules: ExtractionRules | None = None,
) -> Document:
    """Record one docket-site review and return the updated document.

    An admission creates a minimal :class:`DocketCitation`. The next candidate
    sees its locator through the ordinary mask, but this function does not form
    co-location or roots, and it does not invoke court, date, case-name,
    pin-cite, or docket-audit readers. A decline is retained as a document-
    level finding because the inspected text is not a citation record.
    """
    if review.answer is None:
        node = _node(document, site, review, outcome=DECLINED)
        return document.evolve(
            nodes=(*document.nodes, node),
            findings=(
                *document.findings,
                Finding(
                    kind=FindingKind.SITE_REVIEW,
                    stage=STAGE,
                    made_by=MADE_BY,
                    message=review.reason or site.locator_text,
                    node_id=node.node_id,
                    span=site.locator_span,
                ),
            ),
            passes=_after(document),
        )

    record = promote_docket_locator(document.text, site, review.answer)
    node = _node(document, site, review, outcome="accepted")
    observe_citation(record, node)
    citations = tuple(sorted((*document.citations, record), key=lambda item: item.full_span.start))

    # ``suspected_dockets`` computes its next candidate from locator spans, so
    # this record participates in masking immediately without recomputing the
    # unrelated co-location projection.
    return document.evolve(citations=citations, passes=_after(document))


async def hunt_docket_locators(
    document: Document,
    *,
    session: MelleaSession | None = None,
    rules: ExtractionRules | None = None,
) -> Document:
    """Review each currently-unread docket site, updating the document per move.

    The next proposal is generated only after the previous decision has been
    written. An admitted locator is consequently masked before a later site is
    considered; a declined span is retained in the local inspected set so it is
    not asked twice in the same run.
    """
    inspected: set[tuple[int, int]] = set()
    current = document
    while True:
        site = next(
            (
                proposed
                for proposed in suspected_dockets(current)
                if (proposed.locator_span.start, proposed.locator_span.end) not in inspected
            ),
            None,
        )
        if site is None:
            return current.evolve(passes=_after(current))
        inspected.add((site.locator_span.start, site.locator_span.end))
        review = await adjudicate_docket(site, session=session)
        current = apply_docket_site_review(current, site, review, rules=rules)
