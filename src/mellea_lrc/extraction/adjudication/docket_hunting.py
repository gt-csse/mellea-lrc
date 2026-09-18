"""An opt-in, iterative docket-locator hunting plugin.

This module is intentionally outside ``grow_roots``. A caller can place it
before validation, after validation, or omit it, without changing deterministic
extraction. When it admits a locator, it immediately re-runs the ordinary root
field passes so the next candidate sees the updated locator mask and
co-location structure. It never runs the docket audit: model admission and a
court/context audit are separate operations.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.core.findings import Finding, FindingKind
from mellea_lrc.core.record import Node, Reads
from mellea_lrc.extraction.adjudication.candidates.docket_sites import SuspectedDocket, suspected_dockets
from mellea_lrc.extraction.adjudication.promotion import promote_docket_locator
from mellea_lrc.extraction.adjudication.review.docket import (
    RecoveredDocketLocator,
    adjudicate_docket,
)
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind, SiteReview
from mellea_lrc.extraction.reading.unread_names import unread_case_names
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.stages import refine
from mellea_lrc.serialization import serialize_site_review

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.extraction.types import Document

STAGE = "docket_site_hunting"
MADE_BY = "mellea_lrc.extraction.adjudication.review.docket"
DECLINED = "declined"


def _candidate(site: SuspectedDocket) -> Candidate:
    return Candidate(
        generator="suspected_dockets",
        kind=CandidateKind.DOCKET,
        span=site.locator_span,
        window=site.context_span,
        note="Explicit docket label followed by an opaque identifier outside the federal CM/ECF reader.",
    )


def _node(
    site: SuspectedDocket,
    review: SiteReview[RecoveredDocketLocator],
    *,
    outcome: str,
) -> Node:
    return Node(
        node_id=f"docket_site:{site.locator_span.start}-{site.locator_span.end}",
        reads=Reads.DOCUMENT,
        stage=STAGE,
        made_by=MADE_BY,
        outcome=outcome,
        message=review.reason or None,
        details=serialize_site_review(_candidate(site), review),
    )


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

    An admission creates a minimal :class:`DocketCitation` and then re-runs
    co-location, court, date, pin-cite, and root assignment. The review itself
    does not select any of those fields. A decline is retained as a
    document-level finding because the inspected text is not a citation record.
    """
    if review.answer is None:
        node = _node(site, review, outcome=DECLINED)
        return replace(
            document,
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
    node = _node(site, review, outcome="accepted")
    record.observe(node)
    citations = tuple(
        sorted((*document.citations, record), key=lambda item: item.full_span.start)
    )

    # The ordinary deterministic readers own every field following locator
    # admission. In particular, no docket audit is slipped back in here: it is
    # a distinct, optional stage and would erase the review's admission signal.
    root_rules = replace(stable(rules), docket_auditor=None)
    refined = refine(document.text, citations, root_rules)
    return replace(
        document,
        citations=refined,
        unread_case_names=unread_case_names(document.text, refined),
        passes=_after(document),
    )


async def hunt_docket_locators(
    document: Document,
    *,
    session: MelleaSession | None = None,
    rules: ExtractionRules | None = None,
) -> Document:
    """Review each currently-unread docket site, updating the document per move.

    The next proposal is generated only after the previous decision has been
    written. An admitted root is consequently masked and participates in
    co-location before a later site is considered; a declined span is retained
    in the local inspected set so it is not asked twice in the same run.
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
            return current
        inspected.add((site.locator_span.start, site.locator_span.end))
        review = await adjudicate_docket(site, session=session)
        current = apply_docket_site_review(current, site, review, rules=rules)
