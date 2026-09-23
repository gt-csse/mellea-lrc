"""Review uncertain page claims after the document's leaves have grown.

This stage asks about one existing citation at a time. Each answer is recorded
before the next site is proposed, so the next review sees the current fields.
The review reads the filing only; checking a retrieved page's support for a
proposition is a separate validation question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.extraction.adjudication.candidates.pin_cite_sites import pin_cite_sites
from mellea_lrc.extraction.adjudication.review.pin_cite import (
    AdjudicatedPinCite,
    adjudicate_pin_cite,
)
from mellea_lrc.extraction.adjudication.types import Candidate, SiteReview
from mellea_lrc.extraction.structure.leaf_growth import LEAF_GROWTH_STAGE
from mellea_lrc.model.citations import CitationField
from mellea_lrc.model.operations import observe_citation, update_field
from mellea_lrc.model.record import Node, Reads
from mellea_lrc.serialization.ivr import serialize_site_review

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.model.document import Document

PIN_CITE_SITE_VALIDATION_STAGE = "pin_cite_site_validation"
MADE_BY = "mellea_lrc.extraction.adjudication.review.pin_cite"


def _site_key(site: Candidate) -> tuple[str | None, int, int]:
    return site.about, site.span.start, site.span.end


def _node_id(document: Document, site: Candidate) -> str:
    """Keep every attempt if a checkpoint is reviewed again later."""
    base = f"pin_cite_site:{site.about}:{site.span.start}-{site.span.end}"
    held = {node.node_id for node in document.nodes}
    held.update(node.node_id for record in document.citations for node in record.trace)
    if base not in held:
        return base
    attempt = 2
    while f"{base}:{attempt}" in held:
        attempt += 1
    return f"{base}:{attempt}"


def _apply_review(
    document: Document,
    site: Candidate,
    review: SiteReview[AdjudicatedPinCite],
) -> Document:
    """Materialize one review on the cited record and retain its full IVR run."""
    record = next((item for item in document.active_citations if item.citation_id == site.about), None)
    if record is None:
        return document

    # A parseable last answer does not override a failed instruct/validate/repair
    # run. It remains evidence, but cannot change the filing's stated page.
    answer = review.answer if review.run.success else None
    if answer is not None and answer.citation_id != record.citation_id:
        answer = None
    outcome = (
        answer.reading.value if answer is not None else ("failed" if not review.run.success else "declined")
    )
    node = Node(
        node_id=_node_id(document, site),
        reads=Reads.DOCUMENT,
        stage=PIN_CITE_SITE_VALIDATION_STAGE,
        made_by=MADE_BY,
        outcome=outcome,
        message=review.reason or site.note,
        details=serialize_site_review(site, review),
    )
    if answer is not None and answer.pin_cite != record.fields.pin_cite:
        update_field(
            record,
            node,
            CitationField.PIN_CITE,
            answer.pin_cite,
            reason=review.reason or site.note,
        )
    else:
        observe_citation(record, node)
    return document


async def validate_pin_cite_sites(
    document: Document,
    *,
    session: MelleaSession | None = None,
) -> Document:
    """Check uncertain page claims, returning a checkpointable document.

    The stage requires structural leaf growth, since those leaves may also
    claim pages. It records a completed pass even when no sites are proposed.
    A repeated call on a completed checkpoint does no work. Each new site is
    proposed against the document after the preceding decision was applied.
    """
    if LEAF_GROWTH_STAGE not in document.passes:
        msg = "Pin-cite site validation requires leaf_growth."
        raise ValueError(msg)
    if PIN_CITE_SITE_VALIDATION_STAGE in document.passes:
        return document

    current = document.snapshot()
    inspected: set[tuple[str | None, int, int]] = set()
    while True:
        active_ids = {record.citation_id for record in current.active_citations}
        site = next(
            (
                candidate
                for candidate in pin_cite_sites(current)
                if candidate.about in active_ids and _site_key(candidate) not in inspected
            ),
            None,
        )
        if site is None:
            return current.evolve(passes=(*current.passes, PIN_CITE_SITE_VALIDATION_STAGE))
        inspected.add(_site_key(site))
        record = next(item for item in current.active_citations if item.citation_id == site.about)
        review = await adjudicate_pin_cite(current.text, site, record, session=session)
        current = _apply_review(current, site, review)
