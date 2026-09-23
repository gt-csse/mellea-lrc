"""Optional, iterative review of case names missed by the leaf reader.

The root and leaf growth stages remain deterministic. This stage reviews only
name-shaped sites that neither growth admitted, and writes each decision before
looking for the next site. An accepted bare name is consequently in the mask
and available as a leaf to subsequent reviews.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.extraction.adjudication.candidates.case_name_sites import case_name_sites
from mellea_lrc.extraction.adjudication.review.case_name import Reading, adjudicate_case_name
from mellea_lrc.extraction.reading.unread_names import unread_case_names
from mellea_lrc.extraction.structure.leaf_growth import LEAF_GROWTH_STAGE
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import CitationField, CitationKind, ReferenceCitation
from mellea_lrc.model.findings import Finding, FindingKind
from mellea_lrc.model.operations import (
    assign_root,
    create_citation,
    field_values,
    update_field,
    update_fields,
)
from mellea_lrc.model.record import Node, Reads
from mellea_lrc.serialization.ivr import serialize_site_review

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.extraction.adjudication.types import Candidate
    from mellea_lrc.model.document import Document


STAGE = "leaf_case_name_hunting"
_MADE_BY = "mellea_lrc.extraction.adjudication.review.case_name"


def _next_node_id(document: Document, site: Candidate) -> str:
    base = f"leaf_name_site:{site.span.start}-{site.span.end}"
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


def _finding(document: Document, site: Candidate, node: Node, reason: str | None) -> Document:
    return document.evolve(
        nodes=(*document.nodes, node),
        findings=(
            *document.findings,
            Finding(
                kind=FindingKind.SITE_REVIEW,
                stage=STAGE,
                made_by=_MADE_BY,
                message=reason or site.note or "The name site was reviewed without admitting a citation.",
                node_id=node.node_id,
                span=site.span,
            ),
        ),
    )


async def hunt_leaf_case_names(
    document: Document,
    *,
    session: MelleaSession | None = None,
) -> Document:
    """Review unread names after deterministic leaf growth, one site at a time.

    A model may repair the name of an existing citation or attach a bare-name
    leaf to an active root. Declines and non-citation readings remain findings.
    This stage does not revise root identity or run retrieval.
    """
    if LEAF_GROWTH_STAGE not in document.passes:
        msg = "Leaf case-name hunting requires leaf_growth."
        raise ValueError(msg)
    if STAGE in document.passes:
        return document

    current = document.snapshot()
    inspected: set[tuple[int, int]] = set()
    while True:
        site = next(
            (
                candidate
                for candidate in case_name_sites(current)
                if (candidate.span.start, candidate.span.end) not in inspected
            ),
            None,
        )
        if site is None:
            return current.evolve(passes=_after(current))
        inspected.add((site.span.start, site.span.end))
        review = await adjudicate_case_name(current, site, session=session)
        # A parseable last answer from an exhausted IVR run is still a failed
        # review; it must remain trace evidence without changing citation state.
        answer = review.answer if review.run.success else None
        node = Node(
            node_id=_next_node_id(current, site),
            reads=Reads.DOCUMENT,
            stage=STAGE,
            made_by=_MADE_BY,
            outcome=answer.reading.value if answer is not None else "declined",
            message=review.reason or site.note or None,
            details=serialize_site_review(site, review),
        )
        if answer is None:
            current = _finding(current, site, node, review.reason)
            continue
        if (
            answer.span.start < 0
            or answer.span.end > len(current.text)
            or current.text[answer.span.start : answer.span.end] != answer.name
        ):
            current = _finding(
                current,
                site,
                replace(node, outcome="ungrounded"),
                "The reviewed name was not grounded in the filing.",
            )
            continue

        name = CaseName(
            span=answer.span,
            text=answer.name,
            plaintiff=answer.plaintiff,
            defendant=answer.defendant,
        )
        if answer.reading is Reading.NAMES_A_CITATION:
            record = next(
                (item for item in current.citations if item.citation_id == answer.citation_id), None
            )
            if record is None or record.withdrawn:
                current = _finding(current, site, node, review.reason)
                continue
            update_field(
                record, node, CitationField.CASE_NAME, name, reason=node.message or "Case-name site review."
            )
            current = current.evolve(
                unread_case_names=unread_case_names(current.text, current.citations),
            )
            continue

        if answer.reading is Reading.SHORT_FORM:
            root = next(
                (
                    item
                    for item in current.citations
                    if item.citation_id == answer.root_id and item.is_root and not item.withdrawn
                ),
                None,
            )
            if root is None:
                current = _finding(current, site, node, review.reason)
                continue
            citation_id = f"case_name:{answer.span.start}"
            if any(item.citation_id == citation_id for item in current.citations):
                current = _finding(current, site, node, "The name is already an admitted citation.")
                continue
            reference = create_citation(citation_id, CitationKind.REFERENCE, node)
            update_fields(
                reference,
                node,
                field_values(
                    ReferenceCitation(
                        span=answer.span,
                        locator_span=answer.span,
                        matched_text=answer.name,
                        case_name=name,
                        plaintiff=answer.plaintiff,
                        defendant=answer.defendant,
                    )
                ),
                reason="Initial reviewed bare-name leaf reading.",
            )
            reference = assign_root(reference, root.citation_id, node)
            citations = tuple(sorted((*current.citations, reference), key=lambda item: item.full_span.start))
            current = current.evolve(
                citations=citations,
                unread_case_names=unread_case_names(current.text, citations),
            )
            continue

        current = _finding(current, site, node, review.reason)
