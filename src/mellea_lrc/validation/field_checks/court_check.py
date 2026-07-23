"""Court identifier field validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.validation.types import CourtCheckNode, FieldCheckOutcome, ValidationNodeStatus

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation, DocketCourtRetrievalNode


def run_court_check(
    validation: CitationValidation,
    *,
    retrieval: DocketCourtRetrievalNode,
) -> CourtCheckNode:
    """Compare Eyecite's normalized court ID with the docket's court ID."""
    citation = validation.citation.citation
    extracted = citation.court if isinstance(citation, FullCaseCitation) else None
    retrieved = retrieval.court_id
    if extracted is None or retrieved is None:
        status = ValidationNodeStatus.SKIPPED
        outcome = FieldCheckOutcome.UNAVAILABLE
    else:
        status = ValidationNodeStatus.SUCCEEDED
        outcome = FieldCheckOutcome.MATCH if extracted == retrieved else FieldCheckOutcome.MISMATCH
    return CourtCheckNode(
        node_id=f"{validation.citation_id}:court_check",
        status=status,
        outcome=outcome,
        extracted_court_id=extracted,
        retrieved_court_id=retrieved,
        depends_on=(retrieval.node_id,),
    )
