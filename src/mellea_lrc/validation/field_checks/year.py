"""Decision-year field validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.validation.types import FieldCheckOutcome, ValidationNodeStatus, YearCheckNode

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation, ExactLocatorLookupNode


async def run_year_check(
    validation: CitationValidation,
    *,
    lookup: ExactLocatorLookupNode,
) -> YearCheckNode:
    """Compare extracted and retrieved decision years for one found locator."""
    citation = validation.citation.citation
    extracted = citation.year if isinstance(citation, FullCaseCitation) else None
    retrieved = lookup.record.year if lookup.record is not None else None
    if extracted is None or retrieved is None:
        status = ValidationNodeStatus.SKIPPED
        outcome = FieldCheckOutcome.UNAVAILABLE
    else:
        status = ValidationNodeStatus.SUCCEEDED
        outcome = FieldCheckOutcome.MATCH if extracted == retrieved else FieldCheckOutcome.MISMATCH
    return YearCheckNode(
        node_id=f"{validation.citation_id}:year_check",
        status=status,
        outcome=outcome,
        extracted_year=extracted,
        retrieved_year=retrieved,
        depends_on=(lookup.node_id,),
    )
