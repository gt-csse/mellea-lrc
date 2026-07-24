"""Decision-year field validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.validation.types import FieldCheckOutcome, ValidationNodeStatus, YearCheckNode

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation, ExactLocatorLookupNode


def run_year_check(
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
        status_message = "Skipped year comparison because required evidence is missing."
        outcome = FieldCheckOutcome.UNAVAILABLE
        outcome_message = "Year comparison is unavailable because one year is missing."
    else:
        status = ValidationNodeStatus.SUCCEEDED
        status_message = "Year comparison completed."
        outcome = FieldCheckOutcome.MATCH if extracted == retrieved else FieldCheckOutcome.MISMATCH
        outcome_message = (
            "Extracted and retrieved years match."
            if outcome is FieldCheckOutcome.MATCH
            else "Extracted and retrieved years differ."
        )
    return YearCheckNode(
        node_id=f"{validation.citation_id}:year_check",
        status=status,
        outcome=outcome,
        extracted_year=extracted,
        retrieved_year=retrieved,
        depends_on=(lookup.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
    )
