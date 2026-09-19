"""Decision-year field validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.citations import DocketCitation, FullCaseCitation
from mellea_lrc.validation.types import (
    CandidateEvaluationSource,
    FieldCheckOutcome,
    ValidationNodeStatus,
    YearCheckNode,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CandidateEvaluationNode, CitationValidation


def run_year_check(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
) -> YearCheckNode:
    """Compare extracted and retrieved decision years for one found locator."""
    citation = validation.citation.stated
    date = citation.date if isinstance(citation, (FullCaseCitation, DocketCitation)) else None
    # The check compares years; a citation stating a full date states its year too.
    extracted = date.year if date else None
    retrieved = candidate.year
    if (
        isinstance(citation, DocketCitation)
        and candidate.source is CandidateEvaluationSource.DOCKET_SEARCH
        and extracted is not None
        and retrieved is not None
    ):
        # A docket citation can be accompanied by an unpublished-opinion date,
        # while the docket endpoint supplies the date the case was filed.  The
        # same written year therefore refers to different events, so treating
        # unequal values as identity evidence would reject a correct docket.
        status = ValidationNodeStatus.SKIPPED
        outcome = FieldCheckOutcome.UNAVAILABLE
        status_message = "Skipped docket year comparison because the retrieved date is the filing date."
        outcome_message = (
            "A docket citation's decision date and CourtListener docket dateFiled identify different events."
        )
    elif extracted is None or retrieved is None:
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
        node_id=f"{candidate.node_id}:year_check",
        status=status,
        outcome=outcome,
        extracted_year=extracted,
        retrieved_year=retrieved,
        depends_on=(candidate.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
    )
