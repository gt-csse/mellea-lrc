"""Exact case-name field validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.validation.types import (
    ExactCaseNameCheckNode,
    FieldCheckOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CandidateEvaluationNode, CitationValidation


def run_exact_case_name_check(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
) -> ExactCaseNameCheckNode:
    """Compare normalized extracted and retrieved case names exactly."""
    citation = validation.citation.citation
    extracted = _extracted_case_name(citation) if isinstance(citation, FullCaseCitation) else None
    retrieved = candidate.case_name
    if extracted is None or retrieved is None:
        status = ValidationNodeStatus.SKIPPED
        status_message = "Skipped exact case-name comparison because required evidence is missing."
        outcome = FieldCheckOutcome.UNAVAILABLE
        outcome_message = "Case-name comparison is unavailable because one case name is missing."
    else:
        status = ValidationNodeStatus.SUCCEEDED
        status_message = "Exact case-name comparison completed."
        outcome = (
            FieldCheckOutcome.MATCH
            if _normalize_case_name(extracted) == _normalize_case_name(retrieved)
            else FieldCheckOutcome.MISMATCH
        )
        outcome_message = (
            "Extracted and retrieved case names match after whitespace normalization."
            if outcome is FieldCheckOutcome.MATCH
            else "Extracted and retrieved case names differ after whitespace normalization."
        )
    return ExactCaseNameCheckNode(
        node_id=f"{validation.citation_id}:exact_case_name_check",
        status=status,
        outcome=outcome,
        extracted_case_name=extracted,
        retrieved_case_name=retrieved,
        depends_on=(candidate.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
    )


def _extracted_case_name(citation: FullCaseCitation) -> str | None:
    if not citation.plaintiff or not citation.defendant:
        return None
    return f"{citation.plaintiff} v. {citation.defendant}"


def _normalize_case_name(value: str) -> str:
    return " ".join(value.split()).casefold()
