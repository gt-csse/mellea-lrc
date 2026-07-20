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
    from mellea_lrc.validation.types import CitationValidation, ExactLocatorLookupNode


def run_exact_case_name_check(
    validation: CitationValidation,
    *,
    lookup: ExactLocatorLookupNode,
) -> ExactCaseNameCheckNode:
    """Compare normalized extracted and retrieved case names exactly."""
    citation = validation.citation.citation
    extracted = _extracted_case_name(citation) if isinstance(citation, FullCaseCitation) else None
    retrieved = lookup.record.case_name if lookup.record is not None else None
    if extracted is None or retrieved is None:
        status = ValidationNodeStatus.SKIPPED
        outcome = FieldCheckOutcome.UNAVAILABLE
    else:
        status = ValidationNodeStatus.SUCCEEDED
        outcome = (
            FieldCheckOutcome.MATCH
            if _normalize_case_name(extracted) == _normalize_case_name(retrieved)
            else FieldCheckOutcome.MISMATCH
        )
    return ExactCaseNameCheckNode(
        node_id=f"{validation.citation_id}:exact_case_name_check",
        status=status,
        outcome=outcome,
        extracted_case_name=extracted,
        retrieved_case_name=retrieved,
        depends_on=(lookup.node_id,),
    )


def _extracted_case_name(citation: FullCaseCitation) -> str | None:
    if not citation.plaintiff or not citation.defendant:
        return None
    return f"{citation.plaintiff} v. {citation.defendant}"


def _normalize_case_name(value: str) -> str:
    return " ".join(value.split()).casefold()
