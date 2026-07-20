"""Case-name validation-node construction and deterministic comparison."""

from __future__ import annotations

import unicodedata
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeAlias

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.validation.types import (
    CaseNameCheckOutcome,
    CaseNameCheckNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation, ExactLocatorLookupNode

SemanticCaseNameMatcher: TypeAlias = Callable[[str, str], Awaitable[bool]]

_TYPOGRAPHIC_TRANSLATION = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{SINGLE HIGH-REVERSED-9 QUOTATION MARK}": "'",
        "\N{PRIME}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{MINUS SIGN}": "-",
    }
)


async def run_case_name_check(
    validation: CitationValidation,
    *,
    lookup: ExactLocatorLookupNode,
    semantic_matcher: SemanticCaseNameMatcher,
) -> CaseNameCheckNode:
    """Compare case names exactly, then semantically when needed."""
    citation = validation.citation.citation
    extracted = build_extracted_case_name(citation) if isinstance(citation, FullCaseCitation) else None
    retrieved = lookup.record.case_name if lookup.record is not None else None
    status, outcome, error = await _compare_case_names(
        extracted,
        retrieved,
        semantic_matcher=semantic_matcher,
    )
    return CaseNameCheckNode(
        node_id=f"{validation.citation_id}:case_name_check",
        status=status,
        outcome=outcome,
        extracted_case_name=extracted,
        retrieved_case_name=retrieved,
        depends_on=(lookup.node_id,),
        error=error,
    )


async def _compare_case_names(
    extracted: str | None,
    retrieved: str | None,
    *,
    semantic_matcher: SemanticCaseNameMatcher,
) -> tuple[ValidationNodeStatus, CaseNameCheckOutcome, str | None]:
    if not retrieved:
        return ValidationNodeStatus.SKIPPED, CaseNameCheckOutcome.UNASSESSABLE, None
    if case_names_equivalent(extracted, retrieved):
        return ValidationNodeStatus.SUCCEEDED, CaseNameCheckOutcome.EXACT_MATCH, None
    if not extracted:
        return ValidationNodeStatus.SUCCEEDED, CaseNameCheckOutcome.NOT_SEMANTIC_MATCH, None

    try:
        is_semantic_match = await semantic_matcher(extracted, retrieved)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return ValidationNodeStatus.FAILED, CaseNameCheckOutcome.FAILED, error
    outcome = (
        CaseNameCheckOutcome.SEMANTIC_MATCH
        if is_semantic_match
        else CaseNameCheckOutcome.NOT_SEMANTIC_MATCH
    )
    return ValidationNodeStatus.SUCCEEDED, outcome, None


def build_extracted_case_name(citation: FullCaseCitation) -> str | None:
    """Build a display case name from the extracted party fields."""
    if citation.plaintiff and citation.defendant:
        return f"{citation.plaintiff} v. {citation.defendant}"
    return citation.plaintiff or citation.defendant


def case_names_equivalent(left: str | None, right: str | None) -> bool:
    """Return whether names match after removing typographic noise."""
    if not left or not right:
        return False
    return _normalize_case_name(left) == _normalize_case_name(right)


def _normalize_case_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(_TYPOGRAPHIC_TRANSLATION)
    return " ".join(normalized.split())
