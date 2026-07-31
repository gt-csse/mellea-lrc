"""Shared reduction of candidate assessments into one citation conclusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.types import (
    CandidateAssessmentOutcome,
    CitationSummaryAssessmentOutcome,
    LocatorCandidateAssessmentOutcome,
    SearchCandidateAssessmentOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def overall_citation_outcome(
    candidate_outcomes: Iterable[CandidateAssessmentOutcome],
) -> CitationSummaryAssessmentOutcome | None:
    """Return the strongest conclusion across all assessed candidates."""
    normalized = tuple(_normalize_candidate_outcome(outcome) for outcome in candidate_outcomes)
    if not normalized:
        return None
    return max(normalized, key=_outcome_rank)


def _normalize_candidate_outcome(
    outcome: CandidateAssessmentOutcome,
) -> CitationSummaryAssessmentOutcome:
    if outcome is LocatorCandidateAssessmentOutcome.MATCH:
        return CitationSummaryAssessmentOutcome.MATCH
    if outcome in (
        LocatorCandidateAssessmentOutcome.PARTIAL_MATCH,
        SearchCandidateAssessmentOutcome.POSSIBLE_MATCH,
    ):
        return CitationSummaryAssessmentOutcome.POSSIBLE_MATCH
    return CitationSummaryAssessmentOutcome.MISMATCH


def _outcome_rank(outcome: CitationSummaryAssessmentOutcome) -> int:
    return {
        CitationSummaryAssessmentOutcome.MISMATCH: 0,
        CitationSummaryAssessmentOutcome.POSSIBLE_MATCH: 1,
        CitationSummaryAssessmentOutcome.MATCH: 2,
    }[outcome]
