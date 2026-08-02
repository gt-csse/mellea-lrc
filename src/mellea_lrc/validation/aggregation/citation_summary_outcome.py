"""Reduction of one route's candidate assessments into a citation conclusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.types import (
    CitationSummaryAssessmentOutcome,
    LocatorCandidateAssessmentOutcome,
    SearchCandidateAssessmentOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def overall_locator_citation_outcome(
    candidate_outcomes: Iterable[LocatorCandidateAssessmentOutcome],
) -> CitationSummaryAssessmentOutcome | None:
    """Return the strongest conclusion across all locator-route candidates.

    A unique-locator route can confirm a match, so it may report ``match``,
    ``possible_match``, or ``mismatch``.
    """
    normalized = tuple(_normalize_locator_outcome(outcome) for outcome in candidate_outcomes)
    if not normalized:
        return None
    return max(normalized, key=_locator_outcome_rank)


def overall_search_citation_outcome(
    candidate_outcomes: Iterable[SearchCandidateAssessmentOutcome],
) -> CitationSummaryAssessmentOutcome:
    """Return the conclusion across all search-route candidates.

    A search can never confirm a match or a mismatch, only surface a
    possible match - so this reports ``possible_match`` when any candidate
    is a possible match, and ``not_found`` otherwise (including when there
    are no candidates at all).
    """
    if any(outcome is SearchCandidateAssessmentOutcome.POSSIBLE_MATCH for outcome in candidate_outcomes):
        return CitationSummaryAssessmentOutcome.POSSIBLE_MATCH
    return CitationSummaryAssessmentOutcome.NOT_FOUND


def _normalize_locator_outcome(
    outcome: LocatorCandidateAssessmentOutcome,
) -> CitationSummaryAssessmentOutcome:
    if outcome is LocatorCandidateAssessmentOutcome.MATCH:
        return CitationSummaryAssessmentOutcome.MATCH
    if outcome is LocatorCandidateAssessmentOutcome.PARTIAL_MATCH:
        return CitationSummaryAssessmentOutcome.POSSIBLE_MATCH
    return CitationSummaryAssessmentOutcome.MISMATCH


def _locator_outcome_rank(outcome: CitationSummaryAssessmentOutcome) -> int:
    return {
        CitationSummaryAssessmentOutcome.MISMATCH: 0,
        CitationSummaryAssessmentOutcome.POSSIBLE_MATCH: 1,
        CitationSummaryAssessmentOutcome.MATCH: 2,
    }[outcome]
