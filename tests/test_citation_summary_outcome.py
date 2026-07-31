"""Tests for reducing candidate assessment outcomes into one citation conclusion."""

from mellea_lrc.validation.aggregation.citation_summary_outcome import overall_citation_outcome
from mellea_lrc.validation.types import (
    CitationSummaryAssessmentOutcome,
    LocatorCandidateAssessmentOutcome,
    SearchCandidateAssessmentOutcome,
)


def test_overall_citation_outcome_prefers_the_strongest_candidate() -> None:
    """A confirmed locator match wins over possible and mismatched candidates."""
    outcome = overall_citation_outcome(
        (
            SearchCandidateAssessmentOutcome.MISMATCH,
            LocatorCandidateAssessmentOutcome.PARTIAL_MATCH,
            LocatorCandidateAssessmentOutcome.MATCH,
        )
    )

    assert outcome is CitationSummaryAssessmentOutcome.MATCH


def test_overall_citation_outcome_normalizes_possible_candidate_results() -> None:
    """Both candidate routes contribute the shared possible-match conclusion."""
    outcome = overall_citation_outcome(
        (
            LocatorCandidateAssessmentOutcome.MISMATCH,
            LocatorCandidateAssessmentOutcome.PARTIAL_MATCH,
            SearchCandidateAssessmentOutcome.POSSIBLE_MATCH,
        )
    )

    assert outcome is CitationSummaryAssessmentOutcome.POSSIBLE_MATCH


def test_overall_citation_outcome_is_absent_without_assessed_candidates() -> None:
    """An empty terminal list has no citation-level conclusion."""
    assert overall_citation_outcome(()) is None
