"""Tests for reducing candidate assessment outcomes into one citation conclusion."""

from mellea_lrc.validation.aggregation.citation_summary_outcome import (
    overall_locator_citation_outcome,
    overall_search_citation_outcome,
)
from mellea_lrc.validation.types import (
    CitationSummaryAssessmentOutcome,
    LocatorCandidateAssessmentOutcome,
    SearchCandidateAssessmentOutcome,
)


def test_locator_outcome_prefers_the_strongest_candidate() -> None:
    """A confirmed locator match wins over possible and mismatched candidates."""
    outcome = overall_locator_citation_outcome(
        (
            LocatorCandidateAssessmentOutcome.MISMATCH,
            LocatorCandidateAssessmentOutcome.PARTIAL_MATCH,
            LocatorCandidateAssessmentOutcome.MATCH,
        )
    )

    assert outcome is CitationSummaryAssessmentOutcome.MATCH


def test_locator_outcome_normalizes_partial_match() -> None:
    """A partial match reduces to the shared possible-match conclusion."""
    outcome = overall_locator_citation_outcome(
        (
            LocatorCandidateAssessmentOutcome.MISMATCH,
            LocatorCandidateAssessmentOutcome.PARTIAL_MATCH,
        )
    )

    assert outcome is CitationSummaryAssessmentOutcome.POSSIBLE_MATCH


def test_locator_outcome_is_absent_without_assessed_candidates() -> None:
    """An empty terminal list has no citation-level conclusion."""
    assert overall_locator_citation_outcome(()) is None


def test_search_outcome_reports_possible_match_when_any_candidate_is_possible() -> None:
    """A search route can only ever confirm a possible match, never a match."""
    outcome = overall_search_citation_outcome(
        (
            SearchCandidateAssessmentOutcome.MISMATCH,
            SearchCandidateAssessmentOutcome.POSSIBLE_MATCH,
        )
    )

    assert outcome is CitationSummaryAssessmentOutcome.POSSIBLE_MATCH


def test_search_outcome_reports_not_found_when_every_candidate_mismatches() -> None:
    """A search route can never confirm a mismatch, only report not found."""
    outcome = overall_search_citation_outcome(
        (
            SearchCandidateAssessmentOutcome.MISMATCH,
            SearchCandidateAssessmentOutcome.MISMATCH,
        )
    )

    assert outcome is CitationSummaryAssessmentOutcome.NOT_FOUND


def test_search_outcome_reports_not_found_without_any_candidates() -> None:
    """No search candidates at all is also not found, not an absent conclusion."""
    assert overall_search_citation_outcome(()) is CitationSummaryAssessmentOutcome.NOT_FOUND
