"""Terminal aggregation for candidates assessed from CourtListener searches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.aggregation.citation_summary_candidate import citation_summary_candidate
from mellea_lrc.validation.aggregation.citation_summary_outcome import overall_citation_outcome
from mellea_lrc.validation.types import (
    CandidateProvenance,
    CitationSummaryCandidate,
    OpinionSearchCandidateAssessmentNode,
    RecapSearchCandidateAssessmentNode,
    SearchCitationSummaryNode,
    SearchCitationSummaryOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation


def run_search_citation_summary(validation: CitationValidation) -> SearchCitationSummaryNode:
    """List assessed opinion and RECAP search candidates without selecting a winner."""
    assessments = tuple(
        node
        for node in validation.nodes
        if isinstance(
            node,
            (OpinionSearchCandidateAssessmentNode, RecapSearchCandidateAssessmentNode),
        )
    )
    candidates = tuple(_summary_entry(validation, assessment) for assessment in assessments)
    opinion_count = sum(entry.provenance is CandidateProvenance.OPINION for entry in candidates)
    recap_count = len(candidates) - opinion_count
    return SearchCitationSummaryNode(
        node_id=f"{validation.citation_id}:search_citation_summary",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=SearchCitationSummaryOutcome.COMPLETE,
        overall_outcome=overall_citation_outcome(entry.outcome for entry in candidates),
        candidates=candidates,
        depends_on=tuple(entry.assessment_node_id for entry in candidates),
        status_message="Search-candidate summary completed.",
        outcome_message=(
            f"Listed {opinion_count} opinion-search and {recap_count} RECAP-search candidate "
            "assessments without selecting a final citation."
        ),
    )


def _summary_entry(
    validation: CitationValidation,
    assessment: OpinionSearchCandidateAssessmentNode | RecapSearchCandidateAssessmentNode,
) -> CitationSummaryCandidate:
    provenance = (
        CandidateProvenance.OPINION
        if isinstance(assessment, OpinionSearchCandidateAssessmentNode)
        else CandidateProvenance.RECAP
    )
    return citation_summary_candidate(
        validation,
        assessment,
        provenance=provenance,
    )
