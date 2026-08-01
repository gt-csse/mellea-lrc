"""Terminal citation summary across opinion and RECAP search candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.aggregation.citation_summary_candidate import (
    citation_summary_candidate,
)
from mellea_lrc.validation.aggregation.citation_summary_outcome import (
    overall_citation_outcome,
)
from mellea_lrc.validation.types import (
    CandidateProvenance,
    OpinionSearchCandidateAssessmentNode,
    RecapSearchCandidateAssessmentNode,
    SearchCitationSummaryNode,
    SearchCitationSummaryOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation


def run_search_citation_summary(
    validation: CitationValidation,
) -> SearchCitationSummaryNode:
    """List every completed opinion- and RECAP-search candidate assessment."""
    opinion_assessments = tuple(
        node for node in validation.nodes if isinstance(node, OpinionSearchCandidateAssessmentNode)
    )
    recap_assessments = tuple(
        node for node in validation.nodes if isinstance(node, RecapSearchCandidateAssessmentNode)
    )
    candidates = (
        *(
            citation_summary_candidate(
                validation,
                assessment,
                provenance=CandidateProvenance.OPINION,
            )
            for assessment in opinion_assessments
        ),
        *(
            citation_summary_candidate(
                validation,
                assessment,
                provenance=CandidateProvenance.RECAP,
            )
            for assessment in recap_assessments
        ),
    )
    return SearchCitationSummaryNode(
        node_id=f"{validation.citation_id}:search_citation_summary",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=SearchCitationSummaryOutcome.COMPLETE,
        overall_outcome=overall_citation_outcome(candidate.outcome for candidate in candidates),
        candidates=candidates,
        depends_on=tuple(candidate.assessment_node_id for candidate in candidates),
        status_message="Search-candidate summary completed.",
        outcome_message=(
            f"Listed {len(opinion_assessments)} opinion-search and "
            f"{len(recap_assessments)} RECAP-search candidate assessments "
            "without selecting a final citation."
        ),
    )
