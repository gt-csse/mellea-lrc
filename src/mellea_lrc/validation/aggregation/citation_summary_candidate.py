"""Frontend-ready candidate projections shared by terminal citation summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from mellea_lrc.courtlistener.docket_models import courtlistener_docket_url
from mellea_lrc.courtlistener.opinion_models import CourtListenerOpinionCluster, courtlistener_opinion_url
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    CandidateProvenance,
    CitationSummaryCandidate,
    CitationSummaryPinpoint,
    LocatorCandidateAssessmentNode,
    MelleaCitingPropositionExtractionNode,
    MelleaPinpointCheckNode,
    OpinionSearchCandidateAssessmentNode,
    RecapSearchCandidateAssessmentNode,
    ReporterPageRetrievalNode,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation, ValidationNode

NodeT = TypeVar("NodeT", bound="ValidationNode")

CandidateAssessmentNode = (
    LocatorCandidateAssessmentNode | OpinionSearchCandidateAssessmentNode | RecapSearchCandidateAssessmentNode
)


def citation_summary_candidate(
    validation: CitationValidation,
    assessment: CandidateAssessmentNode,
    *,
    provenance: CandidateProvenance,
) -> CitationSummaryCandidate:
    """Project one candidate assessment into the shared citation-summary row."""
    return CitationSummaryCandidate(
        provenance=provenance,
        candidate_index=assessment.candidate_index,
        assessment_node_id=assessment.node_id,
        outcome=assessment.outcome,
        extracted_citation=assessment.extracted_citation,
        extracted_case_name=assessment.extracted_case_name,
        retrieved_case_name=assessment.retrieved_case_name,
        case_name_outcome=assessment.case_name_outcome,
        case_name_evidence=assessment.case_name_evidence,
        extracted_year=assessment.extracted_year,
        retrieved_year=assessment.retrieved_year,
        year_outcome=assessment.year_outcome,
        extracted_court_id=assessment.extracted_court_id,
        retrieved_court_id=assessment.retrieved_court_id,
        court_outcome=assessment.court_outcome,
        docket_id=assessment.docket_id,
        opinion_url=_opinion_url(validation, assessment, provenance=provenance),
        docket_url=_docket_url(validation, assessment, provenance=provenance),
        pinpoint=_pinpoint_summary(validation, assessment),
    )


def _opinion_url(
    validation: CitationValidation,
    assessment: CandidateAssessmentNode,
    *,
    provenance: CandidateProvenance,
) -> str | None:
    """Expose CourtListener's canonical public opinion URL only for opinion candidates."""
    if provenance is not CandidateProvenance.OPINION:
        return None
    if isinstance(assessment, LocatorCandidateAssessmentNode):
        evaluation = _locator_evaluation(validation, assessment)
        return (
            evaluation.record.opinion_url
            if evaluation is not None and isinstance(evaluation.record, CourtListenerOpinionCluster)
            else None
        )
    if isinstance(assessment, OpinionSearchCandidateAssessmentNode):
        evaluation = next(
            (
                node
                for node in validation.nodes
                if isinstance(node, CandidateEvaluationNode)
                and node.source is CandidateEvaluationSource.OPINION_SEARCH
                and _is_ancestor(validation, node.node_id, assessment.node_id)
            ),
            None,
        )
        absolute_url = (
            evaluation.record.get("absolute_url")
            if evaluation is not None and not isinstance(evaluation.record, CourtListenerOpinionCluster)
            else None
        )
        return courtlistener_opinion_url(absolute_url if isinstance(absolute_url, str) else None)
    return None


def _docket_url(
    validation: CitationValidation,
    assessment: CandidateAssessmentNode,
    *,
    provenance: CandidateProvenance,
) -> str | None:
    """Expose CourtListener's canonical docket URL for RECAP candidates."""
    if provenance is not CandidateProvenance.RECAP:
        return None
    evaluation = next(
        (
            node
            for node in validation.nodes
            if isinstance(node, CandidateEvaluationNode)
            and node.source is CandidateEvaluationSource.RECAP_SEARCH
            and _is_ancestor(validation, node.node_id, assessment.node_id)
        ),
        None,
    )
    if evaluation is None or isinstance(evaluation.record, CourtListenerOpinionCluster):
        return None
    absolute_url = evaluation.record.get("docket_absolute_url")
    return courtlistener_docket_url(absolute_url if isinstance(absolute_url, str) else None)


def _pinpoint_summary(
    validation: CitationValidation,
    assessment: CandidateAssessmentNode,
) -> CitationSummaryPinpoint | None:
    if not isinstance(assessment, LocatorCandidateAssessmentNode):
        return None
    evaluation = _locator_evaluation(validation, assessment)
    if evaluation is None:
        return None
    retrieval = _single_child(validation, ReporterPageRetrievalNode, evaluation.node_id)
    if retrieval is None:
        return None
    proposition = _single_child(
        validation,
        MelleaCitingPropositionExtractionNode,
        retrieval.node_id,
    )
    if proposition is None:
        return None
    pinpoint = next(
        (
            node
            for node in validation.nodes
            if isinstance(node, MelleaPinpointCheckNode)
            and retrieval.node_id in node.depends_on
            and proposition.node_id in node.depends_on
        ),
        None,
    )
    if pinpoint is None:
        return None
    evidence = retrieval.evidence
    return CitationSummaryPinpoint(
        node_id=pinpoint.node_id,
        status=pinpoint.status,
        outcome=pinpoint.outcome,
        reporter_citation=retrieval.reporter_citation,
        pin_cite=retrieval.pin_cite,
        opinion_id=evidence.opinion_id if evidence is not None else None,
        opinion_type=evidence.opinion_type if evidence is not None else None,
        reporter_page_text=evidence.text if evidence is not None else None,
        citing_context_span=proposition.context_span,
        citation_span=validation.citation.full_span,
        proposition=proposition.proposition,
        proposition_span=proposition.proposition_span,
        reasoning=pinpoint.reasoning,
        evidence_quote=pinpoint.evidence_quote,
        evidence_span=pinpoint.evidence_span,
        evidence_match_method=pinpoint.evidence_match_method,
        evidence_match_score=pinpoint.evidence_match_score,
        status_message=pinpoint.status_message,
        outcome_message=pinpoint.outcome_message,
        error=pinpoint.error,
    )


def _locator_evaluation(
    validation: CitationValidation,
    assessment: LocatorCandidateAssessmentNode,
) -> CandidateEvaluationNode | None:
    return next(
        (
            node
            for node in validation.nodes
            if isinstance(node, CandidateEvaluationNode)
            and node.source is CandidateEvaluationSource.LOCATOR_LOOKUP
            and _is_ancestor(validation, node.node_id, assessment.node_id)
        ),
        None,
    )


def _single_child(
    validation: CitationValidation,
    node_type: type[NodeT],
    parent_id: str,
) -> NodeT | None:
    children = [
        node for node in validation.nodes if isinstance(node, node_type) and parent_id in node.depends_on
    ]
    return children[0] if len(children) == 1 else None


def _is_ancestor(
    validation: CitationValidation,
    ancestor_id: str,
    node_id: str,
) -> bool:
    by_id = {node.node_id: node for node in validation.nodes}

    def visit(current_id: str, visited: set[str]) -> bool:
        if current_id == ancestor_id:
            return True
        if current_id in visited:
            return False
        visited.add(current_id)
        current = by_id.get(current_id)
        return current is not None and any(
            visit(dependency_id, visited) for dependency_id in current.depends_on
        )

    return visit(node_id, set())
