"""Candidate assessment and terminal summary for one found locator."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerOpinionCluster
from mellea_lrc.model.citations import FullCaseCitation, Reporter
from mellea_lrc.validation.aggregation.citation_summary_candidate import (
    citation_summary_candidate,
)
from mellea_lrc.validation.aggregation.citation_summary_outcome import (
    overall_locator_citation_outcome,
)
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    CandidateProvenance,
    CitationSummaryCandidate,
    CourtCheckNode,
    DocketCourtRetrievalNode,
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorCandidateAssessmentNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorCitationSummaryOutcome,
    LocatorLookupOutcome,
    MelleaPinpointCheckOutcome,
    ValidationNodeStatus,
    YearCheckNode,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.candidates.state import CandidateValidationState
    from mellea_lrc.validation.types import CitationValidation


def run_locator_candidate_assessment(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
    state: CandidateValidationState,
) -> LocatorCandidateAssessmentNode:
    """Reduce one completed locator candidate subtree into a conclusion."""
    exact = _required_child(validation, ExactCaseNameCheckNode, candidate.node_id)
    year = _required_child(validation, YearCheckNode, candidate.node_id)
    court = _required_court_check(validation, candidate.node_id)
    case_name = state.require_case_name_result()
    year_outcome = _field_outcome(year.outcome)
    court_outcome = _field_outcome(court.outcome)
    if not locator_candidate_contains_reporter_locator(validation, candidate):
        outcome = LocatorCandidateAssessmentOutcome.MISMATCH
        outcome_message = (
            "The citation-lookup response does not contain the filing's exact volume, reporter, and page."
        )
    else:
        outcome = _assessment_outcome(case_name.outcome, year_outcome, court_outcome)
        outcome_message = _assessment_message(outcome, case_name=case_name.outcome, court=court_outcome)
    return LocatorCandidateAssessmentNode(
        node_id=f"{candidate.node_id}:locator_candidate_assessment",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        candidate_index=candidate.candidate_index,
        extracted_citation=validation.citation.matched_text,
        extracted_case_name=state.displayed_case_name(exact.extracted_case_name),
        retrieved_case_name=exact.retrieved_case_name,
        case_name_outcome=case_name.outcome,
        case_name_evidence=case_name.evidence,
        extracted_year=year.extracted_year,
        retrieved_year=year.retrieved_year,
        year_outcome=year_outcome,
        extracted_court_id=court.extracted_court_id,
        retrieved_court_id=court.retrieved_court_id,
        court_outcome=court_outcome,
        docket_id=candidate.docket_id,
        depends_on=(case_name.dependency_id, year.node_id, court.node_id),
        status_message="Locator candidate assessment completed.",
        outcome_message=outcome_message,
    )


def run_locator_citation_summary(validation: CitationValidation) -> LocatorCitationSummaryNode:
    """List every evaluated locator candidate without selecting one."""
    assessments = tuple(node for node in validation.nodes if isinstance(node, LocatorCandidateAssessmentNode))
    if not assessments:
        msg = "Locator citation summary requires at least one candidate assessment"
        raise ValueError(msg)
    candidates = tuple(
        citation_summary_candidate(
            validation,
            assessment,
            provenance=_candidate_provenance(validation, assessment),
        )
        for assessment in assessments
    )
    return LocatorCitationSummaryNode(
        node_id=f"{validation.citation_id}:locator_citation_summary",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorCitationSummaryOutcome.COMPLETE,
        overall_outcome=overall_locator_citation_outcome(candidate.outcome for candidate in candidates),
        pinpoint_requires_review=_pinpoint_requires_review(validation, candidates),
        candidates=candidates,
        depends_on=tuple(
            dependency
            for candidate in candidates
            for dependency in (
                candidate.assessment_node_id,
                *((candidate.pinpoint.node_id,) if candidate.pinpoint is not None else ()),
            )
        ),
        status_message="Locator citation summary completed.",
        outcome_message=(f"Listed {len(candidates)} evaluated locator candidates without selecting one."),
    )


def _pinpoint_requires_review(
    validation: CitationValidation,
    candidates: tuple[CitationSummaryCandidate, ...],
) -> bool | None:
    """Flag a matched unique locator whose pinpoint check did not support it."""
    lookup = next(
        (node for node in validation.nodes if isinstance(node, ExactLocatorLookupNode)),
        None,
    )
    # Temporary found-route-only aggregation: reporter-page and pinpoint work
    # currently run only below one exact locator result. Revisit this signal
    # when candidate aggregation gains pinpoint semantics for other routes.
    if lookup is None or lookup.outcome is not LocatorLookupOutcome.FOUND:
        return None
    if not any(candidate.pinpoint is not None for candidate in candidates):
        return None
    matched = tuple(
        candidate for candidate in candidates if candidate.outcome is LocatorCandidateAssessmentOutcome.MATCH
    )
    if not matched:
        return None
    return any(
        candidate.pinpoint is None or candidate.pinpoint.outcome is not MelleaPinpointCheckOutcome.SUPPORTS
        for candidate in matched
    )


def _assessment_outcome(
    case_name: AggregatedFieldOutcome,
    year: AggregatedFieldOutcome | None,
    court: AggregatedFieldOutcome,
) -> LocatorCandidateAssessmentOutcome:
    """Reduce case name, year, and court into one candidate conclusion.

    An outright case-name or court disagreement is a mismatch. Case name
    must be affirmatively confirmed to match - it is the identity anchor.
    Year and court only need to not actively disagree: a missing date has no
    judgment and does not affect identity.
    """
    if case_name is AggregatedFieldOutcome.MISMATCH:
        return LocatorCandidateAssessmentOutcome.MISMATCH
    if court is AggregatedFieldOutcome.MISMATCH:
        return LocatorCandidateAssessmentOutcome.MISMATCH
    if case_name is AggregatedFieldOutcome.MATCH and year is not AggregatedFieldOutcome.MISMATCH:
        return LocatorCandidateAssessmentOutcome.MATCH
    return LocatorCandidateAssessmentOutcome.PARTIAL_MATCH


def locator_candidate_contains_reporter_locator(
    validation: CitationValidation,
    candidate: CandidateEvaluationNode,
) -> bool:
    """Verify a provider's citation-lookup result against the requested locator.

    The CourtListener endpoint can return a nearby citation in the same
    reporter.  Its ``found`` status therefore means only that it produced a
    cluster, not that the cluster carries the source locator.  This check is
    deliberately limited to the exact reporter-lookup route; metadata and
    body-search records are not expected to contain reporter citations.
    """
    if candidate.source is not CandidateEvaluationSource.LOCATOR_LOOKUP:
        return True
    citation = validation.citation.fields
    if not isinstance(citation, FullCaseCitation):
        return False
    if not isinstance(candidate.record, CourtListenerOpinionCluster):
        return False
    if not candidate.record.citations:
        # Older/minimal endpoint responses occasionally omit the citation
        # array.  They cannot positively confirm the locator here, but they
        # also do not supply contradictory locator evidence.
        return True
    wanted_reporters = {_normalized_reporter(citation.reporter)}
    if isinstance(citation.reporter, Reporter):
        # The filing may use a recognized alternate spelling.  Eyecite's
        # canonical reporter and the spelling in the filing are both valid
        # representations of the same edition; the provider may return either.
        wanted_reporters.add(_normalized_reporter(citation.reporter.canonical))
    return any(
        item.volume == citation.volume
        and item.page == citation.page
        and _normalized_reporter(item.reporter) in wanted_reporters
        for item in candidate.record.citations
    )


def _normalized_reporter(value: object) -> str:
    """Compare reporter display forms while retaining volume and page exactly.

    This performs only punctuation and spacing normalization.  A recognized
    alternate spelling is compared separately through ``Reporter.canonical``;
    removing punctuation alone cannot equate Fed. Appx. with F. App'x.
    """
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _assessment_message(
    outcome: LocatorCandidateAssessmentOutcome,
    *,
    case_name: AggregatedFieldOutcome,
    court: AggregatedFieldOutcome,
) -> str:
    if outcome is LocatorCandidateAssessmentOutcome.MATCH:
        return "Case name, year, and court do not disagree with this candidate."
    if outcome is LocatorCandidateAssessmentOutcome.PARTIAL_MATCH:
        if case_name is not AggregatedFieldOutcome.MATCH:
            return "The retrieved candidate's case name could not be confirmed."
        return "Case name and court do not disagree; verify the differing year."
    if case_name is AggregatedFieldOutcome.MISMATCH and court is AggregatedFieldOutcome.MISMATCH:
        return "The retrieved candidate has a different case name and court."
    if case_name is AggregatedFieldOutcome.MISMATCH:
        return "The retrieved candidate has a different case name."
    return "The retrieved candidate has a different court."


def _field_outcome(outcome: FieldCheckOutcome | None) -> AggregatedFieldOutcome | None:
    return AggregatedFieldOutcome(outcome.value) if outcome is not None else None


def _required_child(
    validation: CitationValidation,
    node_type: type[ExactCaseNameCheckNode] | type[YearCheckNode],
    candidate_node_id: str,
) -> ExactCaseNameCheckNode | YearCheckNode:
    matches = [
        node
        for node in validation.nodes
        if isinstance(node, node_type) and candidate_node_id in node.depends_on
    ]
    if len(matches) != 1:
        msg = f"Locator candidate assessment requires one {node_type.__name__} below {candidate_node_id!r}"
        raise ValueError(msg)
    return matches[0]


def _required_court_check(validation: CitationValidation, candidate_node_id: str) -> CourtCheckNode:
    direct = [
        node
        for node in validation.nodes
        if isinstance(node, CourtCheckNode) and candidate_node_id in node.depends_on
    ]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        msg = f"Locator candidate assessment found multiple direct court checks below {candidate_node_id!r}"
        raise ValueError(msg)

    docket = _required_child(validation, DocketCourtRetrievalNode, candidate_node_id)
    indirect = [
        node
        for node in validation.nodes
        if isinstance(node, CourtCheckNode) and docket.node_id in node.depends_on
    ]
    if len(indirect) != 1:
        msg = f"Locator candidate assessment requires one court check below {candidate_node_id!r}"
        raise ValueError(msg)
    return indirect[0]


def _candidate_provenance(
    validation: CitationValidation,
    assessment: LocatorCandidateAssessmentNode,
) -> CandidateProvenance:
    """Keep a candidate summary honest about the retrieval record behind it."""
    candidate = next(
        (
            node
            for node in validation.nodes
            if isinstance(node, CandidateEvaluationNode)
            and _is_ancestor(validation, node.node_id, assessment.node_id)
        ),
        None,
    )
    if candidate is None:
        msg = f"Locator candidate assessment {assessment.node_id!r} has no candidate evaluation dependency"
        raise ValueError(msg)
    provenance = {
        CandidateEvaluationSource.LOCATOR_LOOKUP: CandidateProvenance.OPINION,
        CandidateEvaluationSource.DOCKET_SEARCH: CandidateProvenance.DOCKET,
        CandidateEvaluationSource.GOVINFO_DOCKET_SEARCH: CandidateProvenance.GOVINFO,
        CandidateEvaluationSource.COURTLISTENER_CLUSTER_BODY_SEARCH: CandidateProvenance.OPINION,
        CandidateEvaluationSource.COURTLISTENER_DOCKET_BODY_SEARCH: CandidateProvenance.DOCKET,
        CandidateEvaluationSource.GOVINFO_BODY_SEARCH: CandidateProvenance.GOVINFO,
        CandidateEvaluationSource.FULL_REPORTER_METADATA_SEARCH: CandidateProvenance.DOCKET,
        CandidateEvaluationSource.GOVINFO_FULL_REPORTER_METADATA_SEARCH: CandidateProvenance.GOVINFO,
    }.get(candidate.source)
    if provenance is None:
        msg = f"Locator candidate assessment cannot summarize source {candidate.source.value!r}"
        raise ValueError(msg)
    return provenance


def _is_ancestor(validation: CitationValidation, ancestor_id: str, node_id: str) -> bool:
    """Return whether an assessment transitively depends on an evaluation node."""
    by_id = {node.node_id: node for node in validation.nodes}

    def visit(current_id: str, visited: set[str]) -> bool:
        if current_id == ancestor_id:
            return True
        if current_id in visited:
            return False
        visited.add(current_id)
        current = by_id.get(current_id)
        return current is not None and any(visit(dependency, visited) for dependency in current.depends_on)

    return visit(node_id, set())
