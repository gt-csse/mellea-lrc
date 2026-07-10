"""Project existing post-extraction stages into citation-node steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.assessment.types import (
    AmbiguousCitationAssessment,
    AssessedCitationAssessment,
    CitationAssessment,
    FailedCitationAssessment,
    SkippedCitationAssessment,
    WaitingCitationAssessment,
)
from mellea_lrc.citation_nodes.types import (
    CitationNode,
    CitationNodeDocument,
    CitationNodeStatus,
    CitationStep,
    CitationStepStatus,
)
from mellea_lrc.retrieval.types import (
    CaseNameSearchCandidate,
    CaseNameSearchCorpus,
    CaseNameSearchProbe,
    CaseNameSearchStatus,
    CitationRetrieval,
    CourtListenerRequestTrace,
    LookupFailedCitationRetrieval,
    NotFoundCitationRetrieval,
    RetrievedDocument,
    RetrievalStatus,
    ThrottledCitationRetrieval,
)
from mellea_lrc.serialization import serialize_citation_assessment, serialize_citation_retrieval
from mellea_lrc.serialization.json import serialize_jurisdiction

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.jurisdiction_inference.types import InferredDocument, Jurisdiction


def with_jurisdiction_steps(
    document: CitationNodeDocument,
    inferred: InferredDocument,
) -> CitationNodeDocument:
    """Append jurisdiction inference steps from an inferred document."""
    jurisdictions = dict(
        zip(
            (item.citation_id for item in inferred.citations),
            inferred.jurisdictions,
            strict=True,
        )
    )
    return _replace_nodes(
        document,
        lambda node: _append_jurisdiction_step(node, jurisdictions.get(node.citation_id)),
    )


def with_retrieval_steps(
    document: CitationNodeDocument,
    retrieval: RetrievedDocument,
) -> CitationNodeDocument:
    """Append retrieval steps from a retrieved document."""
    retrievals = {item.citation_id: item for item in retrieval.retrievals}
    return _replace_nodes(
        document,
        lambda node: _append_retrieval_steps(node, retrievals.get(node.citation_id)),
    )


def with_assessment_steps(
    document: CitationNodeDocument,
    assessment_items: tuple[CitationAssessment, ...],
) -> CitationNodeDocument:
    """Append assessment steps from document assessment records."""
    assessments = {item.citation_id: item for item in assessment_items}
    return _replace_nodes(
        document,
        lambda node: _append_assessment_step(node, assessments.get(node.citation_id)),
    )


def _replace_nodes(
    document: CitationNodeDocument,
    transform: Callable[[CitationNode], CitationNode],
) -> CitationNodeDocument:
    nodes = tuple(transform(node) for node in document.nodes)
    return CitationNodeDocument(text=document.text, nodes=nodes)


def _append_jurisdiction_step(
    node: CitationNode,
    jurisdiction: Jurisdiction | None,
) -> CitationNode:
    step_id = _step_id(node.citation_id, "jurisdiction")
    if jurisdiction is None:
        return node.append_step(
            CitationStep(
                operation="jurisdiction.inference",
                status=CitationStepStatus.SKIPPED,
                summary="No jurisdiction inference was available for this citation.",
                step_id=step_id,
            )
        )
    reporter_status = jurisdiction.reporter_inference.status.value
    court_status = jurisdiction.court_inference.status.value
    return node.append_step(
        CitationStep(
            operation="jurisdiction.inference",
            status=CitationStepStatus.SUCCEEDED,
            summary=f"Reporter {reporter_status}; court {court_status}.",
            step_id=step_id,
            data={"jurisdiction": serialize_jurisdiction(jurisdiction)},
        )
    )


def _append_retrieval_steps(
    node: CitationNode,
    retrieval: CitationRetrieval | None,
) -> CitationNode:
    if retrieval is None:
        return node.append_step(
            CitationStep(
                operation="retrieval.exact_lookup",
                status=CitationStepStatus.SKIPPED,
                summary="No retrieval result was available for this citation.",
                step_id=_step_id(node.citation_id, "retrieval", "exact_lookup"),
            )
        )
    status = _retrieval_step_status(retrieval)
    exact_step_id = _step_id(node.citation_id, "retrieval", "exact_lookup")
    updated = node.append_step(
        CitationStep(
            operation="retrieval.exact_lookup",
            status=status,
            summary=f"Exact locator lookup returned {retrieval.status.value}.",
            step_id=exact_step_id,
            data={"retrieval": serialize_citation_retrieval(retrieval)},
            error=_retrieval_error(retrieval),
        )
    )
    if not isinstance(retrieval, NotFoundCitationRetrieval):
        return updated
    search = retrieval.candidate_search
    if search.status is CaseNameSearchStatus.NOT_ATTEMPTED:
        return updated
    return _append_case_name_search_steps(updated, retrieval, depends_on=exact_step_id)


def _append_case_name_search_steps(
    node: CitationNode,
    retrieval: NotFoundCitationRetrieval,
    *,
    depends_on: str,
) -> CitationNode:
    search = retrieval.candidate_search
    fallback_step_id = _step_id(node.citation_id, "retrieval", "fallback_decision")
    preparation_step_id = _step_id(node.citation_id, "retrieval", "case_name_preparation")
    query_step_id = _step_id(node.citation_id, "retrieval", "candidate_query")

    updated = node.append_step(
        CitationStep(
            operation="retrieval.fallback_decision",
            status=_candidate_search_step_status(search.status),
            summary=_fallback_summary(search.status),
            step_id=fallback_step_id,
            depends_on=(depends_on,),
            data={
                "exact_lookup_status": retrieval.status.value,
                "candidate_search_status": search.status.value,
                "locator": retrieval.locator,
            },
        )
    )

    evidence = _case_name_evidence(node, search.preparation)
    updated = updated.append_step(
        CitationStep(
            operation="retrieval.case_name_preparation",
            status=CitationStepStatus.SUCCEEDED
            if evidence["prepared_case_name"]
            else CitationStepStatus.SKIPPED,
            summary=_case_name_preparation_summary(evidence),
            step_id=preparation_step_id,
            depends_on=(fallback_step_id,),
            data=evidence,
        )
    )

    updated = updated.append_step(
        CitationStep(
            operation="retrieval.candidate_query",
            status=CitationStepStatus.SUCCEEDED if search.query else CitationStepStatus.SKIPPED,
            summary=f"Prepared query: {search.query}" if search.query else "No candidate query was prepared.",
            step_id=query_step_id,
            depends_on=(preparation_step_id,),
            data={
                "query": search.query,
                "candidate_search_status": search.status.value,
                "query_source": "deterministic_party_anchors" if search.query else None,
            },
        )
    )

    probe_step_ids: list[str] = []
    for probe in search.probes:
        probe_step_id = _step_id(
            node.citation_id,
            "retrieval",
            "corpus_probe",
            probe.corpus.value,
        )
        probe_step_ids.append(probe_step_id)
        updated = updated.append_step(
            CitationStep(
                operation="retrieval.corpus_probe",
                status=_candidate_search_step_status(probe.status),
                summary=_probe_summary(probe),
                step_id=probe_step_id,
                depends_on=(query_step_id,),
                lane=probe.corpus.value,
                data=_probe_payload(probe),
                error=probe.request_trace.error_message,
            )
        )

    result_dependencies = tuple(probe_step_ids) if probe_step_ids else (query_step_id,)
    return updated.append_step(
        CitationStep(
            operation="retrieval.candidate_results",
            status=_candidate_search_step_status(search.status),
            summary=_candidate_results_summary(search.probes),
            step_id=_step_id(node.citation_id, "retrieval", "candidate_results"),
            depends_on=result_dependencies,
            data={
                "candidate_search_status": search.status.value,
                "total_candidate_summaries": sum(len(probe.candidates) for probe in search.probes),
                "result_counts_by_corpus": {
                    probe.corpus.value: probe.case_count for probe in search.probes
                },
            },
        )
    )


def _append_assessment_step(
    node: CitationNode,
    assessment: CitationAssessment | None,
) -> CitationNode:
    depends_on = _terminal_dependency_ids(node)
    if assessment is None:
        return node.append_step(
            CitationStep(
                operation="assessment.field_check",
                status=CitationStepStatus.SKIPPED,
                summary="No assessment result was available for this citation.",
                step_id=_step_id(node.citation_id, "assessment", "field_check"),
                depends_on=depends_on,
            )
        )
    return node.append_step(
        CitationStep(
            operation="assessment.field_check",
            status=_assessment_step_status(assessment),
            summary=f"Assessment status is {assessment.status.value}.",
            step_id=_step_id(node.citation_id, "assessment", "field_check"),
            depends_on=depends_on,
            data={"assessment": serialize_citation_assessment(assessment)},
            error=assessment.error if isinstance(assessment, FailedCitationAssessment) else None,
        ),
        status=_node_status_after_assessment(node, assessment),
    )


def _terminal_dependency_ids(node: CitationNode) -> tuple[str, ...]:
    """Return current terminal retrieval dependencies for the next consumer step.

    Jurisdiction inference is intentionally excluded because it is an always-run
    side branch at this phase; assessment does not consume its result.
    """
    by_id = {step.step_id: step for step in node.steps if step.step_id}
    consumed = {
        dependency
        for step in node.steps
        for dependency in step.depends_on
        if dependency in by_id
    }
    retrieval_terminals = tuple(
        step_id
        for step_id, step in by_id.items()
        if step_id not in consumed and step.operation.startswith("retrieval.")
    )
    if retrieval_terminals:
        return retrieval_terminals
    retrieval_steps = tuple(step.step_id for step in node.steps if step.step_id and step.operation.startswith("retrieval."))
    if retrieval_steps:
        return (retrieval_steps[-1],)
    return ()


def _retrieval_step_status(retrieval: CitationRetrieval) -> CitationStepStatus:
    if isinstance(retrieval, (LookupFailedCitationRetrieval, ThrottledCitationRetrieval)):
        return CitationStepStatus.FAILED
    if retrieval.status in {RetrievalStatus.SKIPPED, RetrievalStatus.INVALID}:
        return CitationStepStatus.SKIPPED
    return CitationStepStatus.SUCCEEDED


def _candidate_search_step_status(status: CaseNameSearchStatus) -> CitationStepStatus:
    if status in {CaseNameSearchStatus.SEARCH_FAILED, CaseNameSearchStatus.SEARCH_UNAVAILABLE}:
        return CitationStepStatus.FAILED
    if status in {
        CaseNameSearchStatus.SKIPPED_NO_CASE_NAME,
        CaseNameSearchStatus.SKIPPED_PARTIAL_CASE_NAME,
    }:
        return CitationStepStatus.SKIPPED
    return CitationStepStatus.SUCCEEDED


def _assessment_step_status(assessment: CitationAssessment) -> CitationStepStatus:
    if isinstance(assessment, FailedCitationAssessment):
        return CitationStepStatus.FAILED
    if isinstance(assessment, WaitingCitationAssessment):
        return CitationStepStatus.BLOCKED
    if isinstance(assessment, SkippedCitationAssessment):
        return CitationStepStatus.SKIPPED
    return CitationStepStatus.SUCCEEDED


def _node_status_after_assessment(
    node: CitationNode,
    assessment: CitationAssessment,
) -> CitationNodeStatus:
    if isinstance(assessment, FailedCitationAssessment):
        return CitationNodeStatus.FAILED
    if isinstance(assessment, WaitingCitationAssessment):
        return CitationNodeStatus.BLOCKED
    if isinstance(assessment, (AssessedCitationAssessment, AmbiguousCitationAssessment, SkippedCitationAssessment)):
        return CitationNodeStatus.COMPLETE
    return node.status


def _retrieval_error(retrieval: CitationRetrieval) -> str | None:
    if isinstance(retrieval, LookupFailedCitationRetrieval):
        return retrieval.request_trace.error_message
    if isinstance(retrieval, ThrottledCitationRetrieval):
        return retrieval.request_trace.error_message or "CourtListener request was throttled."
    return None


def _step_id(citation_id: str, *parts: str) -> str:
    return ":".join((citation_id, *parts))


def _fallback_summary(status: CaseNameSearchStatus) -> str:
    if status in {CaseNameSearchStatus.SEARCHED, CaseNameSearchStatus.PARTIAL}:
        return "Exact lookup missed; case-name candidate search fallback ran."
    if status is CaseNameSearchStatus.SEARCH_FAILED:
        return "Exact lookup missed; case-name candidate search failed."
    if status is CaseNameSearchStatus.SEARCH_UNAVAILABLE:
        return "Exact lookup missed; case-name candidate search was unavailable."
    return f"Exact lookup missed; case-name candidate search was skipped: {status.value}."


def _case_name_evidence(node: CitationNode, preparation: object) -> dict[str, object]:
    citation = node.input.citation
    plaintiff = getattr(preparation, "plaintiff", None) or getattr(citation, "plaintiff", None)
    defendant = getattr(preparation, "defendant", None) or getattr(citation, "defendant", None)
    original_case_name = getattr(preparation, "original_case_name", None)
    if original_case_name is None and isinstance(plaintiff, str) and isinstance(defendant, str):
        original_case_name = f"{plaintiff} v. {defendant}"
    prepared_case_name = getattr(preparation, "prepared_case_name", None)
    if prepared_case_name is None and isinstance(plaintiff, str) and isinstance(defendant, str):
        prepared_case_name = f"{plaintiff} v. {defendant}"
    return {
        "original_case_name": original_case_name,
        "plaintiff": plaintiff,
        "defendant": defendant,
        "court": getattr(preparation, "court", None) or getattr(citation, "court", None),
        "locator": getattr(preparation, "locator", None) or node.input.matched_locator_text,
        "llm_status": getattr(preparation, "status", None).value
        if getattr(preparation, "status", None) is not None
        else "not_attempted",
        "llm_classification": getattr(preparation, "llm_classification", None),
        "llm_reason": getattr(preparation, "llm_reason", None),
        "preparation_source": getattr(preparation, "source", None),
        "prepared_case_name": prepared_case_name,
        "error_message": getattr(preparation, "error_message", None),
    }


def _case_name_preparation_summary(evidence: dict[str, object]) -> str:
    if evidence["prepared_case_name"]:
        return f"Prepared case-name search parties as {evidence['prepared_case_name']}."
    if evidence["plaintiff"] or evidence["defendant"]:
        return "Case-name preparation found incomplete party evidence."
    return "Case-name preparation found no usable parties for this locator."


def _probe_summary(probe: CaseNameSearchProbe) -> str:
    label = _corpus_label(probe.corpus)
    if probe.status is CaseNameSearchStatus.SEARCHED:
        count = probe.case_count if probe.case_count is not None else 0
        return f"{label} search returned {count} candidate(s)."
    if probe.status is CaseNameSearchStatus.SEARCH_FAILED:
        return f"{label} search failed."
    if probe.status is CaseNameSearchStatus.SEARCH_UNAVAILABLE:
        return f"{label} search was unavailable."
    return f"{label} search returned {probe.status.value}."


def _candidate_results_summary(probes: tuple[CaseNameSearchProbe, ...]) -> str:
    if not probes:
        return "No corpus probes were run, so no candidate summaries were surfaced."
    total = sum(len(probe.candidates) for probe in probes)
    searched = sum(probe.status is CaseNameSearchStatus.SEARCHED for probe in probes)
    return (
        f"Candidate search surfaced {total} bounded candidate summaries "
        f"across {searched} searched corpus probe(s)."
    )


def _probe_payload(probe: CaseNameSearchProbe) -> dict[str, object]:
    return {
        "corpus": probe.corpus.value,
        "corpus_label": _corpus_label(probe.corpus),
        "status": probe.status.value,
        "case_count": probe.case_count,
        "request_trace": _request_trace_payload(probe.request_trace),
        "candidates": [_candidate_payload(candidate) for candidate in probe.candidates],
    }


def _request_trace_payload(trace: CourtListenerRequestTrace) -> dict[str, object]:
    return {
        "http_status": trace.http_status,
        "cache": trace.cache,
        "key": trace.key,
        "error_message": trace.error_message,
    }


def _candidate_payload(candidate: CaseNameSearchCandidate) -> dict[str, object]:
    return {
        "case_name": candidate.case_name,
        "court_id": candidate.court_id,
        "date_filed": candidate.date_filed,
        "docket_number": candidate.docket_number,
        "cluster_id": candidate.cluster_id,
        "docket_id": candidate.docket_id,
        "absolute_url": candidate.absolute_url,
    }


def _corpus_label(corpus: CaseNameSearchCorpus) -> str:
    if corpus is CaseNameSearchCorpus.OPINIONS:
        return "Opinion"
    if corpus is CaseNameSearchCorpus.RECAP:
        return "RECAP"
    return corpus.value
