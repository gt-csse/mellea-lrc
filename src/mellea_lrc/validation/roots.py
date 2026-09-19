"""Document-native identity validation for complete reporter locators.

The runner's typed objects are local working state for one citation. This
module projects their complete, JSON-ready evidence onto the citation record
before handing the same :class:`Document` to the next stage. No validation
wrapper is part of the stage contract or its persisted artifact.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.record import UNJUDGED, Node, Question, Reads, Resolution
from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerOpinionCluster
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.validation.execution import CitationValidationRunner
from mellea_lrc.validation.root_context import masked_root_context
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    CitationValidation,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    ValidationNode,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import Document


ROOT_IDENTITY_STAGE = "root_identity"
_MADE_BY = "mellea_lrc.validation.roots"


async def full_reporter_locator_identity(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Validate reporter-root identity and return the same document shape.

    The current admitted route reads only ``FullCaseCitation`` reporter
    locators. Docket roots stay untouched for their independent docket lookup
    route, and exact-lookup misses are recorded as ``deferred_to_search`` on
    their own citation record. The function is async because bounded ambiguous
    results may require a grounded model choice; ordinary exact lookup itself
    has no model call.
    """
    if ROOT_FORMATION_STAGE not in document.passes:
        msg = "Root identity requires form_roots(document) before validation."
        raise ValueError(msg)
    service = client if client is not None else CourtListenerClient()
    runner = CitationValidationRunner(client=service)
    for record in document.active_citations:
        if not isinstance(record.stated, FullCaseCitation):
            continue
        if not record.is_root:
            continue
        if record.judgement(Question.IDENTITY).outcome != UNJUDGED:
            continue
        if any(node.stage == ROOT_IDENTITY_STAGE for node in record.trace):
            msg = (
                f"Cannot resume partial root-identity work for {record.citation_id!r}; "
                "restart from the document before this stage."
            )
            raise ValueError(msg)

        progression = CitationValidation(citation=record)
        completed = await runner.run_full_reporter_locator_identity(
            progression,
            document_text=masked_root_context(document, record).as_document_text(
                document_length=len(document.text)
            ),
            session=session,
        )
        _write_identity_progression(record, completed)

    if ROOT_IDENTITY_STAGE in document.passes:
        return document
    return replace(document, passes=(*document.passes, ROOT_IDENTITY_STAGE))


def _write_identity_progression(record: CitationRecord, progression: CitationValidation) -> None:
    """Write transient typed execution evidence and its terminal state to a record."""
    projected = {_node.node_id: _trace_node(_node) for _node in progression.nodes}
    for node in projected.values():
        record.observe(node)

    resolution = progression.identity_resolution
    if resolution is None:
        msg = f"Root identity for {record.citation_id!r} ended without an identity decision"
        raise ValueError(msg)
    resolution_node = projected[resolution.node_id]

    if resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED:
        cluster = _selected_cluster(progression, resolution)
        record.resolve(
            resolution_node,
            Resolution(
                cluster_id=cluster.cluster_id,
                case_name=cluster.case_name,
                date_filed=cluster.date_filed,
                court_id=cluster.court_id,
                node_id=resolution_node.node_id,
                opinion_ids=cluster.sub_opinion_ids,
                citations=tuple(
                    f"{citation.volume} {citation.reporter} {citation.page}" for citation in cluster.citations
                ),
                docket_id=cluster.docket_id,
            ),
        )
        if cluster.cluster_id is not None:
            record.reattribute(resolution_node, cluster.cluster_id)

    record.judge(
        resolution_node,
        Question.IDENTITY,
        resolution.outcome.value,
        message=resolution.outcome_message,
    )


def _selected_cluster(
    progression: CitationValidation,
    resolution: LocatorIdentityResolutionNode,
) -> CourtListenerOpinionCluster:
    """Return the archive record selected by a resolved reporter identity."""
    selected_index = resolution.selected_candidate_index
    candidate = next(
        (
            node
            for node in progression.nodes
            if isinstance(node, CandidateEvaluationNode)
            and node.source is CandidateEvaluationSource.LOCATOR_LOOKUP
            and node.candidate_index == selected_index
        ),
        None,
    )
    if candidate is None or not isinstance(candidate.record, CourtListenerOpinionCluster):
        msg = "A resolved reporter identity must select one stored CourtListener opinion cluster"
        raise ValueError(msg)
    return candidate.record


def _trace_node(node: ValidationNode) -> Node:
    """Make one typed working node a stage-neutral, serializable record node."""
    payload = serialize_dataclass(node)
    return Node(
        node_id=node.node_id,
        reads=Reads.RECORD if isinstance(node, LocatorIdentityResolutionNode) else _reads(node),
        stage=ROOT_IDENTITY_STAGE,
        made_by=_MADE_BY,
        outcome=str(payload["outcome"]),
        message=_message(payload),
        depends_on=node.depends_on,
        details={
            "validation_node_type": type(node).__name__,
            "validation": payload,
        },
    )


def _reads(node: ValidationNode) -> Reads:
    """Record whether a validation operation called a model over filing text."""
    return Reads.DOCUMENT if getattr(node, "run", None) is not None else Reads.RECORD


def _message(payload: dict[str, object]) -> str | None:
    """Prefer the operation's outcome explanation in the normalized node surface."""
    for key in ("outcome_message", "status_message", "error"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None
