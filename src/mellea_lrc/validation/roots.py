"""Document-native, checkpointable full-reporter-locator identity stages.

The typed runner is local execution state.  Each public function below projects
that state into a citation's stage-neutral trace and returns the same
:class:`~mellea_lrc.extraction.types.Document`.  Exact lookup, unique identity,
and bounded ambiguity are deliberately separate stages: a saved lookup artifact
can be inspected or resumed without contacting CourtListener again.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.record import UNJUDGED, Node, Question, Reads, Resolution
from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerOpinionCluster
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.serialization.validated_document import deserialize_validation_node
from mellea_lrc.validation.execution import CitationValidationRunner
from mellea_lrc.validation.root_context import masked_root_context
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    CitationValidation,
    ExactLocatorLookupNode,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    LocatorLookupOutcome,
    ValidationNode,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import Document


EXACT_LOOKUP_STAGE = "full_reporter_locator_exact_lookup"
UNIQUE_IDENTITY_STAGE = "full_reporter_locator_unique_identity"
AMBIGUITY_RESOLUTION_STAGE = "full_reporter_locator_ambiguity_resolution"
_MADE_BY = "mellea_lrc.validation.roots"


async def lookup_full_reporter_locators_exact(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
) -> Document:
    """Fetch exact reporter-locator candidates and persist only that evidence.

    ``found`` candidate sets wait for :func:`validate_unique_full_reporter_locator_identities`.
    Multiple candidates are explicitly recorded as ``deferred_to_ambiguity``
    under :class:`Question.LOCATOR_LOOKUP`; this stage never makes an identity
    judgement.  Lookup misses remain ``deferred_to_search`` for a later search
    route, and dockets stay untouched for their own lookup design.
    """
    _require_stage(document, ROOT_FORMATION_STAGE, "Exact full-reporter lookup")
    if EXACT_LOOKUP_STAGE in document.passes:
        return document
    _reject_partial_stage(document, EXACT_LOOKUP_STAGE)

    service = client if client is not None else CourtListenerClient()
    runner = CitationValidationRunner(client=service)
    for record in _reporter_roots(document):
        progression = runner.run_exact_full_reporter_locator_lookup(CitationValidation(citation=record))
        lookup = progression.nodes[-1]
        if not isinstance(lookup, ExactLocatorLookupNode):
            msg = f"Exact lookup for {record.citation_id!r} did not produce ExactLocatorLookupNode"
            raise ValueError(msg)
        trace_node = _trace_node(lookup, stage=EXACT_LOOKUP_STAGE)
        record.observe(trace_node)
        record.judge(
            trace_node,
            Question.LOCATOR_LOOKUP,
            _lookup_stage_outcome(lookup.outcome),
            message=lookup.outcome_message,
        )
    return replace(document, passes=(*document.passes, EXACT_LOOKUP_STAGE))


async def validate_unique_full_reporter_locator_identities(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Validate only roots whose saved exact lookup returned one candidate."""
    _require_stage(document, EXACT_LOOKUP_STAGE, "Unique full-reporter identity")
    if UNIQUE_IDENTITY_STAGE in document.passes:
        return document
    _reject_partial_stage(document, UNIQUE_IDENTITY_STAGE)

    service = client if client is not None else CourtListenerClient()
    runner = CitationValidationRunner(client=service)
    for record in _reporter_roots(document):
        lookup = _saved_exact_lookup(record)
        if lookup.outcome is not LocatorLookupOutcome.FOUND:
            continue
        if record.judgement(Question.IDENTITY).outcome != UNJUDGED:
            continue
        completed = await runner.run_locator_found_identity(
            CitationValidation(citation=record, nodes=(lookup,)),
            lookup=lookup,
            document_text=masked_root_context(document, record).as_document_text(
                document_length=len(document.text)
            ),
            session=session,
        )
        _write_identity_progression(record, completed, stage=UNIQUE_IDENTITY_STAGE)
    return replace(document, passes=(*document.passes, UNIQUE_IDENTITY_STAGE))


async def resolve_full_reporter_locator_ambiguities(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Resolve only saved bounded exact-lookup candidate sets.

    Every returned candidate is kept in the trace.  Candidate sets of twenty or
    more are retained and end as ``deferred_to_future_implementation``; they
    never get silently truncated.  A later opinion-reading route can add more
    fine-grained selection without coupling it to this stage.
    """
    _require_stage(document, UNIQUE_IDENTITY_STAGE, "Full-reporter ambiguity resolution")
    if AMBIGUITY_RESOLUTION_STAGE in document.passes:
        return document
    _reject_partial_stage(document, AMBIGUITY_RESOLUTION_STAGE)

    service = client if client is not None else CourtListenerClient()
    runner = CitationValidationRunner(client=service)
    for record in _reporter_roots(document):
        lookup = _saved_exact_lookup(record)
        if lookup.outcome is not LocatorLookupOutcome.AMBIGUOUS:
            continue
        if record.judgement(Question.IDENTITY).outcome != UNJUDGED:
            continue
        completed = await runner.run_locator_ambiguous(
            CitationValidation(citation=record, nodes=(lookup,)),
            lookup=lookup,
            document_text=masked_root_context(document, record).as_document_text(
                document_length=len(document.text)
            ),
            session=session,
        )
        _write_identity_progression(record, completed, stage=AMBIGUITY_RESOLUTION_STAGE)
    return replace(document, passes=(*document.passes, AMBIGUITY_RESOLUTION_STAGE))


def _reporter_roots(document: Document) -> tuple[CitationRecord, ...]:
    return tuple(
        record
        for record in document.active_citations
        if record.is_root and isinstance(record.stated, FullCaseCitation)
    )


def _require_stage(document: Document, required_stage: str, stage_name: str) -> None:
    if required_stage not in document.passes:
        msg = f"{stage_name} requires {required_stage} before validation."
        raise ValueError(msg)


def _reject_partial_stage(document: Document, stage: str) -> None:
    if any(node.stage == stage for record in document.citations for node in record.trace):
        msg = f"Cannot resume partial {stage}; restart from the document before this stage."
        raise ValueError(msg)


def _saved_exact_lookup(record: CitationRecord) -> ExactLocatorLookupNode:
    matches = [
        node
        for node in record.trace
        if node.stage == EXACT_LOOKUP_STAGE
        and node.details.get("validation_node_type") == ExactLocatorLookupNode.__name__
    ]
    if len(matches) != 1:
        msg = f"Expected exactly one saved exact lookup for {record.citation_id!r}, found {len(matches)}"
        raise ValueError(msg)
    raw = matches[0].details.get("validation")
    if not isinstance(raw, dict):
        msg = f"Saved exact lookup for {record.citation_id!r} has no validation payload"
        raise ValueError(msg)
    node = deserialize_validation_node({"node_type": ExactLocatorLookupNode.__name__, **raw})
    if not isinstance(node, ExactLocatorLookupNode):
        msg = f"Saved exact lookup for {record.citation_id!r} decoded as {type(node).__name__}"
        raise ValueError(msg)
    return node


def _lookup_stage_outcome(outcome: LocatorLookupOutcome) -> str:
    if outcome is LocatorLookupOutcome.FOUND:
        return "found"
    if outcome is LocatorLookupOutcome.AMBIGUOUS:
        return "deferred_to_ambiguity"
    if outcome is LocatorLookupOutcome.NOT_FOUND:
        return "deferred_to_search"
    return "deferred_to_future_implementation"


def _write_identity_progression(
    record: CitationRecord,
    progression: CitationValidation,
    *,
    stage: str,
) -> None:
    """Project completed typed evidence and its terminal identity state."""
    projected = {_node.node_id: _trace_node(_node, stage=stage) for _node in progression.nodes}
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


def _trace_node(node: ValidationNode, *, stage: str) -> Node:
    """Make one typed working node a stage-neutral, serializable record node."""
    payload = serialize_dataclass(node)
    return Node(
        node_id=node.node_id,
        reads=Reads.RECORD if isinstance(node, LocatorIdentityResolutionNode) else _reads(node),
        stage=stage,
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
