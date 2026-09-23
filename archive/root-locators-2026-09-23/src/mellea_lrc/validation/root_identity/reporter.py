"""Document-native, checkpointable full-reporter-locator identity stages.

The typed runner is local execution state.  Each public function below projects
that state into a citation's stage-neutral trace and returns the same
:class:`~mellea_lrc.model.document.Document`.  Exact lookup, unique identity,
and bounded ambiguity are deliberately separate stages: a saved lookup artifact
can be inspected or resumed without contacting CourtListener again.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerOpinionCluster
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE
from mellea_lrc.model.citations import CitationField, FullCaseCitation
from mellea_lrc.model.operations import (
    attribute_authority,
    judge_citation,
    mark_extraction_reviewed,
    observe_citation,
    resolve_citation,
    update_fields,
)
from mellea_lrc.model.record import UNJUDGED, Node, Question, Reads, Resolution
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.serialization.validated_document import deserialize_validation_node
from mellea_lrc.validation.aggregation import (
    run_locator_candidate_assessment,
    run_locator_citation_summary,
)
from mellea_lrc.validation.candidates.evaluation import (
    run_full_reporter_metadata_candidate_evaluation,
    run_govinfo_full_reporter_metadata_candidate_evaluation,
)
from mellea_lrc.validation.candidates.selection import CANDIDATE_SELECTION_LIMIT
from mellea_lrc.validation.candidates.state import CandidateValidationState
from mellea_lrc.validation.execution import CitationValidationRunner
from mellea_lrc.validation.field_checks import (
    run_court_check,
    run_exact_case_name_check,
    run_year_check,
)
from mellea_lrc.validation.root_identity.context import masked_root_context
from mellea_lrc.validation.root_identity.retrospective import (
    consistent_cutoff,
    eligible_cluster,
    eligible_metadata_candidate,
    require_replay_cutoff,
)
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    CitationValidation,
    ExactLocatorLookupNode,
    FullReporterSearchNode,
    FullReporterSearchOutcome,
    GovInfoFullReporterSearchNode,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameReextractionNode,
    MelleaCaseNameReextractionOutcome,
    MelleaCaseNameReviewNode,
    ValidationNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.model.document import Document
    from mellea_lrc.model.record import CitationRecord


EXACT_LOOKUP_STAGE = "full_reporter_locator_exact_lookup"
UNIQUE_IDENTITY_STAGE = "full_reporter_locator_unique_identity"
AMBIGUITY_RESOLUTION_STAGE = "full_reporter_locator_ambiguity_resolution"
COURTLISTENER_FULL_REPORTER_SEARCH_STAGE = "courtlistener_full_reporter_metadata_search"
GOVINFO_FULL_REPORTER_SEARCH_STAGE = "govinfo_full_reporter_metadata_search"
FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE = "full_reporter_search_candidate_resolution"
_MADE_BY = "mellea_lrc.validation.root_identity.reporter"


async def lookup_full_reporter_locators_exact(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    retrospective_date: date | None = None,
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
        for record in _reporter_roots(document):
            require_replay_cutoff(
                _saved_exact_lookup(record).retrospective_date,
                retrospective_date,
                stage=EXACT_LOOKUP_STAGE,
            )
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
        lookup = _asof_exact_lookup(lookup, retrospective_date)
        trace_node = _trace_node(lookup, stage=EXACT_LOOKUP_STAGE)
        observe_citation(record, trace_node)
        judge_citation(
            record,
            trace_node,
            Question.LOCATOR_LOOKUP,
            _lookup_stage_outcome(lookup.outcome),
            message=lookup.outcome_message,
        )
    return document.evolve(passes=(*document.passes, EXACT_LOOKUP_STAGE))


async def validate_unique_full_reporter_locator_identities(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
    retrospective_date: date | None = None,
) -> Document:
    """Validate only roots whose saved exact lookup returned one candidate."""
    _require_stage(document, EXACT_LOOKUP_STAGE, "Unique full-reporter identity")
    _reject_changed_cutoff_after_identity(document, retrospective_date, UNIQUE_IDENTITY_STAGE)
    if UNIQUE_IDENTITY_STAGE in document.passes:
        for record in _reporter_roots(document):
            require_replay_cutoff(
                _saved_exact_lookup(record).retrospective_date,
                retrospective_date,
                stage=UNIQUE_IDENTITY_STAGE,
            )
        return document
    _reject_partial_stage(document, UNIQUE_IDENTITY_STAGE)

    service = client if client is not None else CourtListenerClient()
    runner = CitationValidationRunner(client=service)
    for record in _reporter_roots(document):
        lookup = _asof_exact_lookup(
            _saved_exact_lookup(record),
            consistent_cutoff(_saved_exact_lookup(record).retrospective_date, retrospective_date),
        )
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
        _write_identity_progression(
            record,
            completed,
            stage=UNIQUE_IDENTITY_STAGE,
            document_text=document.text,
        )
    return document.evolve(passes=(*document.passes, UNIQUE_IDENTITY_STAGE))


async def resolve_full_reporter_locator_ambiguities(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
    retrospective_date: date | None = None,
) -> Document:
    """Resolve only saved bounded exact-lookup candidate sets.

    Every returned candidate is kept in the trace.  Candidate sets of twenty or
    more are retained and end as ``deferred_to_future_implementation``; they
    never get silently truncated.  A later opinion-reading route can add more
    fine-grained selection without coupling it to this stage.
    """
    _require_stage(document, UNIQUE_IDENTITY_STAGE, "Full-reporter ambiguity resolution")
    _reject_changed_cutoff_after_identity(document, retrospective_date, AMBIGUITY_RESOLUTION_STAGE)
    if AMBIGUITY_RESOLUTION_STAGE in document.passes:
        for record in _reporter_roots(document):
            require_replay_cutoff(
                _saved_exact_lookup(record).retrospective_date,
                retrospective_date,
                stage=AMBIGUITY_RESOLUTION_STAGE,
            )
        return document
    _reject_partial_stage(document, AMBIGUITY_RESOLUTION_STAGE)

    service = client if client is not None else CourtListenerClient()
    runner = CitationValidationRunner(client=service)
    for record in _reporter_roots(document):
        lookup = _asof_exact_lookup(
            _saved_exact_lookup(record),
            consistent_cutoff(_saved_exact_lookup(record).retrospective_date, retrospective_date),
        )
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
        _write_identity_progression(
            record,
            completed,
            stage=AMBIGUITY_RESOLUTION_STAGE,
            document_text=document.text,
        )
    return document.evolve(passes=(*document.passes, AMBIGUITY_RESOLUTION_STAGE))


async def resolve_full_reporter_search_candidates(
    document: Document,
    *,
    session: MelleaSession | None = None,
    retrospective_date: date | None = None,
) -> Document:
    """Assess saved provider metadata and retain it for later locator search.

    CourtListener docket and GovInfo package metadata can identify a plausible
    case, but neither record connects the *stated reporter locator* to that
    case. Every candidate receives field checks for the saved trace, then the
    root proceeds to independent locator/body corroboration. A similar caption
    or a compatible docket date cannot establish the reported disposition.
    """
    _require_stage(document, AMBIGUITY_RESOLUTION_STAGE, "Full-reporter search candidate resolution")
    _require_stage(
        document,
        COURTLISTENER_FULL_REPORTER_SEARCH_STAGE,
        "Full-reporter search candidate resolution",
    )
    _require_stage(
        document,
        GOVINFO_FULL_REPORTER_SEARCH_STAGE,
        "Full-reporter search candidate resolution",
    )
    _reject_changed_cutoff_after_identity(
        document, retrospective_date, FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE
    )
    if FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE in document.passes:
        for record in _reporter_roots(document):
            if _saved_exact_lookup(record).outcome is LocatorLookupOutcome.NOT_FOUND:
                require_replay_cutoff(
                    _saved_courtlistener_full_reporter_search(record).retrospective_date,
                    retrospective_date,
                    stage=FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE,
                )
                require_replay_cutoff(
                    _saved_govinfo_full_reporter_search(record).retrospective_date,
                    retrospective_date,
                    stage=FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE,
                )
        return document
    _reject_partial_stage(document, FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE)

    for record in _reporter_roots(document):
        if record.judgement(Question.IDENTITY).outcome != UNJUDGED:
            continue
        if _saved_exact_lookup(record).outcome is not LocatorLookupOutcome.NOT_FOUND:
            continue
        courtlistener = _saved_courtlistener_full_reporter_search(record)
        govinfo = _saved_govinfo_full_reporter_search(record)
        cutoff = consistent_cutoff(courtlistener.retrospective_date, retrospective_date)
        cutoff = consistent_cutoff(govinfo.retrospective_date, cutoff)
        progression = await _resolve_metadata_candidates(
            record,
            _asof_metadata_search(courtlistener, source="courtlistener", retrospective_date=cutoff),
            _asof_metadata_search(govinfo, source="govinfo", retrospective_date=cutoff),
            document=document,
            session=session,
        )
        _write_metadata_identity_progression(
            record,
            progression,
            stage=FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE,
        )
    return document.evolve(passes=(*document.passes, FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE))


async def _resolve_metadata_candidates(
    record: CitationRecord,
    courtlistener: FullReporterSearchNode,
    govinfo: GovInfoFullReporterSearchNode,
    *,
    document: Document,
    session: MelleaSession | None,
) -> CitationValidation:
    """Build the deterministic candidate graph for one exact-lookup miss."""
    validation = CitationValidation(citation=record, nodes=(courtlistener, govinfo))
    search_nodes = (courtlistener, govinfo)
    if any(node.outcome is FullReporterSearchOutcome.EXCEEDS_REVIEW_LIMIT for node in search_nodes):
        return validation.append(
            _metadata_deferred_resolution(
                validation,
                outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION,
                depends_on=tuple(node.node_id for node in search_nodes),
                reason=(
                    "At least one metadata search returned twenty or more candidates; "
                    "the complete result set is retained for a later bounded review stage."
                ),
            )
        )

    sources = (
        (courtlistener, run_full_reporter_metadata_candidate_evaluation),
        (govinfo, run_govinfo_full_reporter_metadata_candidate_evaluation),
    )
    total = sum(len(node.candidates) for node, _ in sources)
    if total == 0:
        return validation.append(
            _metadata_deferred_resolution(
                validation,
                outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION,
                depends_on=tuple(node.node_id for node in search_nodes),
                reason=(
                    "Neither metadata provider returned a candidate that can identify this "
                    "exact reporter-locator miss; body corroboration remains a separate route."
                ),
            )
        )
    if total >= CANDIDATE_SELECTION_LIMIT:
        return validation.append(
            _metadata_deferred_resolution(
                validation,
                outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION,
                depends_on=tuple(node.node_id for node in search_nodes),
                reason=(
                    f"The two metadata providers retained {total} candidates, meeting the "
                    f"review limit of {CANDIDATE_SELECTION_LIMIT}."
                ),
            )
        )

    candidate_index = 1
    for search, make_candidate in sources:
        for result in search.candidates:
            candidate = make_candidate(
                validation,
                result=result,
                candidate_index=candidate_index,
                depends_on=(search.node_id,),
            )
            validation = validation.append(candidate)
            exact_case_name = run_exact_case_name_check(validation, candidate=candidate)
            year = run_year_check(validation, candidate=candidate)
            court = run_court_check(validation, evidence=candidate)
            validation = validation.append(exact_case_name).append(year).append(court)
            state = CandidateValidationState().with_case_name_result(
                outcome=AggregatedFieldOutcome(exact_case_name.outcome.value),
                evidence="exact",
                dependency_id=exact_case_name.node_id,
            )
            validation = validation.append(
                run_locator_candidate_assessment(validation, candidate=candidate, state=state)
            )
            candidate_index += 1

    summary = run_locator_citation_summary(validation)
    validation = validation.append(summary)
    return validation.append(
        _metadata_deferred_resolution(
            validation,
            outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH,
            depends_on=(summary.node_id,),
            reason=(
                "Caption metadata identifies possible case records but does not contain "
                "the stated reporter locator; continue with corpus and open-web retrieval."
            ),
        )
    )


def _metadata_deferred_resolution(
    validation: CitationValidation,
    *,
    outcome: LocatorIdentityResolutionOutcome,
    depends_on: tuple[str, ...],
    reason: str,
    matching_candidate_indices: tuple[int, ...] = (),
) -> LocatorIdentityResolutionNode:
    """Represent a non-selecting metadata result without declaring it wrong."""
    return LocatorIdentityResolutionNode(
        node_id=f"{validation.citation_id}:locator_identity_resolution",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        selected_candidate_index=None,
        selected_assessment_node_id=None,
        matching_candidate_indices=matching_candidate_indices,
        selection_evidence_node_id=None,
        depends_on=depends_on,
        status_message="Full-reporter metadata identity resolution deferred.",
        outcome_message=reason,
    )


def _asof_exact_lookup(
    lookup: ExactLocatorLookupNode,
    retrospective_date: date | None,
) -> ExactLocatorLookupNode:
    """Keep dated opinion clusters available at the filing's cutoff."""
    if retrospective_date is None:
        return lookup
    prior = consistent_cutoff(lookup.retrospective_date, retrospective_date)
    clusters = (
        (lookup.cluster,)
        if lookup.outcome is LocatorLookupOutcome.FOUND and lookup.cluster is not None
        else lookup.candidate_clusters
    )
    if lookup.outcome not in {LocatorLookupOutcome.FOUND, LocatorLookupOutcome.AMBIGUOUS}:
        return replace(lookup, retrospective_date=prior.isoformat())
    eligible = tuple(cluster for cluster in clusters if eligible_cluster(cluster, prior))
    raw_count = lookup.raw_candidate_count if lookup.raw_candidate_count is not None else len(clusters)
    if raw_count >= CANDIDATE_SELECTION_LIMIT:
        # The exact endpoint may return a bounded first page. Reducing that
        # page by date cannot prove there are fewer than 20 eligible records
        # elsewhere in the full result set. Preserve its over-limit guard;
        # ambiguity resolution will defer before examining any cluster.
        return replace(
            lookup,
            retrospective_date=prior.isoformat(),
            raw_candidate_count=raw_count,
            excluded_candidate_count=raw_count - len(eligible),
            outcome_message=(
                f"CourtListener returned at least {raw_count} opinion clusters; "
                f"{len(eligible)} on the returned page were dated on or before {prior}. "
                "The original over-limit deferral remains in force."
            ),
        )
    common = {
        "retrospective_date": prior.isoformat(),
        "raw_candidate_count": raw_count,
        "excluded_candidate_count": raw_count - len(eligible),
        "candidate_count": len(eligible),
        "outcome_message": (
            f"CourtListener returned {raw_count} opinion cluster(s); {len(eligible)} "
            f"were dated on or before {prior}. Undated clusters were excluded."
        ),
    }
    if len(eligible) == 0:
        return replace(
            lookup,
            **common,
            outcome=LocatorLookupOutcome.NOT_FOUND,
            cluster=None,
            candidate_clusters=(),
        )
    if len(eligible) == 1:
        return replace(
            lookup,
            **common,
            outcome=LocatorLookupOutcome.FOUND,
            cluster=eligible[0],
            candidate_clusters=(),
        )
    return replace(
        lookup,
        **common,
        outcome=LocatorLookupOutcome.AMBIGUOUS,
        cluster=None,
        candidate_clusters=eligible,
    )


def _reject_changed_cutoff_after_identity(
    document: Document,
    retrospective_date: date | None,
    stage: str,
) -> None:
    """Do not treat a prior undated reporter judgment as retrospective."""
    if retrospective_date is None:
        return
    for record in _reporter_roots(document):
        if record.judgement(Question.IDENTITY).outcome != UNJUDGED:
            require_replay_cutoff(
                _saved_exact_lookup(record).retrospective_date,
                retrospective_date,
                stage=stage,
            )


def _asof_metadata_search(
    node: FullReporterSearchNode | GovInfoFullReporterSearchNode,
    *,
    source: str,
    retrospective_date: date | None,
) -> FullReporterSearchNode | GovInfoFullReporterSearchNode:
    """Recheck saved metadata before field comparison or model review."""
    if retrospective_date is None:
        return node
    prior = consistent_cutoff(node.retrospective_date, retrospective_date)
    eligible = tuple(
        candidate
        for candidate in node.candidates
        if eligible_metadata_candidate(candidate, source=source, retrospective_date=prior)
    )
    raw_count = node.raw_candidate_count if node.raw_candidate_count is not None else len(node.candidates)
    if node.outcome in {FullReporterSearchOutcome.FOUND, FullReporterSearchOutcome.AMBIGUOUS}:
        outcome = (
            FullReporterSearchOutcome.NOT_FOUND
            if not eligible
            else FullReporterSearchOutcome.FOUND
            if len(eligible) == 1
            else FullReporterSearchOutcome.AMBIGUOUS
        )
    else:
        outcome = node.outcome
    return replace(
        node,
        outcome=outcome,
        candidate_count=len(eligible),
        candidates=eligible,
        retrospective_date=prior.isoformat(),
        raw_candidate_count=raw_count,
        excluded_candidate_count=raw_count - len(eligible),
        outcome_message=(
            f"{source} metadata search retained {raw_count} selected candidate(s); "
            f"{len(eligible)} were dated on or before {prior}. Undated candidates were excluded."
        ),
    )


def _reporter_roots(document: Document) -> tuple[CitationRecord, ...]:
    return tuple(
        record
        for record in document.active_citations
        if record.is_root and isinstance(record.fields, FullCaseCitation)
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


def _saved_courtlistener_full_reporter_search(record: CitationRecord) -> FullReporterSearchNode:
    """Restore the one persisted CourtListener metadata-search record."""
    return _saved_full_reporter_search(
        record,
        stage=COURTLISTENER_FULL_REPORTER_SEARCH_STAGE,
        node_type=FullReporterSearchNode,
    )


def _saved_govinfo_full_reporter_search(record: CitationRecord) -> GovInfoFullReporterSearchNode:
    """Restore the one persisted GovInfo metadata-search record."""
    return _saved_full_reporter_search(
        record,
        stage=GOVINFO_FULL_REPORTER_SEARCH_STAGE,
        node_type=GovInfoFullReporterSearchNode,
    )


def _saved_full_reporter_search(
    record: CitationRecord,
    *,
    stage: str,
    node_type: type[FullReporterSearchNode] | type[GovInfoFullReporterSearchNode],
) -> FullReporterSearchNode | GovInfoFullReporterSearchNode:
    matches = [
        node
        for node in record.trace
        if node.stage == stage and node.details.get("validation_node_type") == node_type.__name__
    ]
    if len(matches) != 1:
        msg = f"Expected exactly one saved {node_type.__name__} for {record.citation_id!r}, found {len(matches)}"
        raise ValueError(msg)
    raw = matches[0].details.get("validation")
    if not isinstance(raw, dict):
        msg = f"Saved {node_type.__name__} for {record.citation_id!r} has no validation payload"
        raise ValueError(msg)
    restored = deserialize_validation_node({"node_type": node_type.__name__, **raw})
    if not isinstance(restored, node_type):
        msg = f"Saved {node_type.__name__} for {record.citation_id!r} decoded as {type(restored).__name__}"
        raise ValueError(msg)
    return restored


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
    document_text: str,
) -> None:
    """Project typed evidence, source rereads, and the terminal identity state."""
    for typed in progression.nodes:
        if (
            not isinstance(typed, (MelleaCaseNameReextractionNode, MelleaCaseNameReviewNode))
            or typed.status is not ValidationNodeStatus.SUCCEEDED
            or typed.case_name is None
        ):
            continue
        reread = typed.case_name
        if reread is None or (
            reread.span.end > record.locator_span.start
            or document_text[reread.span.start : reread.span.end] != reread.text
        ):
            msg = f"Re-extracted case name for {record.citation_id!r} is not grounded before its locator"
            raise ValueError(msg)
    projected = {_node.node_id: _trace_node(_node, stage=stage) for _node in progression.nodes}
    for node in projected.values():
        observe_citation(record, node)
    for typed in progression.nodes:
        if not isinstance(typed, (MelleaCaseNameReextractionNode, MelleaCaseNameReviewNode)):
            continue
        if typed.status is not ValidationNodeStatus.SUCCEEDED:
            continue
        source_node = projected[typed.node_id]
        mark_extraction_reviewed(record, source_node)
        changes: dict[CitationField, object] = {}
        if typed.case_name is not None:
            reread = typed.case_name
            changes = {
                field: value
                for field, value in (
                    (CitationField.CASE_NAME, reread),
                    (CitationField.PLAINTIFF, reread.plaintiff),
                    (CitationField.DEFENDANT, reread.defendant),
                )
                if getattr(record.fields, field.value) != value
            }
        elif typed.plaintiff is not None or typed.defendant is not None:
            # A fragment does not establish a complete CaseName, but a party
            # copied from that fragment is still a durable source reading.
            changes = {
                field: value
                for field, value in (
                    (CitationField.PLAINTIFF, typed.plaintiff),
                    (CitationField.DEFENDANT, typed.defendant),
                )
                if value is not None and getattr(record.fields, field.value) != value
            }
        update_fields(
            record,
            source_node,
            changes,
            reason="Model re-read the case name from the target citation's source text.",
        )

    resolution = progression.identity_resolution
    if resolution is None:
        msg = f"Root identity for {record.citation_id!r} ended without an identity decision"
        raise ValueError(msg)
    resolution_node = projected[resolution.node_id]

    if resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED:
        cluster = _selected_cluster(progression, resolution)
        resolve_citation(
            record,
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
            attribute_authority(record, resolution_node, cluster.cluster_id)

    judge_citation(
        record,
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


def _write_metadata_identity_progression(
    record: CitationRecord,
    progression: CitationValidation,
    *,
    stage: str,
) -> None:
    """Project metadata-candidate evidence and its one identity decision."""
    projected = {node.node_id: _trace_node(node, stage=stage) for node in progression.nodes}
    for node in projected.values():
        observe_citation(record, node)

    resolution = progression.identity_resolution
    if resolution is None:
        msg = f"Metadata identity for {record.citation_id!r} ended without an identity decision"
        raise ValueError(msg)
    resolution_node = projected[resolution.node_id]
    if resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED:
        candidate = _selected_metadata_candidate(progression, resolution)
        resolve_citation(
            record,
            resolution_node,
            Resolution(
                cluster_id=candidate.cluster_id,
                case_name=candidate.case_name,
                date_filed=candidate.date_filed,
                court_id=candidate.court_id,
                node_id=resolution_node.node_id,
                docket_id=candidate.docket_id,
                govinfo_package_id=candidate.govinfo_package_id,
            ),
        )
        if candidate.docket_id is not None:
            attribute_authority(record, resolution_node, f"courtlistener:docket:{candidate.docket_id}")
        elif candidate.govinfo_package_id is not None:
            attribute_authority(record, resolution_node, f"govinfo:package:{candidate.govinfo_package_id}")
    judge_citation(
        record,
        resolution_node,
        Question.IDENTITY,
        resolution.outcome.value,
        message=resolution.outcome_message,
    )


def _selected_metadata_candidate(
    progression: CitationValidation,
    resolution: LocatorIdentityResolutionNode,
) -> CandidateEvaluationNode:
    """Return the metadata candidate explicitly selected by this resolution."""
    candidate = next(
        (
            node
            for node in progression.nodes
            if isinstance(node, CandidateEvaluationNode)
            and node.source
            in {
                CandidateEvaluationSource.FULL_REPORTER_METADATA_SEARCH,
                CandidateEvaluationSource.GOVINFO_FULL_REPORTER_METADATA_SEARCH,
            }
            and node.candidate_index == resolution.selected_candidate_index
        ),
        None,
    )
    if candidate is None:
        msg = "A resolved reporter metadata identity must select one persisted provider candidate"
        raise ValueError(msg)
    return candidate


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
    if isinstance(node, (MelleaCaseNameReextractionNode, MelleaCaseNameReviewNode)):
        return Reads.DOCUMENT
    return Reads.DOCUMENT if getattr(node, "run", None) is not None else Reads.RECORD


def _message(payload: dict[str, object]) -> str | None:
    """Prefer the operation's outcome explanation in the normalized node surface."""
    for key in ("outcome_message", "status_message", "error"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None
