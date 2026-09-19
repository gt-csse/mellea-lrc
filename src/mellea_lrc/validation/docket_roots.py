"""Document-native, checkpointable docket-root identity stages.

A docket number has no CourtListener exact-citation endpoint.  Its retrieval is
therefore a docket search, using the stated court only as an optional narrowing
constraint.  Search, unambiguous resolution, and bounded ambiguity review are
separate ``Document -> Document`` stages so each checkpoint can be inspected or
resumed without repeating earlier work.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.core.record import UNJUDGED, Node, Question, Reads, Resolution
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.serialization.validated_document import deserialize_validation_node
from mellea_lrc.validation.aggregation.citation_summary_candidate import citation_summary_candidate
from mellea_lrc.validation.aggregation.citation_summary_outcome import overall_locator_citation_outcome
from mellea_lrc.validation.aggregation.locator_identity import (
    requires_mellea_locator_candidate_choice,
    run_locator_identity_resolution,
)
from mellea_lrc.validation.aggregation.mellea_locator_candidate_choice import (
    run_mellea_locator_candidate_choice,
)
from mellea_lrc.validation.candidate_evaluation import run_docket_search_candidate_evaluation
from mellea_lrc.validation.field_checks.court_check import run_court_check
from mellea_lrc.validation.field_checks.docket_number_check import run_docket_number_check
from mellea_lrc.validation.field_checks.exact_case_name_check import run_exact_case_name_check
from mellea_lrc.validation.field_checks.mellea_docket_number_review import (
    run_mellea_docket_number_review,
)
from mellea_lrc.validation.field_checks.year_check import run_year_check
from mellea_lrc.validation.root_context import masked_root_context
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    CandidateProvenance,
    CitationValidation,
    DocketNumberCheckNode,
    DocketRootSearchNode,
    DocketRootSearchOutcome,
    ExactCaseNameCheckNode,
    FieldCheckOutcome,
    LocatorCandidateAssessmentNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorCitationSummaryOutcome,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    MelleaDocketNumberReviewNode,
    MelleaDocketNumberReviewOutcome,
    ValidationNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea import MelleaSession

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import Document


DOCKET_ROOT_SEARCH_STAGE = "docket_root_search"
DOCKET_ROOT_UNIQUE_IDENTITY_STAGE = "docket_root_unique_identity"
DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE = "docket_root_ambiguity_resolution"
DOCKET_ROOT_LOCATOR_REVIEW_STAGE = "docket_root_locator_review"
DOCKET_ROOT_RELOOKUP_STAGE = "docket_root_relookup"
DOCKET_ROOT_RELOOKUP_UNIQUE_IDENTITY_STAGE = "docket_root_relookup_unique_identity"
DOCKET_ROOT_RELOOKUP_AMBIGUITY_RESOLUTION_STAGE = "docket_root_relookup_ambiguity_resolution"
MAX_DOCKET_CANDIDATE_REVIEW = 20
_MADE_BY = "mellea_lrc.validation.docket_roots"


async def search_docket_roots(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
) -> Document:
    """Retrieve and persist CourtListener docket candidates for every docket root.

    The query is the stated docket number, with ``court_id:<court>`` appended
    only when extraction already read a court.  No court is invented and a
    courtless docket remains searchable.  Candidate lists below the review
    limit are fully stored. A result set of twenty or more is outside this
    stage's decision boundary, so its count, first response, and continuation
    cursor are retained without exhausting a potentially unbounded search.
    """
    _require_stage(document, ROOT_FORMATION_STAGE, "Docket-root search")
    if DOCKET_ROOT_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_SEARCH_STAGE)

    service = client if client is not None else CourtListenerClient()
    for record in _docket_roots(document):
        search = _run_docket_root_search(record, service)
        trace_node = _trace_node(search, stage=DOCKET_ROOT_SEARCH_STAGE)
        record.observe(trace_node)
        record.judge(
            trace_node,
            Question.DOCKET_LOOKUP,
            _docket_lookup_outcome(search.outcome),
            message=search.outcome_message,
        )
    return replace(document, passes=(*document.passes, DOCKET_ROOT_SEARCH_STAGE))


async def validate_unique_docket_root_identities(
    document: Document,
    *,
    session: MelleaSession | None = None,
) -> Document:
    """Resolve saved zero- or one-candidate docket-search results only.

    A found candidate must state the same docket number under the shared
    grounding policy.  Court and date disagreements are preserved and deferred
    to their separate semantic-correction stage.  A missing or mismatched case
    name is recorded, but does not negate a matching docket identifier.
    """
    _require_stage(document, DOCKET_ROOT_SEARCH_STAGE, "Unique docket-root identity")
    if DOCKET_ROOT_UNIQUE_IDENTITY_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_UNIQUE_IDENTITY_STAGE)

    for record in _docket_roots(document):
        search = _saved_docket_root_search(record)
        if record.judgement(Question.IDENTITY).outcome != UNJUDGED:
            continue
        if search.outcome is DocketRootSearchOutcome.NOT_FOUND:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _no_match_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    reason="CourtListener docket search returned no candidate.",
                )
            )
        elif search.outcome is DocketRootSearchOutcome.FOUND:
            progression = await _review_docket_candidates(
                record,
                search=search,
                document=document,
                session=session,
            )
        else:
            continue
        _write_identity_progression(record, progression, stage=DOCKET_ROOT_UNIQUE_IDENTITY_STAGE)
    return replace(document, passes=(*document.passes, DOCKET_ROOT_UNIQUE_IDENTITY_STAGE))


async def resolve_docket_root_ambiguities(
    document: Document,
    *,
    session: MelleaSession | None = None,
) -> Document:
    """Resolve bounded docket-search candidate lists after unique candidates.

    Every candidate returned by a bounded search is assessed and retained.  A
    model chooses only when deterministic evidence does not identify exactly
    one candidate.  Candidate sets of twenty or more are explicitly deferred;
    future opinion-reading control belongs to a later stage.
    """
    _require_stage(document, DOCKET_ROOT_UNIQUE_IDENTITY_STAGE, "Docket-root ambiguity resolution")
    if DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE)

    for record in _docket_roots(document):
        search = _saved_docket_root_search(record)
        if record.judgement(Question.IDENTITY).outcome != UNJUDGED:
            continue
        if search.outcome is DocketRootSearchOutcome.AMBIGUOUS:
            progression = await _review_docket_candidates(
                record,
                search=search,
                document=document,
                session=session,
            )
        elif search.outcome is DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _deferred_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    reason=(
                        f"Docket search returned {search.candidate_count} candidates; "
                        f"the review limit is {MAX_DOCKET_CANDIDATE_REVIEW}."
                    ),
                )
            )
        elif search.outcome is DocketRootSearchOutcome.FAILED:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _deferred_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    reason="Docket search failed; no identity decision was admitted.",
                )
            )
        else:
            continue
        _write_identity_progression(record, progression, stage=DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE)
    return replace(document, passes=(*document.passes, DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE))


async def review_unresolved_docket_root_locators(
    document: Document,
    *,
    session: MelleaSession | None = None,
) -> Document:
    """Ask once whether a no-match docket root was parsed with the right number.

    This is deliberately after the initial bounded docket route: a model is not
    asked merely because a candidate list is ambiguous or too large.  It reads
    only a root whose initial route reached ``no_match``.  Its own first-class
    judgement records every terminal review state, including a failed model
    run, so an unchanged document can never loop back into this model call.
    """
    _require_stage(document, DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE, "Docket-root locator review")
    if DOCKET_ROOT_LOCATOR_REVIEW_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_LOCATOR_REVIEW_STAGE)

    for record in _docket_roots(document):
        if record.judgement(Question.IDENTITY).outcome != LocatorIdentityResolutionOutcome.NO_MATCH.value:
            continue
        if record.docket_number_reviewed_by_model:
            continue
        trigger_node_id = record.judgement(Question.IDENTITY).node_id
        if trigger_node_id is None:
            msg = f"No-match docket root {record.citation_id!r} has no identity decision node"
            raise ValueError(msg)
        review = await run_mellea_docket_number_review(
            record,
            document=document,
            trigger_node_id=trigger_node_id,
            session=session,
        )
        trace_node = _trace_node(
            review,
            stage=DOCKET_ROOT_LOCATOR_REVIEW_STAGE,
            reads=Reads.DOCUMENT,
        )
        if review.status is ValidationNodeStatus.SUCCEEDED:
            record.mark_stated_fields_reparsed_by_model()
        if review.outcome is MelleaDocketNumberReviewOutcome.CORRECTED:
            if review.grounded_docket_number is None:
                msg = "A corrected docket review requires a grounded source docket number"
                raise ValueError(msg)
            record.correct(
                trace_node,
                "docket_number",
                review.grounded_docket_number,
                reason=review.reason or review.outcome_message or "Model re-read the source docket locator.",
            )
        else:
            record.observe(trace_node)
        record.judge(
            trace_node,
            Question.DOCKET_LOCATOR_REVIEW,
            review.outcome.value,
            message=review.outcome_message or review.reason,
        )
    return replace(document, passes=(*document.passes, DOCKET_ROOT_LOCATOR_REVIEW_STAGE))


async def relookup_reviewed_docket_roots(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
) -> Document:
    """Re-enqueue only source-grounded docket corrections for one fresh search.

    The review stage never mutates a CourtListener finding.  A correction
    instead creates a second, explicitly dependent search node and changes the
    live identity judgement to ``deferred_to_search`` until its own unique or
    ambiguity stage decides it.  The original no-match node remains in trace.
    """
    _require_stage(document, DOCKET_ROOT_LOCATOR_REVIEW_STAGE, "Reviewed docket-root relookup")
    if DOCKET_ROOT_RELOOKUP_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_RELOOKUP_STAGE)

    service = client if client is not None else CourtListenerClient()
    for record in _docket_roots(document):
        review = _saved_docket_number_review(record)
        if review is None or review.outcome is not MelleaDocketNumberReviewOutcome.CORRECTED:
            continue
        search = _run_docket_root_search(
            record,
            service,
            node_id=f"{record.citation_id}:docket_root_relookup",
            depends_on=(review.node_id,),
        )
        trace_node = _trace_node(search, stage=DOCKET_ROOT_RELOOKUP_STAGE)
        record.observe(trace_node)
        record.judge(
            trace_node,
            Question.DOCKET_LOOKUP,
            _docket_lookup_outcome(search.outcome),
            message=search.outcome_message,
        )
        record.judge(
            trace_node,
            Question.IDENTITY,
            LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH.value,
            message="The model-corrected docket number was queued for a fresh CourtListener search.",
        )
    return replace(document, passes=(*document.passes, DOCKET_ROOT_RELOOKUP_STAGE))


async def validate_unique_relooked_up_docket_root_identities(
    document: Document,
    *,
    session: MelleaSession | None = None,
) -> Document:
    """Resolve saved zero- or one-candidate searches after one docket correction."""
    _require_stage(document, DOCKET_ROOT_RELOOKUP_STAGE, "Re-looked-up docket-root unique identity")
    if DOCKET_ROOT_RELOOKUP_UNIQUE_IDENTITY_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_RELOOKUP_UNIQUE_IDENTITY_STAGE)

    for record in _docket_roots(document):
        if (
            record.judgement(Question.IDENTITY).outcome
            != LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH.value
        ):
            continue
        search = _saved_docket_root_search(record, stage=DOCKET_ROOT_RELOOKUP_STAGE)
        if search.outcome is DocketRootSearchOutcome.NOT_FOUND:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _no_match_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    reason="CourtListener returned no candidate for the model-corrected docket number.",
                    scope=search.node_id,
                )
            )
        elif search.outcome is DocketRootSearchOutcome.FOUND:
            progression = await _review_docket_candidates(
                record,
                search=search,
                document=document,
                session=session,
            )
        else:
            continue
        _write_identity_progression(record, progression, stage=DOCKET_ROOT_RELOOKUP_UNIQUE_IDENTITY_STAGE)
    return replace(document, passes=(*document.passes, DOCKET_ROOT_RELOOKUP_UNIQUE_IDENTITY_STAGE))


async def resolve_relooked_up_docket_root_ambiguities(
    document: Document,
    *,
    session: MelleaSession | None = None,
) -> Document:
    """Finish bounded candidate review for one model-corrected docket relookup."""
    _require_stage(
        document,
        DOCKET_ROOT_RELOOKUP_UNIQUE_IDENTITY_STAGE,
        "Re-looked-up docket-root ambiguity resolution",
    )
    if DOCKET_ROOT_RELOOKUP_AMBIGUITY_RESOLUTION_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_RELOOKUP_AMBIGUITY_RESOLUTION_STAGE)

    for record in _docket_roots(document):
        if (
            record.judgement(Question.IDENTITY).outcome
            != LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH.value
        ):
            continue
        search = _saved_docket_root_search(record, stage=DOCKET_ROOT_RELOOKUP_STAGE)
        if search.outcome is DocketRootSearchOutcome.AMBIGUOUS:
            progression = await _review_docket_candidates(
                record,
                search=search,
                document=document,
                session=session,
            )
        elif search.outcome is DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _deferred_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    reason=(
                        f"Re-looked-up docket search returned {search.candidate_count} candidates; "
                        f"the review limit is {MAX_DOCKET_CANDIDATE_REVIEW}."
                    ),
                    scope=search.node_id,
                )
            )
        elif search.outcome is DocketRootSearchOutcome.FAILED:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _deferred_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    reason="Re-looked-up docket search failed; no identity decision was admitted.",
                    scope=search.node_id,
                )
            )
        else:
            continue
        _write_identity_progression(
            record,
            progression,
            stage=DOCKET_ROOT_RELOOKUP_AMBIGUITY_RESOLUTION_STAGE,
        )
    return replace(document, passes=(*document.passes, DOCKET_ROOT_RELOOKUP_AMBIGUITY_RESOLUTION_STAGE))


def _run_docket_root_search(
    record: CitationRecord,
    client: CourtListenerServiceClient,
    *,
    node_id: str | None = None,
    depends_on: tuple[str, ...] = (),
) -> DocketRootSearchNode:
    citation = record.stated
    if not isinstance(citation, DocketCitation) or not citation.docket_number:
        return _search_node(
            record,
            status=ValidationNodeStatus.FAILED,
            outcome=DocketRootSearchOutcome.FAILED,
            docket_number=citation.docket_number if isinstance(citation, DocketCitation) else None,
            query=None,
            node_id=node_id,
            depends_on=depends_on,
            status_message="Docket-root search could not run.",
            outcome_message="The root has no stated docket number.",
            error="Docket root lacks a docket number",
        )

    query = _docket_query(citation)
    try:
        result = client.search(query, "d")
        outcome = _search_outcome(result.count)
        if outcome is DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT:
            # This stage defers sets at the review boundary. Paging an
            # unbounded archive result would add no admissible identity
            # evidence; a future scoped review can resume from this cursor.
            return _search_node(
                record,
                status=ValidationNodeStatus.SUCCEEDED,
                outcome=outcome,
                docket_number=citation.docket_number,
                query=query,
                node_id=node_id,
                depends_on=depends_on,
                candidate_count=result.count,
                candidates=tuple(result.results),
                next_cursor=result.next_cursor,
                status_message="CourtListener docket search reached the review boundary.",
                outcome_message=_search_message(outcome, result.count),
            )
        candidates, next_cursor = _all_docket_candidates(client, query, result)
        if len(candidates) != result.count:
            return _search_node(
                record,
                status=ValidationNodeStatus.FAILED,
                outcome=DocketRootSearchOutcome.FAILED,
                docket_number=citation.docket_number,
                query=query,
                node_id=node_id,
                depends_on=depends_on,
                candidate_count=result.count,
                candidates=candidates,
                next_cursor=next_cursor,
                status_message="Docket-root search returned an incomplete result set.",
                outcome_message="Every docket-search candidate must be stored before later review.",
                error=f"Search reported {result.count} candidates but returned {len(candidates)} across its pages",
            )
        return _search_node(
            record,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=outcome,
            docket_number=citation.docket_number,
            query=query,
            node_id=node_id,
            depends_on=depends_on,
            candidate_count=result.count,
            candidates=candidates,
            next_cursor=next_cursor,
            status_message="CourtListener docket search completed.",
            outcome_message=_search_message(outcome, result.count),
        )
    except Exception as exc:
        return _search_node(
            record,
            status=ValidationNodeStatus.FAILED,
            outcome=DocketRootSearchOutcome.FAILED,
            docket_number=citation.docket_number,
            query=query,
            node_id=node_id,
            depends_on=depends_on,
            status_message="CourtListener docket search failed.",
            outcome_message="No docket candidates were available for identity review.",
            error=f"{type(exc).__name__}: {exc}",
        )


def _all_docket_candidates(
    client: CourtListenerServiceClient,
    query: str,
    first_page,
) -> tuple[tuple[Mapping[str, object], ...], str | None]:
    """Follow the docket search cursor until its reported result set is complete."""
    candidates = list(first_page.results)
    cursor = first_page.next_cursor
    seen_cursors: set[str] = set()
    while cursor is not None:
        if cursor in seen_cursors:
            msg = "CourtListener repeated a docket-search cursor"
            raise ValueError(msg)
        seen_cursors.add(cursor)
        page = client.search(query, "d", cursor=cursor)
        if page.count != first_page.count:
            msg = "CourtListener changed the docket-search result count while paging"
            raise ValueError(msg)
        candidates.extend(page.results)
        cursor = page.next_cursor
    return tuple(candidates), cursor


def _docket_query(citation: DocketCitation) -> str:
    """Build the one docket-search query, optionally narrowed by a known court."""
    assert citation.docket_number is not None
    return " ".join(
        part
        for part in (citation.docket_number, f"court_id:{citation.court}" if citation.court else None)
        if part
    )


def _search_outcome(count: int) -> DocketRootSearchOutcome:
    if count == 0:
        return DocketRootSearchOutcome.NOT_FOUND
    if count == 1:
        return DocketRootSearchOutcome.FOUND
    if count < MAX_DOCKET_CANDIDATE_REVIEW:
        return DocketRootSearchOutcome.AMBIGUOUS
    return DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT


def _search_message(outcome: DocketRootSearchOutcome, count: int) -> str:
    return {
        DocketRootSearchOutcome.NOT_FOUND: "CourtListener docket search returned no candidates.",
        DocketRootSearchOutcome.FOUND: "CourtListener docket search returned one candidate.",
        DocketRootSearchOutcome.AMBIGUOUS: f"CourtListener docket search returned {count} candidates.",
        DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT: (
            f"CourtListener docket search returned {count} candidates, at or above the review limit."
        ),
    }[outcome]


def _search_node(
    record: CitationRecord,
    *,
    status: ValidationNodeStatus,
    outcome: DocketRootSearchOutcome,
    docket_number: str | None,
    query: str | None,
    node_id: str | None = None,
    depends_on: tuple[str, ...] = (),
    candidate_count: int = 0,
    candidates: tuple[Mapping[str, object], ...] = (),
    next_cursor: str | None = None,
    status_message: str | None,
    outcome_message: str | None,
    error: str | None = None,
) -> DocketRootSearchNode:
    return DocketRootSearchNode(
        node_id=node_id or f"{record.citation_id}:docket_root_search",
        status=status,
        outcome=outcome,
        docket_number=docket_number,
        query=query,
        candidate_count=candidate_count,
        candidates=candidates,
        next_cursor=next_cursor,
        depends_on=depends_on,
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
    )


async def _review_docket_candidates(
    record: CitationRecord,
    *,
    search: DocketRootSearchNode,
    document: Document,
    session: MelleaSession | None,
) -> CitationValidation:
    """Assess every stored candidate, then use deterministic or bounded model choice."""
    validation = CitationValidation(citation=record, nodes=(search,))
    scope = search.node_id
    for index, result in enumerate(search.candidates, start=1):
        candidate = run_docket_search_candidate_evaluation(
            validation,
            result=result,
            candidate_index=index,
            depends_on=(search.node_id,),
            node_prefix=scope,
        )
        validation = validation.append(candidate)
        docket = run_docket_number_check(validation, candidate=candidate)
        case_name = run_exact_case_name_check(validation, candidate=candidate)
        year = run_year_check(validation, candidate=candidate)
        court = run_court_check(validation, evidence=candidate)
        validation = validation.append(docket).append(case_name).append(year).append(court)
        validation = validation.append(
            _docket_candidate_assessment(
                validation,
                candidate=candidate,
                docket=docket,
                case_name=case_name,
                year_outcome=year.outcome,
                court_outcome=court.outcome,
            )
        )

    summary = _docket_citation_summary(validation, scope=scope)
    validation = validation.append(summary)
    eligible = _docket_number_matches(validation)
    if not eligible:
        # The model may choose among candidates that state the same docket, but
        # cannot turn a differently written docket number into an identity. A
        # later explicitly configured fuzzy comparison can add candidates here.
        return validation.append(
            _no_match_resolution(
                validation,
                depends_on=(summary.node_id,),
                reason="No retrieved candidate reproduced the stated docket number.",
                scope=scope,
            )
        )
    if requires_mellea_locator_candidate_choice(summary):
        choice = await run_mellea_locator_candidate_choice(
            validation,
            summary=summary,
            document_text=masked_root_context(document, record).as_document_text(
                document_length=len(document.text)
            ),
            session=session,
            eligible_candidate_indices=eligible,
        )
        validation = validation.append(choice)
        resolution = run_locator_identity_resolution(validation, summary=summary, choice=choice)
    else:
        resolution = run_locator_identity_resolution(validation, summary=summary)
    return validation.append(replace(resolution, node_id=f"{scope}:identity_resolution"))


def _docket_number_matches(validation: CitationValidation) -> tuple[int, ...]:
    """Return candidate indexes whose retrieved docket number matched deterministically."""
    checks = {
        node.node_id.removesuffix(":docket_number_check"): node
        for node in validation.nodes
        if isinstance(node, DocketNumberCheckNode)
    }
    return tuple(
        node.candidate_index
        for node in validation.nodes
        if isinstance(node, CandidateEvaluationNode)
        and node.source is CandidateEvaluationSource.DOCKET_SEARCH
        and checks.get(node.node_id) is not None
        and checks[node.node_id].outcome is FieldCheckOutcome.MATCH
    )


def _docket_candidate_assessment(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
    docket: DocketNumberCheckNode,
    case_name: ExactCaseNameCheckNode,
    year_outcome: FieldCheckOutcome,
    court_outcome: FieldCheckOutcome,
) -> LocatorCandidateAssessmentNode:
    """Reduce docket identity evidence while preserving non-docket field findings."""
    docket_outcome = AggregatedFieldOutcome(docket.outcome.value)
    case_outcome = AggregatedFieldOutcome(case_name.outcome.value)
    year = AggregatedFieldOutcome(year_outcome.value)
    court = AggregatedFieldOutcome(court_outcome.value)
    if docket_outcome is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorCandidateAssessmentOutcome.MISMATCH
        message = "The retrieved candidate states a different docket number."
    elif docket_outcome is AggregatedFieldOutcome.UNAVAILABLE:
        outcome = LocatorCandidateAssessmentOutcome.PARTIAL_MATCH
        message = "The retrieved candidate lacks a docket number for direct identity comparison."
    elif court is AggregatedFieldOutcome.MISMATCH or year is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorCandidateAssessmentOutcome.PARTIAL_MATCH
        message = "The docket number matches, but a stated court or date needs semantic correction."
    else:
        outcome = LocatorCandidateAssessmentOutcome.MATCH
        message = (
            "The docket number matches and no stated court or date conflicts. "
            "Any case-name difference remains recorded for the later case-name stage."
        )
    return LocatorCandidateAssessmentNode(
        node_id=f"{candidate.node_id}:docket_candidate_assessment",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        candidate_index=candidate.candidate_index,
        extracted_citation=validation.citation.matched_text,
        extracted_case_name=case_name.extracted_case_name,
        retrieved_case_name=case_name.retrieved_case_name,
        case_name_outcome=case_outcome,
        case_name_evidence="exact",
        extracted_year=_year(validation, candidate),
        retrieved_year=candidate.year,
        year_outcome=year,
        extracted_court_id=_court(validation),
        retrieved_court_id=candidate.court_id,
        court_outcome=court,
        docket_id=candidate.docket_id,
        depends_on=(
            docket.node_id,
            case_name.node_id,
            f"{candidate.node_id}:year_check",
            f"{candidate.node_id}:court_check",
        ),
        status_message="Docket candidate assessment completed.",
        outcome_message=message,
    )


def _docket_citation_summary(
    validation: CitationValidation,
    *,
    scope: str,
) -> LocatorCitationSummaryNode:
    assessments = tuple(node for node in validation.nodes if isinstance(node, LocatorCandidateAssessmentNode))
    if not assessments:
        msg = "Docket candidate summary requires at least one candidate assessment"
        raise ValueError(msg)
    candidates = tuple(
        citation_summary_candidate(validation, assessment, provenance=CandidateProvenance.DOCKET)
        for assessment in assessments
    )
    return LocatorCitationSummaryNode(
        node_id=f"{scope}:docket_citation_summary",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorCitationSummaryOutcome.COMPLETE,
        overall_outcome=overall_locator_citation_outcome(candidate.outcome for candidate in candidates),
        pinpoint_requires_review=None,
        candidates=candidates,
        depends_on=tuple(candidate.assessment_node_id for candidate in candidates),
        status_message="Docket candidate summary completed.",
        outcome_message=f"Listed {len(candidates)} evaluated docket candidates without selecting one.",
    )


def _no_match_resolution(
    validation: CitationValidation,
    *,
    depends_on: tuple[str, ...],
    reason: str,
    scope: str | None = None,
) -> LocatorIdentityResolutionNode:
    return LocatorIdentityResolutionNode(
        node_id=f"{scope or validation.citation_id}:locator_identity_resolution",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorIdentityResolutionOutcome.NO_MATCH,
        selected_candidate_index=None,
        selected_assessment_node_id=None,
        matching_candidate_indices=(),
        selection_evidence_node_id=None,
        depends_on=depends_on,
        status_message="Docket-root identity resolution completed.",
        outcome_message=reason,
    )


def _deferred_resolution(
    validation: CitationValidation,
    *,
    depends_on: tuple[str, ...],
    reason: str,
    scope: str | None = None,
) -> LocatorIdentityResolutionNode:
    return LocatorIdentityResolutionNode(
        node_id=f"{scope or validation.citation_id}:locator_identity_resolution",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION,
        selected_candidate_index=None,
        selected_assessment_node_id=None,
        matching_candidate_indices=(),
        selection_evidence_node_id=None,
        depends_on=depends_on,
        status_message="Docket-root identity resolution deferred to future implementation.",
        outcome_message=reason,
    )


def _write_identity_progression(
    record: CitationRecord,
    progression: CitationValidation,
    *,
    stage: str,
) -> None:
    projected = {node.node_id: _trace_node(node, stage=stage) for node in progression.nodes}
    for node in projected.values():
        record.observe(node)
    resolution = progression.identity_resolution
    if resolution is None:
        msg = f"Docket-root identity for {record.citation_id!r} ended without a decision"
        raise ValueError(msg)
    resolution_node = projected[resolution.node_id]
    if resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED:
        candidate = _selected_docket_candidate(progression, resolution)
        record.resolve(
            resolution_node,
            Resolution(
                cluster_id=None,
                case_name=candidate.case_name,
                date_filed=candidate.date_filed,
                court_id=candidate.court_id,
                node_id=resolution_node.node_id,
                docket_id=candidate.docket_id,
            ),
        )
        if candidate.docket_id is not None:
            # This is the resolved authority pointer.  It deliberately names
            # the docket resource rather than pretending CourtListener's docket
            # id is an opinion-cluster id.
            record.reattribute(resolution_node, f"courtlistener:docket:{candidate.docket_id}")
    record.judge(
        resolution_node,
        Question.IDENTITY,
        resolution.outcome.value,
        message=resolution.outcome_message,
    )


def _selected_docket_candidate(
    progression: CitationValidation,
    resolution: LocatorIdentityResolutionNode,
) -> CandidateEvaluationNode:
    candidate = next(
        (
            node
            for node in progression.nodes
            if isinstance(node, CandidateEvaluationNode)
            and node.source is CandidateEvaluationSource.DOCKET_SEARCH
            and node.candidate_index == resolution.selected_candidate_index
        ),
        None,
    )
    if candidate is None:
        msg = "A resolved docket identity must select one stored docket-search candidate"
        raise ValueError(msg)
    return candidate


def _saved_docket_root_search(
    record: CitationRecord,
    *,
    stage: str = DOCKET_ROOT_SEARCH_STAGE,
) -> DocketRootSearchNode:
    matches = [
        node
        for node in record.trace
        if node.stage == stage and node.details.get("validation_node_type") == DocketRootSearchNode.__name__
    ]
    if len(matches) != 1:
        msg = (
            f"Expected exactly one saved docket search for {record.citation_id!r} in {stage!r}, "
            f"found {len(matches)}"
        )
        raise ValueError(msg)
    raw = matches[0].details.get("validation")
    if not isinstance(raw, dict):
        msg = f"Saved docket search for {record.citation_id!r} has no validation payload"
        raise ValueError(msg)
    node = deserialize_validation_node({"node_type": DocketRootSearchNode.__name__, **raw})
    if not isinstance(node, DocketRootSearchNode):
        msg = f"Saved docket search for {record.citation_id!r} decoded as {type(node).__name__}"
        raise ValueError(msg)
    return node


def _saved_docket_number_review(record: CitationRecord) -> MelleaDocketNumberReviewNode | None:
    """Return the one persisted docket-number review, if this root needed one."""
    matches = [
        node
        for node in record.trace
        if node.stage == DOCKET_ROOT_LOCATOR_REVIEW_STAGE
        and node.details.get("validation_node_type") == MelleaDocketNumberReviewNode.__name__
    ]
    if not matches:
        return None
    if len(matches) != 1:
        msg = f"Expected at most one docket-number review for {record.citation_id!r}, found {len(matches)}"
        raise ValueError(msg)
    raw = matches[0].details.get("validation")
    if not isinstance(raw, dict):
        msg = f"Saved docket-number review for {record.citation_id!r} has no validation payload"
        raise ValueError(msg)
    node = deserialize_validation_node({"node_type": MelleaDocketNumberReviewNode.__name__, **raw})
    if not isinstance(node, MelleaDocketNumberReviewNode):
        msg = f"Saved docket-number review for {record.citation_id!r} decoded as {type(node).__name__}"
        raise ValueError(msg)
    return node


def _docket_roots(document: Document) -> tuple[CitationRecord, ...]:
    return tuple(
        record
        for record in document.active_citations
        if record.is_root and isinstance(record.stated, DocketCitation)
    )


def _docket_lookup_outcome(outcome: DocketRootSearchOutcome) -> str:
    return {
        DocketRootSearchOutcome.FOUND: "found",
        DocketRootSearchOutcome.NOT_FOUND: "not_found",
        DocketRootSearchOutcome.AMBIGUOUS: "deferred_to_ambiguity",
        DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT: "deferred_to_future_implementation",
        DocketRootSearchOutcome.FAILED: "failed",
    }[outcome]


def _require_stage(document: Document, required_stage: str, stage_name: str) -> None:
    if required_stage not in document.passes:
        msg = f"{stage_name} requires {required_stage} before validation."
        raise ValueError(msg)


def _reject_partial_stage(document: Document, stage: str) -> None:
    if any(node.stage == stage for record in document.citations for node in record.trace):
        msg = f"Cannot resume partial {stage}; restart from the document before this stage."
        raise ValueError(msg)


def _trace_node(
    node: ValidationNode,
    *,
    stage: str,
    reads: Reads = Reads.RECORD,
) -> Node:
    payload = serialize_dataclass(node)
    return Node(
        node_id=node.node_id,
        reads=reads,
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


def _message(payload: dict[str, object]) -> str | None:
    for key in ("outcome_message", "status_message", "error"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _year(validation: CitationValidation, candidate: CandidateEvaluationNode) -> str | None:
    citation = validation.citation.stated
    if not isinstance(citation, DocketCitation) or citation.date is None:
        return None
    return citation.date.year


def _court(validation: CitationValidation) -> str | None:
    citation = validation.citation.stated
    return citation.court if isinstance(citation, DocketCitation) else None
