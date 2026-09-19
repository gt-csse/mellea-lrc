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
from mellea_lrc.govinfo import GovInfoClient, govinfo_package_candidate
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.serialization.validated_document import deserialize_validation_node
from mellea_lrc.validation.aggregation.citation_summary_candidate import citation_summary_candidate
from mellea_lrc.validation.aggregation.citation_summary_outcome import overall_locator_citation_outcome
from mellea_lrc.validation.aggregation.locator_identity import run_locator_identity_resolution
from mellea_lrc.validation.aggregation.mellea_locator_candidate_choice import (
    run_mellea_locator_candidate_choice,
)
from mellea_lrc.validation.candidate_evaluation import (
    run_docket_search_candidate_evaluation,
    run_govinfo_docket_search_candidate_evaluation,
)
from mellea_lrc.validation.field_checks.court_check import run_court_check
from mellea_lrc.validation.field_checks.docket_number_check import run_docket_number_check
from mellea_lrc.validation.field_checks.exact_case_name_check import run_exact_case_name_check
from mellea_lrc.validation.field_checks.mellea_docket_citation_reextraction import (
    run_mellea_docket_citation_reextraction,
)
from mellea_lrc.validation.field_checks.mellea_docket_number_equivalence import (
    run_mellea_docket_number_equivalence_check,
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
    GovInfoDocketSearchNode,
    LocatorCandidateAssessmentNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorCitationSummaryOutcome,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    MelleaDocketCitationReextractionNode,
    MelleaDocketCitationReextractionOutcome,
    MelleaDocketNumberEquivalenceNode,
    MelleaDocketNumberEquivalenceOutcome,
    MelleaLocatorCandidateChoiceNode,
    ValidationNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea import MelleaSession

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import Document
    from mellea_lrc.govinfo import GovInfoClient as GovInfoServiceClient


DOCKET_ROOT_SEARCH_STAGE = "docket_root_search"
DOCKET_ROOT_UNIQUE_IDENTITY_STAGE = "docket_root_unique_identity"
DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE = "docket_root_ambiguity_resolution"
DOCKET_ROOT_EXTRACTION_REVIEW_STAGE = "docket_root_extraction_review"
DOCKET_ROOT_REQUEUED_SEARCH_STAGE = "docket_root_requeued_search"
DOCKET_ROOT_REQUEUED_UNIQUE_IDENTITY_STAGE = "docket_root_requeued_search_unique_identity"
DOCKET_ROOT_REQUEUED_AMBIGUITY_RESOLUTION_STAGE = "docket_root_requeued_search_ambiguity_resolution"
DOCKET_ROOT_SEMANTIC_RESOLUTION_STAGE = "docket_root_semantic_resolution"
GOVINFO_DOCKET_ROOT_SEARCH_STAGE = "govinfo_docket_root_search"
GOVINFO_DOCKET_ROOT_UNIQUE_IDENTITY_STAGE = "govinfo_docket_root_unique_identity"
GOVINFO_DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE = "govinfo_docket_root_ambiguity_resolution"
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


async def validate_unique_docket_root_identities(document: Document) -> Document:
    """Resolve saved zero- or one-candidate docket-search results only.

    This stage uses only programmatic comparisons. It resolves a root only
    when the docket number and every stated identity field agree; all other
    results stay unresolved for extraction review or later semantic work.
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
            progression = _review_docket_candidates(record, search=search)
        else:
            continue
        _write_identity_progression(record, progression, stage=DOCKET_ROOT_UNIQUE_IDENTITY_STAGE)
    return replace(document, passes=(*document.passes, DOCKET_ROOT_UNIQUE_IDENTITY_STAGE))


async def resolve_docket_root_ambiguities(document: Document) -> Document:
    """Resolve bounded docket-search candidate lists after unique candidates.

    Every candidate returned by a bounded search is assessed and retained.
    This stage makes no model call: ambiguous or incomplete programmatic
    evidence stays unresolved for extraction review or later semantic work.
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
            progression = _review_docket_candidates(record, search=search)
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


async def lookup_govinfo_docket_roots(
    document: Document,
    *,
    client: GovInfoServiceClient | None = None,
) -> Document:
    """Persist a GovInfo USCOURTS fallback after a CourtListener miss.

    GovInfo carries published Federal opinions, not a comprehensive docket
    index. It is therefore an independent lookup route only for roots whose
    completed CourtListener search returned no candidate; it never reinterprets
    a bounded CourtListener result or searches every docket speculatively.
    """
    _require_stage(document, DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE, "GovInfo docket-root lookup")
    if GOVINFO_DOCKET_ROOT_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document, GOVINFO_DOCKET_ROOT_SEARCH_STAGE)

    service = client if client is not None else GovInfoClient()
    for record in _docket_roots(document):
        search = _saved_docket_root_search(record)
        if search.outcome is not DocketRootSearchOutcome.NOT_FOUND:
            continue
        result = _run_govinfo_docket_root_search(record, service, depends_on=(search.node_id,))
        trace_node = _trace_node(result, stage=GOVINFO_DOCKET_ROOT_SEARCH_STAGE)
        record.observe(trace_node)
        record.judge(
            trace_node,
            Question.DOCKET_LOOKUP,
            _govinfo_docket_lookup_outcome(result.outcome),
            message=result.outcome_message,
        )
    return replace(document, passes=(*document.passes, GOVINFO_DOCKET_ROOT_SEARCH_STAGE))


async def validate_unique_govinfo_docket_root_identities(document: Document) -> Document:
    """Resolve zero- or one-package GovInfo fallback results programmatically."""
    _require_stage(document, GOVINFO_DOCKET_ROOT_SEARCH_STAGE, "Unique GovInfo docket-root identity")
    if GOVINFO_DOCKET_ROOT_UNIQUE_IDENTITY_STAGE in document.passes:
        return document
    _reject_partial_stage(document, GOVINFO_DOCKET_ROOT_UNIQUE_IDENTITY_STAGE)

    for record in _docket_roots(document):
        search = _saved_govinfo_docket_root_search(record)
        if search is None:
            continue
        if search.outcome is DocketRootSearchOutcome.NOT_FOUND:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _no_match_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    reason="Neither CourtListener nor GovInfo returned a docket candidate.",
                    scope=search.node_id,
                )
            )
        elif search.outcome is DocketRootSearchOutcome.FOUND:
            progression = _review_govinfo_docket_candidates(record, search=search)
        else:
            continue
        _write_identity_progression(record, progression, stage=GOVINFO_DOCKET_ROOT_UNIQUE_IDENTITY_STAGE)
    return replace(document, passes=(*document.passes, GOVINFO_DOCKET_ROOT_UNIQUE_IDENTITY_STAGE))


async def resolve_govinfo_docket_root_ambiguities(document: Document) -> Document:
    """Resolve bounded GovInfo fallback results without suppressing candidates."""
    _require_stage(
        document,
        GOVINFO_DOCKET_ROOT_UNIQUE_IDENTITY_STAGE,
        "GovInfo docket-root ambiguity resolution",
    )
    if GOVINFO_DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE in document.passes:
        return document
    _reject_partial_stage(document, GOVINFO_DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE)

    for record in _docket_roots(document):
        search = _saved_govinfo_docket_root_search(record)
        if search is None:
            continue
        if search.outcome is DocketRootSearchOutcome.AMBIGUOUS:
            progression = _review_govinfo_docket_candidates(record, search=search)
        elif search.outcome is DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _deferred_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    scope=search.node_id,
                    reason=(
                        f"GovInfo returned {search.candidate_count} packages; "
                        f"the review limit is {MAX_DOCKET_CANDIDATE_REVIEW}."
                    ),
                )
            )
        elif search.outcome is DocketRootSearchOutcome.FAILED:
            # The failed fallback adds no identity evidence. Preserve the
            # completed CourtListener terminal judgement so retrying GovInfo
            # later does not turn a transport error into semantic uncertainty.
            continue
        else:
            continue
        _write_identity_progression(
            record,
            progression,
            stage=GOVINFO_DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE,
        )
    return replace(document, passes=(*document.passes, GOVINFO_DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE))


async def review_and_requeue_unresolved_docket_roots(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Review unresolved docket extraction once, then requeue a correction once.

    This is the bounded extraction-recovery loop for docket roots. It reviews
    every root that programmatic identity did not resolve, records one complete
    source-grounded citation re-extraction, and re-searches only when that read
    corrects the stated docket number. A successfully reviewed root that still
    does not resolve is left for the later semantic stage; it is never sent to
    this model review again.
    """
    _require_stage(document, DOCKET_ROOT_AMBIGUITY_RESOLUTION_STAGE, "Docket-root extraction review")
    if DOCKET_ROOT_REQUEUED_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_EXTRACTION_REVIEW_STAGE)
    _reject_partial_stage(document, DOCKET_ROOT_REQUEUED_SEARCH_STAGE)

    for record in _docket_roots(document):
        if record.judgement(Question.IDENTITY).outcome == LocatorIdentityResolutionOutcome.RESOLVED.value:
            continue
        if record.extraction_reviewed_by_llm:
            continue
        trigger_node_id = record.judgement(Question.IDENTITY).node_id
        if trigger_node_id is None:
            msg = f"Unresolved docket root {record.citation_id!r} has no identity decision node"
            raise ValueError(msg)
        review = await run_mellea_docket_citation_reextraction(
            record,
            document=document,
            trigger_node_id=trigger_node_id,
            session=session,
        )
        trace_node = _trace_node(
            review,
            stage=DOCKET_ROOT_EXTRACTION_REVIEW_STAGE,
            reads=Reads.DOCUMENT,
        )
        if review.status is ValidationNodeStatus.SUCCEEDED:
            record.mark_extraction_reviewed_by_llm()
        if review.outcome is MelleaDocketCitationReextractionOutcome.CORRECTED:
            if review.grounded_docket_number is None:
                msg = "A corrected docket-citation re-extraction requires a grounded source docket number"
                raise ValueError(msg)
            record.correct(
                trace_node,
                "docket_number",
                review.grounded_docket_number,
                reason=review.reason or review.outcome_message or "Model re-read the source docket citation.",
            )
        else:
            record.observe(trace_node)
        record.judge(
            trace_node,
            Question.EXTRACTION_REVIEW,
            review.outcome.value,
            message=review.outcome_message or review.reason,
        )
        if review.outcome is not MelleaDocketCitationReextractionOutcome.CORRECTED:
            record.judge(
                trace_node,
                Question.IDENTITY,
                LocatorIdentityResolutionOutcome.DEFERRED_TO_SEMANTIC_REVIEW.value,
                message="Programmatic docket resolution remained inconclusive after the extraction review.",
            )

    reviewed = replace(document, passes=(*document.passes, DOCKET_ROOT_EXTRACTION_REVIEW_STAGE))
    return await _requeue_reviewed_docket_roots(reviewed, client=client)


async def _requeue_reviewed_docket_roots(
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
    _require_stage(document, DOCKET_ROOT_EXTRACTION_REVIEW_STAGE, "Reviewed docket-root requeue")
    if DOCKET_ROOT_REQUEUED_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_REQUEUED_SEARCH_STAGE)

    service = client if client is not None else CourtListenerClient()
    for record in _docket_roots(document):
        review = _saved_docket_citation_reextraction(record)
        if review is None or review.outcome is not MelleaDocketCitationReextractionOutcome.CORRECTED:
            continue
        search = _run_docket_root_search(
            record,
            service,
            node_id=f"{record.citation_id}:docket_root_requeued_search",
            depends_on=(review.node_id,),
        )
        trace_node = _trace_node(search, stage=DOCKET_ROOT_REQUEUED_SEARCH_STAGE)
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
    return replace(document, passes=(*document.passes, DOCKET_ROOT_REQUEUED_SEARCH_STAGE))


async def validate_unique_requeued_docket_root_identities(document: Document) -> Document:
    """Resolve saved zero- or one-candidate searches after one requeued docket."""
    _require_stage(document, DOCKET_ROOT_REQUEUED_SEARCH_STAGE, "Requeued docket-root unique identity")
    if DOCKET_ROOT_REQUEUED_UNIQUE_IDENTITY_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_REQUEUED_UNIQUE_IDENTITY_STAGE)

    for record in _docket_roots(document):
        if (
            record.judgement(Question.IDENTITY).outcome
            != LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH.value
        ):
            continue
        search = _saved_docket_root_search(record, stage=DOCKET_ROOT_REQUEUED_SEARCH_STAGE)
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
            progression = _review_docket_candidates(record, search=search)
        else:
            continue
        _write_identity_progression(record, progression, stage=DOCKET_ROOT_REQUEUED_UNIQUE_IDENTITY_STAGE)
    return replace(document, passes=(*document.passes, DOCKET_ROOT_REQUEUED_UNIQUE_IDENTITY_STAGE))


async def resolve_requeued_docket_root_ambiguities(document: Document) -> Document:
    """Finish programmatic candidate resolution for one requeued docket search."""
    _require_stage(
        document,
        DOCKET_ROOT_REQUEUED_UNIQUE_IDENTITY_STAGE,
        "Requeued docket-root ambiguity resolution",
    )
    if DOCKET_ROOT_REQUEUED_AMBIGUITY_RESOLUTION_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_REQUEUED_AMBIGUITY_RESOLUTION_STAGE)

    for record in _docket_roots(document):
        if (
            record.judgement(Question.IDENTITY).outcome
            != LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH.value
        ):
            continue
        search = _saved_docket_root_search(record, stage=DOCKET_ROOT_REQUEUED_SEARCH_STAGE)
        if search.outcome is DocketRootSearchOutcome.AMBIGUOUS:
            progression = _review_docket_candidates(record, search=search)
        elif search.outcome is DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT:
            validation = CitationValidation(citation=record, nodes=(search,))
            progression = validation.append(
                _deferred_resolution(
                    validation,
                    depends_on=(search.node_id,),
                    reason=(
                        f"Requeued docket search returned {search.candidate_count} candidates; "
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
                    reason="Requeued docket search failed; no identity decision was admitted.",
                    scope=search.node_id,
                )
            )
        else:
            continue
        _write_identity_progression(
            record,
            progression,
            stage=DOCKET_ROOT_REQUEUED_AMBIGUITY_RESOLUTION_STAGE,
        )
    return replace(document, passes=(*document.passes, DOCKET_ROOT_REQUEUED_AMBIGUITY_RESOLUTION_STAGE))


async def resolve_docket_root_semantics(
    document: Document,
    *,
    session: MelleaSession | None = None,
) -> Document:
    """Resolve bounded, extraction-reviewed docket searches with semantic evidence.

    This is deliberately a separate checkpoint from docket extraction review.
    It never searches again and it never modifies a docket number. It first
    compares two written docket forms only where literal comparison differed.
    A complete record confirms the root programmatically only when every
    available stated field agrees.  A docket-and-court match alone establishes
    that a record exists, but it cannot establish a citation that also asserts
    a different case name or decision date.  Bounded partial matches therefore
    proceed to grounded representative selection.

    Searches with no candidates, a failed response, or at least twenty
    candidates remain unresolved.  Those outcomes need a different retrieval
    route or a later, more finely scoped review; this stage must not turn a
    retrieval limit into a negative citation finding.
    """
    _require_stage(
        document,
        DOCKET_ROOT_REQUEUED_AMBIGUITY_RESOLUTION_STAGE,
        "Docket-root semantic resolution",
    )
    if DOCKET_ROOT_SEMANTIC_RESOLUTION_STAGE in document.passes:
        return document
    _reject_partial_stage(document, DOCKET_ROOT_SEMANTIC_RESOLUTION_STAGE)

    for record in _docket_roots(document):
        if (
            record.judgement(Question.IDENTITY).outcome
            != LocatorIdentityResolutionOutcome.DEFERRED_TO_SEMANTIC_REVIEW.value
        ):
            continue
        search = _latest_docket_root_search(record)
        if search.outcome not in {DocketRootSearchOutcome.FOUND, DocketRootSearchOutcome.AMBIGUOUS}:
            _write_semantic_deferred_resolution(record, search=search)
            continue

        (
            progression,
            eligible_indices,
            semantic_unavailable,
        ) = await _semantic_docket_candidates(
            record,
            search=search,
            document=document,
            session=session,
        )
        summary = _docket_citation_summary(
            progression,
            scope=f"{search.node_id}:semantic",
        )
        progression = progression.append(summary)

        if len(eligible_indices) == 1 and _summary_has_one_confirmed_match(summary):
            resolution = run_locator_identity_resolution(progression, summary=summary)
        elif not eligible_indices:
            if semantic_unavailable:
                resolution = _future_implementation_resolution(
                    progression,
                    depends_on=(summary.node_id,),
                    scope=f"{search.node_id}:semantic",
                    reason="Semantic docket comparison was unavailable or failed for every retrieved candidate.",
                )
            else:
                resolution = _no_match_resolution(
                    progression,
                    depends_on=(summary.node_id,),
                    scope=f"{search.node_id}:semantic",
                    reason="No retrieved docket candidate represents the stated docket number after semantic comparison.",
                )
        else:
            choice = await run_mellea_locator_candidate_choice(
                progression,
                summary=summary,
                document_text=masked_root_context(
                    document,
                    record,
                    before=480,
                    after=240,
                ).as_document_text(document_length=len(document.text)),
                session=session,
                eligible_candidate_indices=eligible_indices,
            )
            progression = progression.append(choice)
            resolution = run_locator_identity_resolution(
                progression,
                summary=summary,
                choice=choice,
            )

        resolution = replace(
            resolution,
            node_id=f"{search.node_id}:semantic:identity_resolution",
        )
        progression = progression.append(resolution)
        _write_identity_progression(record, progression, stage=DOCKET_ROOT_SEMANTIC_RESOLUTION_STAGE)
    return replace(document, passes=(*document.passes, DOCKET_ROOT_SEMANTIC_RESOLUTION_STAGE))


async def _semantic_docket_candidates(
    record: CitationRecord,
    *,
    search: DocketRootSearchNode | GovInfoDocketSearchNode,
    document: Document,
    session: MelleaSession | None,
) -> tuple[CitationValidation, tuple[int, ...], bool]:
    """Evaluate every stored candidate and return semantic docket anchors.

    The returned indices are only a selection boundary.  The later model sees
    the complete summary, including candidates that failed the docket anchor,
    so the serialized trace remains a full account of the retrieved set.
    """
    scope = f"{search.node_id}:semantic"
    validation = CitationValidation(citation=record, nodes=(search,))
    eligible: list[int] = []
    unavailable = False
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
        validation = validation.append(docket)

        equivalence: MelleaDocketNumberEquivalenceNode | None = None
        docket_matches = docket.outcome is FieldCheckOutcome.MATCH
        if docket.outcome is FieldCheckOutcome.MISMATCH:
            equivalence = await run_mellea_docket_number_equivalence_check(
                validation,
                deterministic_check=docket,
                candidate=candidate,
                document=document,
                session=session,
            )
            validation = validation.append(equivalence)
            docket_matches = equivalence.outcome is MelleaDocketNumberEquivalenceOutcome.MATCH
            unavailable = unavailable or equivalence.outcome in {
                MelleaDocketNumberEquivalenceOutcome.UNAVAILABLE,
                MelleaDocketNumberEquivalenceOutcome.FAILED,
            }
        elif docket.outcome is FieldCheckOutcome.UNAVAILABLE:
            unavailable = True

        case_name = run_exact_case_name_check(validation, candidate=candidate)
        year = run_year_check(validation, candidate=candidate)
        court = run_court_check(validation, evidence=candidate)
        validation = validation.append(case_name).append(year).append(court)
        validation = validation.append(
            _semantic_docket_candidate_assessment(
                validation,
                candidate=candidate,
                docket=docket,
                docket_matches=docket_matches,
                equivalence=equivalence,
                case_name=case_name,
                year_outcome=year.outcome,
                court_outcome=court.outcome,
            )
        )
        if (
            docket_matches
            and court.outcome is not FieldCheckOutcome.MISMATCH
            and year.outcome is not FieldCheckOutcome.MISMATCH
        ):
            eligible.append(index)
    return validation, tuple(eligible), unavailable


def _summary_has_one_confirmed_match(summary: LocatorCitationSummaryNode) -> bool:
    """Return whether its one eligible docket candidate also agrees on all fields.

    Candidate eligibility is intentionally broader: it keeps a record that
    has a semantic docket anchor but a nonliteral or absent name available to
    the grounded representative reviewer.  Programmatic admission needs the
    stronger complete-citation match represented by the summary itself.
    """
    return sum(
        candidate.outcome is LocatorCandidateAssessmentOutcome.MATCH
        for candidate in summary.candidates
    ) == 1


def _semantic_docket_candidate_assessment(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
    docket: DocketNumberCheckNode,
    docket_matches: bool,
    equivalence: MelleaDocketNumberEquivalenceNode | None,
    case_name: ExactCaseNameCheckNode,
    year_outcome: FieldCheckOutcome,
    court_outcome: FieldCheckOutcome,
) -> LocatorCandidateAssessmentNode:
    """Project a semantic docket anchor with the unchanged field evidence.

    Semantic docket agreement gives a candidate to the bounded representative
    choice. It never alone admits identity: case name, court, and date still
    appear in the candidate summary and the model can select no match.
    """
    case_outcome = AggregatedFieldOutcome(case_name.outcome.value)
    year = AggregatedFieldOutcome(year_outcome.value)
    court = AggregatedFieldOutcome(court_outcome.value)
    if not docket_matches:
        outcome = LocatorCandidateAssessmentOutcome.MISMATCH
        message = "The retrieved docket number does not semantically represent the stated docket number."
    elif court is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorCandidateAssessmentOutcome.MISMATCH
        message = "The docket forms agree, but the courts conflict."
    elif year is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorCandidateAssessmentOutcome.MISMATCH
        message = "The docket forms agree, but the dates conflict."
    elif case_outcome is not AggregatedFieldOutcome.MATCH:
        outcome = LocatorCandidateAssessmentOutcome.PARTIAL_MATCH
        message = "The docket forms agree, but the case name needs semantic representative selection."
    else:
        outcome = LocatorCandidateAssessmentOutcome.MATCH
        message = "The docket forms and every available stated identity field agree."
    dependencies = [
        docket.node_id,
        case_name.node_id,
        f"{candidate.node_id}:year_check",
        f"{candidate.node_id}:court_check",
    ]
    if equivalence is not None:
        dependencies.append(equivalence.node_id)
    return LocatorCandidateAssessmentNode(
        node_id=f"{candidate.node_id}:semantic_docket_candidate_assessment",
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
        depends_on=tuple(dependencies),
        status_message="Semantic docket candidate assessment completed.",
        outcome_message=message,
    )


def _write_semantic_deferred_resolution(
    record: CitationRecord,
    *,
    search: DocketRootSearchNode | GovInfoDocketSearchNode,
) -> None:
    """Keep an unavailable retrieval path unresolved at the semantic boundary."""
    validation = CitationValidation(citation=record, nodes=(search,))
    resolution = _future_implementation_resolution(
        validation,
        depends_on=(search.node_id,),
        scope=f"{search.node_id}:semantic",
        reason=(
            "Semantic docket selection requires a complete bounded candidate list; "
            f"the saved search ended as {search.outcome.value}."
        ),
    )
    validation = validation.append(
        replace(resolution, node_id=f"{search.node_id}:semantic:identity_resolution")
    )
    _write_identity_progression(record, validation, stage=DOCKET_ROOT_SEMANTIC_RESOLUTION_STAGE)


def _latest_docket_root_search(
    record: CitationRecord,
) -> DocketRootSearchNode | GovInfoDocketSearchNode:
    """Choose the most specific candidate-bearing docket retrieval path.

    A source-grounded correction is the newest stated locator and its
    CourtListener requeue therefore takes precedence.  Otherwise a positive
    GovInfo fallback is the only candidate set available after a
    CourtListener miss, so semantic docket equivalence must inspect it rather
    than returning to that earlier empty CourtListener result.
    """
    if any(node.stage == DOCKET_ROOT_REQUEUED_SEARCH_STAGE for node in record.trace):
        return _saved_docket_root_search(record, stage=DOCKET_ROOT_REQUEUED_SEARCH_STAGE)
    govinfo = _saved_govinfo_docket_root_search(record)
    if govinfo is not None and govinfo.outcome in {
        DocketRootSearchOutcome.FOUND,
        DocketRootSearchOutcome.AMBIGUOUS,
    }:
        return govinfo
    return _saved_docket_root_search(record)


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


def _run_govinfo_docket_root_search(
    record: CitationRecord,
    client: GovInfoServiceClient,
    *,
    depends_on: tuple[str, ...],
) -> GovInfoDocketSearchNode:
    """Look up a CourtListener miss in GovInfo's published-opinion index."""
    citation = record.stated
    if not isinstance(citation, DocketCitation) or not citation.docket_number:
        return _govinfo_search_node(
            record,
            status=ValidationNodeStatus.FAILED,
            outcome=DocketRootSearchOutcome.FAILED,
            docket_number=citation.docket_number if isinstance(citation, DocketCitation) else None,
            query=None,
            depends_on=depends_on,
            status_message="GovInfo docket lookup could not run.",
            outcome_message="The root has no stated docket number.",
            error="Docket root lacks a docket number",
        )
    try:
        result = client.search_uscourts_docket(
            citation.docket_number,
            court_id=citation.court,
            page_size=MAX_DOCKET_CANDIDATE_REVIEW,
        )
        outcome = _search_outcome(result.count)
        candidates = tuple(govinfo_package_candidate(result_item) for result_item in result.results)
        if outcome is not DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT and len(candidates) != result.count:
            return _govinfo_search_node(
                record,
                status=ValidationNodeStatus.FAILED,
                outcome=DocketRootSearchOutcome.FAILED,
                docket_number=citation.docket_number,
                query=result.query,
                depends_on=depends_on,
                candidate_count=result.count,
                candidates=candidates,
                next_offset_mark=result.next_offset_mark,
                status_message="GovInfo docket lookup returned an incomplete result set.",
                outcome_message="Every GovInfo package must be stored before later review.",
                error=f"Search reported {result.count} packages but returned {len(candidates)}",
            )
        return _govinfo_search_node(
            record,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=outcome,
            docket_number=citation.docket_number,
            query=result.query,
            depends_on=depends_on,
            candidate_count=result.count,
            candidates=candidates,
            next_offset_mark=result.next_offset_mark,
            status_message="GovInfo docket lookup completed.",
            outcome_message=_govinfo_search_message(outcome, result.count),
        )
    except Exception as exc:
        return _govinfo_search_node(
            record,
            status=ValidationNodeStatus.FAILED,
            outcome=DocketRootSearchOutcome.FAILED,
            docket_number=citation.docket_number,
            query=None,
            depends_on=depends_on,
            status_message="GovInfo docket lookup failed.",
            outcome_message="No GovInfo package candidates were available for identity review.",
            error=f"{type(exc).__name__}: {exc}",
        )


def _govinfo_search_node(
    record: CitationRecord,
    *,
    status: ValidationNodeStatus,
    outcome: DocketRootSearchOutcome,
    docket_number: str | None,
    query: str | None,
    depends_on: tuple[str, ...],
    candidate_count: int = 0,
    candidates: tuple[Mapping[str, object], ...] = (),
    next_offset_mark: str | None = None,
    status_message: str | None,
    outcome_message: str | None,
    error: str | None = None,
) -> GovInfoDocketSearchNode:
    return GovInfoDocketSearchNode(
        node_id=f"{record.citation_id}:govinfo_docket_root_search",
        status=status,
        outcome=outcome,
        docket_number=docket_number,
        query=query,
        candidate_count=candidate_count,
        candidates=candidates,
        next_offset_mark=next_offset_mark,
        depends_on=depends_on,
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
    )


def _govinfo_search_message(outcome: DocketRootSearchOutcome, count: int) -> str:
    return {
        DocketRootSearchOutcome.NOT_FOUND: "GovInfo USCOURTS lookup returned no packages.",
        DocketRootSearchOutcome.FOUND: "GovInfo USCOURTS lookup returned one package.",
        DocketRootSearchOutcome.AMBIGUOUS: f"GovInfo USCOURTS lookup returned {count} packages.",
        DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT: (
            f"GovInfo USCOURTS lookup returned {count} packages, at or above the review limit."
        ),
    }[outcome]


def _review_docket_candidates(
    record: CitationRecord,
    *,
    search: DocketRootSearchNode | GovInfoDocketSearchNode,
    provenance: CandidateProvenance = CandidateProvenance.DOCKET,
) -> CitationValidation:
    """Apply only exact programmatic evidence to every stored candidate.

    Docket equivalence, ambiguous candidate selection, and semantic correction
    belong to the later semantic stage. This checkpoint admits a root only when
    one retrieved candidate matches all available stated identity fields under
    deterministic comparisons.
    """
    validation = CitationValidation(citation=record, nodes=(search,))
    scope = search.node_id
    for index, result in enumerate(search.candidates, start=1):
        if provenance is CandidateProvenance.GOVINFO:
            candidate = run_govinfo_docket_search_candidate_evaluation(
                validation,
                result=result,
                candidate_index=index,
                depends_on=(search.node_id,),
                node_prefix=scope,
            )
        else:
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

    summary = _docket_citation_summary(validation, scope=scope, provenance=provenance)
    validation = validation.append(summary)
    matching = tuple(
        candidate
        for candidate in summary.candidates
        if candidate.outcome is LocatorCandidateAssessmentOutcome.MATCH
    )
    partial = tuple(
        candidate
        for candidate in summary.candidates
        if candidate.outcome is LocatorCandidateAssessmentOutcome.PARTIAL_MATCH
    )
    if len(matching) == 1:
        resolution = run_locator_identity_resolution(validation, summary=summary)
    elif not matching:
        if partial:
            resolution = _deferred_resolution(
                validation,
                depends_on=(summary.node_id,),
                reason=(
                    "At least one retrieved docket candidate has incomplete programmatic identity "
                    "evidence and requires extraction or semantic review."
                ),
                scope=scope,
            )
        else:
            resolution = _no_match_resolution(
                validation,
                depends_on=(summary.node_id,),
                reason="No retrieved docket candidate passed every programmatic identity comparison.",
                scope=scope,
            )
    else:
        resolution = _deferred_resolution(
            validation,
            depends_on=(summary.node_id,),
            reason="More than one retrieved docket candidate passed programmatic identity comparison.",
            scope=scope,
        )
    return validation.append(replace(resolution, node_id=f"{scope}:identity_resolution"))


def _review_govinfo_docket_candidates(
    record: CitationRecord,
    *,
    search: GovInfoDocketSearchNode,
) -> CitationValidation:
    """Evaluate GovInfo package candidates through the shared docket checks."""
    return _review_docket_candidates(record, search=search, provenance=CandidateProvenance.GOVINFO)


def _docket_candidate_assessment(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
    docket: DocketNumberCheckNode,
    case_name: ExactCaseNameCheckNode,
    year_outcome: FieldCheckOutcome,
    court_outcome: FieldCheckOutcome,
) -> LocatorCandidateAssessmentNode:
    """Record exact docket-citation evidence without semantic repair.

    An unavailable court does not block a citation when the filing did not
    state a court. A deterministic court or date contradiction is a negative
    identity finding, rather than a reason to override the filing. A missing
    case name remains incomplete evidence for later semantic review.
    """
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
    elif court is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorCandidateAssessmentOutcome.MISMATCH
        message = "The docket number matches, but the courts conflict."
    elif year is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorCandidateAssessmentOutcome.MISMATCH
        message = "The docket number matches, but the dates conflict."
    elif case_outcome is not AggregatedFieldOutcome.MATCH:
        outcome = LocatorCandidateAssessmentOutcome.PARTIAL_MATCH
        message = "The docket number matches, but the case name is missing or conflicts."
    else:
        outcome = LocatorCandidateAssessmentOutcome.MATCH
        message = "The docket number and every available stated identity field match."
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
    provenance: CandidateProvenance = CandidateProvenance.DOCKET,
) -> LocatorCitationSummaryNode:
    assessments = tuple(node for node in validation.nodes if isinstance(node, LocatorCandidateAssessmentNode))
    if not assessments:
        msg = "Docket candidate summary requires at least one candidate assessment"
        raise ValueError(msg)
    candidates = tuple(
        citation_summary_candidate(validation, assessment, provenance=provenance)
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
        outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_SEMANTIC_REVIEW,
        selected_candidate_index=None,
        selected_assessment_node_id=None,
        matching_candidate_indices=(),
        selection_evidence_node_id=None,
        depends_on=depends_on,
        status_message="Docket-root identity resolution deferred to semantic review.",
        outcome_message=reason,
    )


def _future_implementation_resolution(
    validation: CitationValidation,
    *,
    depends_on: tuple[str, ...],
    reason: str,
    scope: str | None = None,
) -> LocatorIdentityResolutionNode:
    """Finish this route without recasting unavailable retrieval as no match."""
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
    projected = {
        node.node_id: _trace_node(
            node,
            stage=stage,
            reads=(
                Reads.DOCUMENT
                if isinstance(node, (MelleaDocketNumberEquivalenceNode, MelleaLocatorCandidateChoiceNode))
                else Reads.RECORD
            ),
        )
        for node in progression.nodes
    }
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
                govinfo_package_id=candidate.govinfo_package_id,
            ),
        )
        if candidate.docket_id is not None:
            # This is the resolved authority pointer.  It deliberately names
            # the docket resource rather than pretending CourtListener's docket
            # id is an opinion-cluster id.
            record.reattribute(resolution_node, f"courtlistener:docket:{candidate.docket_id}")
        elif candidate.govinfo_package_id is not None:
            record.reattribute(resolution_node, f"govinfo:package:{candidate.govinfo_package_id}")
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
            and node.source
            in {
                CandidateEvaluationSource.DOCKET_SEARCH,
                CandidateEvaluationSource.GOVINFO_DOCKET_SEARCH,
            }
            and node.candidate_index == resolution.selected_candidate_index
        ),
        None,
    )
    if candidate is None:
        msg = "A resolved docket identity must select one stored docket-search or GovInfo candidate"
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


def _saved_govinfo_docket_root_search(record: CitationRecord) -> GovInfoDocketSearchNode | None:
    """Return the persisted GovInfo fallback lookup when this root needed one."""
    matches = [
        node
        for node in record.trace
        if node.stage == GOVINFO_DOCKET_ROOT_SEARCH_STAGE
        and node.details.get("validation_node_type") == GovInfoDocketSearchNode.__name__
    ]
    if not matches:
        return None
    if len(matches) != 1:
        msg = f"Expected at most one saved GovInfo lookup for {record.citation_id!r}, found {len(matches)}"
        raise ValueError(msg)
    raw = matches[0].details.get("validation")
    if not isinstance(raw, dict):
        msg = f"Saved GovInfo lookup for {record.citation_id!r} has no validation payload"
        raise ValueError(msg)
    node = deserialize_validation_node({"node_type": GovInfoDocketSearchNode.__name__, **raw})
    if not isinstance(node, GovInfoDocketSearchNode):
        msg = f"Saved GovInfo lookup for {record.citation_id!r} decoded as {type(node).__name__}"
        raise ValueError(msg)
    return node


def _saved_docket_citation_reextraction(
    record: CitationRecord,
) -> MelleaDocketCitationReextractionNode | None:
    """Return the one persisted docket-citation re-extraction, if this root needed one."""
    matches = [
        node
        for node in record.trace
        if node.stage == DOCKET_ROOT_EXTRACTION_REVIEW_STAGE
        and node.details.get("validation_node_type") == MelleaDocketCitationReextractionNode.__name__
    ]
    if not matches:
        return None
    if len(matches) != 1:
        msg = f"Expected at most one docket-citation re-extraction for {record.citation_id!r}, found {len(matches)}"
        raise ValueError(msg)
    raw = matches[0].details.get("validation")
    if not isinstance(raw, dict):
        msg = f"Saved docket-citation re-extraction for {record.citation_id!r} has no validation payload"
        raise ValueError(msg)
    node = deserialize_validation_node({"node_type": MelleaDocketCitationReextractionNode.__name__, **raw})
    if not isinstance(node, MelleaDocketCitationReextractionNode):
        msg = (
            f"Saved docket-citation re-extraction for {record.citation_id!r} decoded as {type(node).__name__}"
        )
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


def _govinfo_docket_lookup_outcome(outcome: DocketRootSearchOutcome) -> str:
    """Keep the active lookup judgement explicit about the fallback source."""
    return {
        DocketRootSearchOutcome.FOUND: "govinfo_found",
        DocketRootSearchOutcome.NOT_FOUND: "govinfo_not_found",
        DocketRootSearchOutcome.AMBIGUOUS: "govinfo_deferred_to_ambiguity",
        DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT: "govinfo_deferred_to_future_implementation",
        DocketRootSearchOutcome.FAILED: "govinfo_failed",
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
