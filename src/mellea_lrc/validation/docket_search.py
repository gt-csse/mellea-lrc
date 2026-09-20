"""Bounded CourtListener and GovInfo metadata discovery for unresolved docket roots.

The preceding docket-root path performs literal identifier lookup and bounded
identity review.  These two stages preserve an independent, general metadata
search family after that path remains unresolved.  They retain every query and
candidate, but make no identity decision.  A later body/opinion corroboration
route has different evidence semantics and is intentionally separate.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.core.record import UNJUDGED, Node, Question, Reads
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.extraction.stages import CASE_NAME_STAGE
from mellea_lrc.govinfo import GovInfoClient, govinfo_uscourts_case_name_query, govinfo_uscourts_docket_query
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.serialization.validated_document import deserialize_validation_node
from mellea_lrc.validation.docket_roots import DOCKET_ROOT_SEMANTIC_RESOLUTION_STAGE
from mellea_lrc.validation.metadata_search_common import (
    MAX_CANDIDATES_FOR_LATER_REVIEW,
    MetadataQuery,
    MetadataTermPlan,
    courtlistener_case_name_query,
    deduplicate_queries,
    merge_metadata_candidates,
    prepare_case_name_terms,
    run_courtlistener_metadata_attempt,
    run_govinfo_metadata_attempt,
    saved_case_name_terms,
)
from mellea_lrc.validation.types import (
    DocketRootSearchNode,
    DocketRootSearchOutcome,
    GovInfoDocketSearchNode,
    MelleaDocketCitationReextractionNode,
    MetadataSearchAttempt,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea import MelleaSession

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import Document


COURTLISTENER_DOCKET_SEARCH_STAGE = "courtlistener_docket_search"
GOVINFO_DOCKET_SEARCH_STAGE = "govinfo_docket_search"
_MADE_BY = "mellea_lrc.validation.docket_search"

# Local aliases keep the stage's test seam narrow.  The actual mechanics live
# in metadata_search_common so reporter and docket stages share one contract.
_TermPlan = MetadataTermPlan
_QuerySpec = MetadataQuery


async def search_courtlistener_docket_roots(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Discover CourtListener docket metadata through bounded query variants.

    Query order is fixed and general: literal docket; a source-grounded full
    case name within the stated court, if one was read; the same name without
    a court; and court-bounded individual terms only if the full court-bounded
    name query returned no result.  No docket normalization or body-text query
    is attempted here.
    """
    _require_stage(document, DOCKET_ROOT_SEMANTIC_RESOLUTION_STAGE, "CourtListener docket search")
    _require_stage(document, CASE_NAME_STAGE, "CourtListener docket search")
    if COURTLISTENER_DOCKET_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document, COURTLISTENER_DOCKET_SEARCH_STAGE)

    service = client if client is not None else CourtListenerClient()
    for record in _unresolved_docket_roots(document):
        plan = await _prepare_terms(record, stage=COURTLISTENER_DOCKET_SEARCH_STAGE, session=session)
        record.observe(plan.node)
        attempts = _courtlistener_attempts(record, terms=plan.terms, client=service)
        node = _courtlistener_node(record, plan=plan, attempts=attempts)
        trace = _trace_node(node, stage=COURTLISTENER_DOCKET_SEARCH_STAGE)
        record.judge(
            trace, Question.DOCKET_LOOKUP, _lookup_outcome(node.outcome), message=node.outcome_message
        )
    return replace(document, passes=(*document.passes, COURTLISTENER_DOCKET_SEARCH_STAGE))


async def search_govinfo_docket_roots(
    document: Document,
    *,
    client: GovInfoClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Discover GovInfo USCOURTS package metadata through the same query family.

    GovInfo runs even when CourtListener found candidates because they are
    independent archives.  It reuses the recorded CourtListener term plan so
    the source is not needlessly sent to the model twice.
    """
    _require_stage(document, COURTLISTENER_DOCKET_SEARCH_STAGE, "GovInfo docket search")
    _require_stage(document, CASE_NAME_STAGE, "GovInfo docket search")
    if GOVINFO_DOCKET_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document, GOVINFO_DOCKET_SEARCH_STAGE)

    service = client if client is not None else GovInfoClient()
    for record in _unresolved_docket_roots(document):
        plan = _saved_term_plan(record, stage=COURTLISTENER_DOCKET_SEARCH_STAGE)
        if plan is None:
            plan = await _prepare_terms(record, stage=GOVINFO_DOCKET_SEARCH_STAGE, session=session)
            record.observe(plan.node)
        attempts = _govinfo_attempts(record, terms=plan.terms, client=service)
        node = _govinfo_node(record, plan=plan, attempts=attempts)
        trace = _trace_node(node, stage=GOVINFO_DOCKET_SEARCH_STAGE)
        record.judge(
            trace, Question.DOCKET_LOOKUP, _lookup_outcome(node.outcome), message=node.outcome_message
        )
    return replace(document, passes=(*document.passes, GOVINFO_DOCKET_SEARCH_STAGE))


async def _prepare_terms(
    record: CitationRecord,
    *,
    stage: str,
    session: MelleaSession | None,
) -> MetadataTermPlan:
    return await prepare_case_name_terms(
        record,
        stage=stage,
        made_by=_MADE_BY,
        source_case_name=_source_case_name(record),
        session=session,
    )


def _saved_term_plan(record: CitationRecord, *, stage: str) -> MetadataTermPlan | None:
    return saved_case_name_terms(record, stage=stage)


def _source_case_name(record: CitationRecord) -> str | None:
    citation = _docket_citation(record)
    if citation.case_name is not None and citation.case_name.text.strip():
        return citation.case_name.text
    review = _saved_reextraction(record)
    if review is not None and review.reparsed_case_name and review.reparsed_case_name.strip():
        return review.reparsed_case_name
    parties = tuple(
        part.strip() for part in (citation.plaintiff, citation.defendant) if part and part.strip()
    )
    if len(parties) == 2:
        return f"{parties[0]} v. {parties[1]}"
    return parties[0] if parties else None


def _saved_reextraction(record: CitationRecord) -> MelleaDocketCitationReextractionNode | None:
    matches = [
        node
        for node in record.trace
        if node.details.get("validation_node_type") == MelleaDocketCitationReextractionNode.__name__
    ]
    if not matches:
        return None
    if len(matches) != 1:
        msg = f"Expected at most one saved docket-citation re-extraction for {record.citation_id!r}"
        raise ValueError(msg)
    raw = matches[0].details.get("validation")
    if not isinstance(raw, dict):
        msg = f"Saved docket-citation re-extraction for {record.citation_id!r} has no validation payload"
        raise ValueError(msg)
    decoded = deserialize_validation_node({"node_type": MelleaDocketCitationReextractionNode.__name__, **raw})
    if not isinstance(decoded, MelleaDocketCitationReextractionNode):
        msg = f"Unexpected saved re-extraction type: {type(decoded).__name__}"
        raise ValueError(msg)
    return decoded


def _courtlistener_attempts(
    record: CitationRecord,
    *,
    terms: tuple[str, ...],
    client: CourtListenerServiceClient,
) -> tuple[MetadataSearchAttempt, ...]:
    attempts = [
        run_courtlistener_metadata_attempt(spec, client) for spec in _courtlistener_queries(record, terms)
    ]
    attempts.extend(
        run_courtlistener_metadata_attempt(spec, client)
        for spec in _single_term_court_queries(record, terms, attempts, provider="courtlistener")
    )
    return tuple(attempts)


def _govinfo_attempts(
    record: CitationRecord,
    *,
    terms: tuple[str, ...],
    client: GovInfoClient,
) -> tuple[MetadataSearchAttempt, ...]:
    attempts = [run_govinfo_metadata_attempt(spec, client) for spec in _govinfo_queries(record, terms)]
    attempts.extend(
        run_govinfo_metadata_attempt(spec, client)
        for spec in _single_term_court_queries(record, terms, attempts, provider="govinfo")
    )
    return tuple(attempts)


def _single_term_court_queries(
    record: CitationRecord,
    terms: tuple[str, ...],
    attempts: list[MetadataSearchAttempt],
    *,
    provider: Literal["courtlistener", "govinfo"],
) -> tuple[MetadataQuery, ...]:
    """Relax a zero-result name conjunction without removing its court boundary."""
    citation = _docket_citation(record)
    if citation.court is None or len(terms) < 2:
        return ()
    full_attempt = next(
        (attempt for attempt in attempts if attempt.kind == "case_name_with_stated_court"), None
    )
    if (
        full_attempt is None
        or full_attempt.status is not ValidationNodeStatus.SUCCEEDED
        or full_attempt.candidate_count != 0
    ):
        return ()
    if provider == "courtlistener":
        return tuple(
            MetadataQuery(
                "case_name_term_with_stated_court",
                courtlistener_case_name_query((term,), citation.court),
                citation.court,
            )
            for term in terms
        )
    return tuple(
        MetadataQuery(
            "case_name_term_with_stated_court",
            govinfo_uscourts_case_name_query((term,), court_id=citation.court),
            citation.court,
        )
        for term in terms
    )


def _courtlistener_queries(record: CitationRecord, terms: tuple[str, ...]) -> tuple[MetadataQuery, ...]:
    citation = _docket_citation(record)
    if citation.docket_number is None:
        return ()
    queries = [MetadataQuery("literal_docket", citation.docket_number, None)]
    if terms:
        if citation.court:
            queries.append(
                MetadataQuery(
                    "case_name_with_stated_court",
                    courtlistener_case_name_query(terms, citation.court),
                    citation.court,
                )
            )
        queries.append(MetadataQuery("case_name", courtlistener_case_name_query(terms, None), None))
    return deduplicate_queries(queries)


def _govinfo_queries(record: CitationRecord, terms: tuple[str, ...]) -> tuple[MetadataQuery, ...]:
    citation = _docket_citation(record)
    if citation.docket_number is None:
        return ()
    queries = [
        MetadataQuery(
            "literal_docket",
            govinfo_uscourts_docket_query(citation.docket_number, court_id=None),
            None,
        )
    ]
    if terms:
        if citation.court:
            queries.append(
                MetadataQuery(
                    "case_name_with_stated_court",
                    govinfo_uscourts_case_name_query(terms, court_id=citation.court),
                    citation.court,
                )
            )
        queries.append(
            MetadataQuery("case_name", govinfo_uscourts_case_name_query(terms, court_id=None), None)
        )
    return deduplicate_queries(queries)


def _courtlistener_node(
    record: CitationRecord,
    *,
    plan: MetadataTermPlan,
    attempts: tuple[MetadataSearchAttempt, ...],
) -> DocketRootSearchNode:
    candidates = merge_metadata_candidates(attempts, key="docket_id")
    status, outcome = _aggregate_outcome(attempts, candidates)
    return DocketRootSearchNode(
        node_id=f"{record.citation_id}:courtlistener_docket_search",
        status=status,
        outcome=outcome,
        docket_number=_docket_citation(record).docket_number,
        query=None,
        candidate_count=len(candidates),
        candidates=candidates,
        next_cursor=None,
        depends_on=(plan.node.node_id,),
        status_message="CourtListener docket metadata discovery completed.",
        outcome_message=_outcome_message("CourtListener", outcome, len(candidates), attempts),
        error=_aggregate_error(attempts),
        attempts=attempts,
    )


def _govinfo_node(
    record: CitationRecord,
    *,
    plan: MetadataTermPlan,
    attempts: tuple[MetadataSearchAttempt, ...],
) -> GovInfoDocketSearchNode:
    candidates = merge_metadata_candidates(attempts, key="govinfo_package_id")
    status, outcome = _aggregate_outcome(attempts, candidates)
    return GovInfoDocketSearchNode(
        node_id=f"{record.citation_id}:govinfo_docket_search",
        status=status,
        outcome=outcome,
        docket_number=_docket_citation(record).docket_number,
        query=None,
        candidate_count=len(candidates),
        candidates=candidates,
        next_offset_mark=None,
        depends_on=(plan.node.node_id,),
        status_message="GovInfo USCOURTS docket metadata discovery completed.",
        outcome_message=_outcome_message("GovInfo", outcome, len(candidates), attempts),
        error=_aggregate_error(attempts),
        attempts=attempts,
    )


def _aggregate_outcome(
    attempts: tuple[MetadataSearchAttempt, ...],
    candidates: tuple[Mapping[str, object], ...],
) -> tuple[ValidationNodeStatus, DocketRootSearchOutcome]:
    if not attempts or all(attempt.status is ValidationNodeStatus.FAILED for attempt in attempts):
        return ValidationNodeStatus.FAILED, DocketRootSearchOutcome.FAILED
    if any(
        attempt.status is ValidationNodeStatus.SUCCEEDED
        and attempt.candidate_count is not None
        and attempt.candidate_count >= MAX_CANDIDATES_FOR_LATER_REVIEW
        for attempt in attempts
    ):
        return ValidationNodeStatus.SUCCEEDED, DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT
    if not candidates:
        return ValidationNodeStatus.SUCCEEDED, DocketRootSearchOutcome.NOT_FOUND
    if len(candidates) == 1:
        return ValidationNodeStatus.SUCCEEDED, DocketRootSearchOutcome.FOUND
    return ValidationNodeStatus.SUCCEEDED, DocketRootSearchOutcome.AMBIGUOUS


def _outcome_message(
    provider: str,
    outcome: DocketRootSearchOutcome,
    count: int,
    attempts: tuple[MetadataSearchAttempt, ...],
) -> str:
    completed = sum(attempt.status is ValidationNodeStatus.SUCCEEDED for attempt in attempts)
    return (
        f"{provider} metadata discovery ran {completed}/{len(attempts)} bounded query attempts, "
        f"retaining {count} distinct candidate record(s); outcome={outcome.value}."
    )


def _aggregate_error(attempts: tuple[MetadataSearchAttempt, ...]) -> str | None:
    errors = tuple(attempt.error for attempt in attempts if attempt.error)
    return "; ".join(errors) if errors else None


def _lookup_outcome(outcome: DocketRootSearchOutcome) -> str:
    return {
        DocketRootSearchOutcome.FOUND: "search_found",
        DocketRootSearchOutcome.NOT_FOUND: "search_not_found",
        DocketRootSearchOutcome.AMBIGUOUS: "search_candidates_found",
        DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT: "search_exceeds_candidate_limit",
        DocketRootSearchOutcome.FAILED: "search_failed",
    }[outcome]


def _docket_citation(record: CitationRecord) -> DocketCitation:
    if not isinstance(record.stated, DocketCitation):
        msg = f"Expected a docket root, got {type(record.stated).__name__}"
        raise TypeError(msg)
    return record.stated


def _unresolved_docket_roots(document: Document) -> tuple[CitationRecord, ...]:
    return tuple(
        record
        for record in document.active_citations
        if record.is_root
        and isinstance(record.stated, DocketCitation)
        and record.judgement(Question.IDENTITY).outcome != "resolved"
        and record.judgement(Question.IDENTITY).outcome != UNJUDGED
    )


def _require_stage(document: Document, required_stage: str, stage_name: str) -> None:
    if required_stage not in document.passes:
        msg = f"{stage_name} requires {required_stage} before validation."
        raise ValueError(msg)


def _reject_partial_stage(document: Document, stage: str) -> None:
    if any(node.stage == stage for record in document.citations for node in record.trace):
        msg = f"Cannot resume partial {stage}; restart from the document before this stage."
        raise ValueError(msg)


def _trace_node(node: DocketRootSearchNode | GovInfoDocketSearchNode, *, stage: str) -> Node:
    payload = serialize_dataclass(node)
    return Node(
        node_id=node.node_id,
        reads=Reads.RECORD,
        stage=stage,
        made_by=_MADE_BY,
        outcome=str(payload["outcome"]),
        message=node.outcome_message,
        depends_on=node.depends_on,
        details={"validation_node_type": type(node).__name__, "validation": payload},
    )
