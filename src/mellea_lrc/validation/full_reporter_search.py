"""Bounded metadata discovery after an exact reporter-citation lookup misses.

An exact reporter lookup remains the authoritative first route for a complete
reporter locator.  This module is deliberately narrower than a general
full-text search: it searches provider *case metadata* using only case-name
text explicitly written in the citation.  A later opinion/filing-body
corroboration route has different evidence semantics and must remain a
separate stage.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.record import UNJUDGED, Node, Question, Reads
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.extraction.stages import CASE_NAME_STAGE
from mellea_lrc.govinfo import GovInfoClient, govinfo_uscourts_case_name_query
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.validation.metadata_search_common import (
    MAX_CANDIDATES_FOR_LATER_REVIEW,
    MetadataQuery,
    MetadataTermPlan,
    courtlistener_case_name_query,
    merge_metadata_candidates,
    prepare_case_name_terms,
    run_courtlistener_metadata_attempt,
    run_govinfo_metadata_attempt,
    saved_case_name_terms,
)
from mellea_lrc.validation.roots import EXACT_LOOKUP_STAGE, _saved_exact_lookup
from mellea_lrc.validation.types import (
    FullReporterSearchNode,
    FullReporterSearchOutcome,
    GovInfoFullReporterSearchNode,
    LocatorLookupOutcome,
    MetadataSearchAttempt,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import Document


COURTLISTENER_FULL_REPORTER_SEARCH_STAGE = "courtlistener_full_reporter_metadata_search"
GOVINFO_FULL_REPORTER_SEARCH_STAGE = "govinfo_full_reporter_metadata_search"
_MADE_BY = "mellea_lrc.validation.full_reporter_search"


async def search_courtlistener_full_reporter_roots(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Discover CourtListener docket metadata for exact-lookup reporter misses.

    The provider is queried only through the docket metadata corpus.  The
    literal reporter string is intentionally absent here: finding that string
    in an opinion body means a document cited the case, not that the returned
    opinion is the case.  Body-text corroboration is a distinct later route.
    """
    _require_stage(document, EXACT_LOOKUP_STAGE, "CourtListener full-reporter metadata search")
    _require_stage(document, CASE_NAME_STAGE, "CourtListener full-reporter metadata search")
    if COURTLISTENER_FULL_REPORTER_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document, COURTLISTENER_FULL_REPORTER_SEARCH_STAGE)

    service = client if client is not None else CourtListenerClient()
    for record, lookup in _eligible_roots(document):
        plan = await _prepare_terms(record, stage=COURTLISTENER_FULL_REPORTER_SEARCH_STAGE, session=session)
        record.observe(plan.node)
        attempts = _courtlistener_attempts(record, plan.terms, service)
        node = _courtlistener_node(record, lookup.node_id, plan, attempts)
        _write_search_result(record, node, stage=COURTLISTENER_FULL_REPORTER_SEARCH_STAGE)
    return replace(document, passes=(*document.passes, COURTLISTENER_FULL_REPORTER_SEARCH_STAGE))


async def search_govinfo_full_reporter_roots(
    document: Document,
    *,
    client: GovInfoClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Discover GovInfo USCOURTS package metadata for the same reporter misses."""
    _require_stage(
        document, COURTLISTENER_FULL_REPORTER_SEARCH_STAGE, "GovInfo full-reporter metadata search"
    )
    _require_stage(document, CASE_NAME_STAGE, "GovInfo full-reporter metadata search")
    if GOVINFO_FULL_REPORTER_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document, GOVINFO_FULL_REPORTER_SEARCH_STAGE)

    service = client if client is not None else GovInfoClient()
    for record, lookup in _eligible_roots(document):
        plan = saved_case_name_terms(record, stage=COURTLISTENER_FULL_REPORTER_SEARCH_STAGE)
        if plan is None:
            plan = await _prepare_terms(record, stage=GOVINFO_FULL_REPORTER_SEARCH_STAGE, session=session)
            record.observe(plan.node)
        attempts = _govinfo_attempts(record, plan.terms, service)
        node = _govinfo_node(record, lookup.node_id, plan, attempts)
        _write_search_result(record, node, stage=GOVINFO_FULL_REPORTER_SEARCH_STAGE)
    return replace(document, passes=(*document.passes, GOVINFO_FULL_REPORTER_SEARCH_STAGE))


def _eligible_roots(document: Document) -> tuple[tuple[CitationRecord, object], ...]:
    eligible: list[tuple[CitationRecord, object]] = []
    for record in document.active_citations:
        if not record.is_root or not isinstance(record.stated, FullCaseCitation):
            continue
        if record.judgement(Question.IDENTITY).outcome != UNJUDGED:
            continue
        lookup = _saved_exact_lookup(record)
        if lookup.outcome is LocatorLookupOutcome.NOT_FOUND:
            eligible.append((record, lookup))
    return tuple(eligible)


async def _prepare_terms(
    record: CitationRecord,
    *,
    stage: str,
    session: MelleaSession | None,
) -> MetadataTermPlan:
    """Prepare the same grounded name evidence used by docket discovery."""
    citation = _citation(record)
    source_name = (
        citation.case_name.text
        if citation.case_name is not None and citation.case_name.text.strip()
        else None
    )
    if source_name is None:
        parties = tuple(
            value.strip() for value in (citation.plaintiff, citation.defendant) if value and value.strip()
        )
        source_name = (
            f"{parties[0]} v. {parties[1]}" if len(parties) == 2 else (parties[0] if parties else None)
        )
    if source_name is None and citation.antecedent and citation.antecedent.strip():
        source_name = citation.antecedent
    return await prepare_case_name_terms(
        record,
        stage=stage,
        made_by=_MADE_BY,
        source_case_name=source_name,
        session=session,
    )


def _courtlistener_attempts(
    record: CitationRecord,
    terms: tuple[str, ...],
    client: CourtListenerServiceClient,
) -> tuple[MetadataSearchAttempt, ...]:
    citation = _citation(record)
    attempts: list[MetadataSearchAttempt] = []
    if terms:
        if citation.court:
            attempts.append(
                run_courtlistener_metadata_attempt(
                    MetadataQuery(
                        "case_name_with_stated_court",
                        courtlistener_case_name_query(terms, citation.court),
                        citation.court,
                    ),
                    client,
                )
            )
        attempts.append(
            run_courtlistener_metadata_attempt(
                MetadataQuery("case_name", courtlistener_case_name_query(terms, None), None), client
            )
        )
        attempts.extend(
            run_courtlistener_metadata_attempt(query, client)
            for query in _single_term_court_queries(citation, terms, attempts)
        )
    return tuple(attempts)


def _govinfo_attempts(
    record: CitationRecord,
    terms: tuple[str, ...],
    client: GovInfoClient,
) -> tuple[MetadataSearchAttempt, ...]:
    citation = _citation(record)
    attempts: list[MetadataSearchAttempt] = []
    if terms:
        if citation.court:
            attempts.append(
                run_govinfo_metadata_attempt(
                    MetadataQuery(
                        "case_name_with_stated_court",
                        govinfo_uscourts_case_name_query(terms, court_id=citation.court),
                        citation.court,
                    ),
                    client,
                )
            )
        attempts.append(
            run_govinfo_metadata_attempt(
                MetadataQuery("case_name", govinfo_uscourts_case_name_query(terms, court_id=None), None),
                client,
            )
        )
        attempts.extend(
            run_govinfo_metadata_attempt(query, client)
            for query in _single_term_court_queries(citation, terms, attempts, govinfo=True)
        )
    return tuple(attempts)


def _single_term_court_queries(
    citation: FullCaseCitation,
    terms: tuple[str, ...],
    attempts: list[MetadataSearchAttempt],
    *,
    govinfo: bool = False,
) -> tuple[MetadataQuery, ...]:
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
    if govinfo:
        return tuple(
            MetadataQuery(
                "case_name_term_with_stated_court",
                govinfo_uscourts_case_name_query((term,), court_id=citation.court),
                citation.court,
            )
            for term in terms
        )
    return tuple(
        MetadataQuery(
            "case_name_term_with_stated_court",
            courtlistener_case_name_query((term,), citation.court),
            citation.court,
        )
        for term in terms
    )


def _courtlistener_node(
    record: CitationRecord,
    lookup_node_id: str,
    plan: MetadataTermPlan,
    attempts: tuple[MetadataSearchAttempt, ...],
) -> FullReporterSearchNode:
    candidates = merge_metadata_candidates(attempts, key="docket_id")
    status, outcome = _outcome(attempts, candidates, plan)
    return FullReporterSearchNode(
        node_id=f"{record.citation_id}:courtlistener_full_reporter_metadata_search",
        status=status,
        outcome=outcome,
        reporter_locator=_reporter_locator(record),
        candidate_count=len(candidates),
        candidates=candidates,
        depends_on=(lookup_node_id, plan.node.node_id),
        status_message="CourtListener full-reporter metadata discovery completed.",
        outcome_message=_message("CourtListener", outcome, attempts, len(candidates)),
        error=_errors(attempts),
        attempts=attempts,
    )


def _govinfo_node(
    record: CitationRecord,
    lookup_node_id: str,
    plan: MetadataTermPlan,
    attempts: tuple[MetadataSearchAttempt, ...],
) -> GovInfoFullReporterSearchNode:
    candidates = merge_metadata_candidates(attempts, key="govinfo_package_id")
    status, outcome = _outcome(attempts, candidates, plan)
    return GovInfoFullReporterSearchNode(
        node_id=f"{record.citation_id}:govinfo_full_reporter_metadata_search",
        status=status,
        outcome=outcome,
        reporter_locator=_reporter_locator(record),
        candidate_count=len(candidates),
        candidates=candidates,
        depends_on=(lookup_node_id, plan.node.node_id),
        status_message="GovInfo full-reporter metadata discovery completed.",
        outcome_message=_message("GovInfo", outcome, attempts, len(candidates)),
        error=_errors(attempts),
        attempts=attempts,
    )


def _outcome(
    attempts: tuple[MetadataSearchAttempt, ...],
    candidates: tuple[object, ...],
    plan: MetadataTermPlan,
) -> tuple[ValidationNodeStatus, FullReporterSearchOutcome]:
    if not attempts:
        if plan.node.outcome == "failed":
            return ValidationNodeStatus.FAILED, FullReporterSearchOutcome.FAILED
        return ValidationNodeStatus.SUCCEEDED, FullReporterSearchOutcome.UNAVAILABLE
    if not attempts or all(attempt.status is ValidationNodeStatus.FAILED for attempt in attempts):
        return ValidationNodeStatus.FAILED, FullReporterSearchOutcome.FAILED
    if any(
        attempt.status is ValidationNodeStatus.SUCCEEDED
        and attempt.candidate_count is not None
        and attempt.candidate_count >= MAX_CANDIDATES_FOR_LATER_REVIEW
        for attempt in attempts
    ):
        return ValidationNodeStatus.SUCCEEDED, FullReporterSearchOutcome.EXCEEDS_REVIEW_LIMIT
    if not candidates:
        return ValidationNodeStatus.SUCCEEDED, FullReporterSearchOutcome.NOT_FOUND
    if len(candidates) == 1:
        return ValidationNodeStatus.SUCCEEDED, FullReporterSearchOutcome.FOUND
    return ValidationNodeStatus.SUCCEEDED, FullReporterSearchOutcome.AMBIGUOUS


def _write_search_result(
    record: CitationRecord,
    node: FullReporterSearchNode | GovInfoFullReporterSearchNode,
    *,
    stage: str,
) -> None:
    trace = Node(
        node_id=node.node_id,
        reads=Reads.RECORD,
        stage=stage,
        made_by=_MADE_BY,
        outcome=node.outcome.value,
        message=node.outcome_message,
        depends_on=node.depends_on,
        details={"validation_node_type": type(node).__name__, "validation": serialize_dataclass(node)},
    )
    record.observe(trace)
    record.judge(trace, Question.LOCATOR_LOOKUP, _lookup_outcome(node.outcome), message=node.outcome_message)


def _lookup_outcome(outcome: FullReporterSearchOutcome) -> str:
    return {
        FullReporterSearchOutcome.FOUND: "search_found",
        FullReporterSearchOutcome.NOT_FOUND: "search_not_found",
        FullReporterSearchOutcome.AMBIGUOUS: "search_candidates_found",
        FullReporterSearchOutcome.EXCEEDS_REVIEW_LIMIT: "search_exceeds_candidate_limit",
        FullReporterSearchOutcome.UNAVAILABLE: "search_unavailable",
        FullReporterSearchOutcome.FAILED: "search_failed",
    }[outcome]


def _citation(record: CitationRecord) -> FullCaseCitation:
    if not isinstance(record.stated, FullCaseCitation):
        msg = f"Expected FullCaseCitation, got {type(record.stated).__name__}"
        raise ValueError(msg)
    return record.stated


def _reporter_locator(record: CitationRecord) -> str:
    citation = _citation(record)
    reporter = citation.reporter.as_written if hasattr(citation.reporter, "as_written") else citation.reporter
    return " ".join(str(value) for value in (citation.volume, reporter, citation.page) if value)


def _message(
    provider: str,
    outcome: FullReporterSearchOutcome,
    attempts: tuple[MetadataSearchAttempt, ...],
    candidate_count: int,
) -> str:
    completed = sum(attempt.status is ValidationNodeStatus.SUCCEEDED for attempt in attempts)
    return (
        f"{provider} full-reporter metadata discovery ran {completed}/{len(attempts)} bounded query attempts, "
        f"retaining {candidate_count} distinct candidate record(s); outcome={outcome.value}."
    )


def _errors(attempts: tuple[MetadataSearchAttempt, ...]) -> str | None:
    errors = tuple(attempt.error for attempt in attempts if attempt.error)
    return "; ".join(errors) if errors else None


def _require_stage(document: Document, required: str, stage_name: str) -> None:
    if required not in document.passes:
        msg = f"{stage_name} requires {required} before it can run."
        raise ValueError(msg)


def _reject_partial_stage(document: Document, stage: str) -> None:
    if any(node.stage == stage for record in document.citations for node in record.trace):
        msg = f"Cannot resume partial {stage}; restart from the preceding Document checkpoint."
        raise ValueError(msg)
