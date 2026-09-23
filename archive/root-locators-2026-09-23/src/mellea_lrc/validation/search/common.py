"""Shared, source-grounded mechanics for provider metadata discovery.

This module owns the reusable mechanics only: preparing grounded case-name
terms, constructing provider-neutral query provenance, executing bounded
provider requests, and retaining result pages.  Citation-family stages own
eligibility, their query family, and every identity decision.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mellea_lrc.govinfo import GovInfoClient, govinfo_package_candidate
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.model.fuzziness import FuzzinessOption, fuzzy_literal
from mellea_lrc.model.record import Node, Reads
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.validation.types import MetadataSearchAttempt, ValidationNodeStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.model.record import CitationRecord


# Discovery and later model review are deliberately separate bounded work.
MAX_DISCOVERY_CANDIDATES_PER_ATTEMPT = 100
MAX_CANDIDATES_FOR_LATER_REVIEW = 20
MAX_TERM_PREPARATION_REPAIR_TURNS = 2
MAX_TERM_PREPARATION_TOKENS = 256
# Provider throttling and transient upstream failures are retriable.  Keep the
# policy here so every metadata-discovery caller gets the same small recovery
# window and the persisted attempt says when it was used.
_RETRY_DELAYS_SECONDS = (0.5, 1.0)
# A shared client is reused across a corpus. Pace distinct metadata requests
# modestly so a burst of source-derived terms does not turn an otherwise
# searchable root into a sequence of 429 responses. Retries have their own
# explicit backoff and are not additionally paced.
_MINIMUM_REQUEST_INTERVAL_SECONDS = 0.35
_LAST_METADATA_REQUEST_AT: dict[int, float] = {}
_Result = TypeVar("_Result")

_TERM_PREFIX = """
You prepare source-grounded search terms for a legal case metadata search.

Choose one to three short, distinctive search fragments from the supplied
case_name. These are retrieval cues, not a reconstruction of the full caption:
database captions often omit a party, a corporate suffix, a middle word, or
other ordinary detail. Prefer one distinctive proper-name or organization word
before a multiword phrase: a source can abbreviate a word that a database
spells out. Use terms from both sides only when a single fragment would be too
broad. For In re, Ex parte, or similar one-party matters, choose the
distinctive title words.

Each term must be copied from case_name exactly apart from whitespace layout.
Keep each fragment to at most three words. If a longer organization or party
name contains ordinary connectors or abbreviations, choose a shorter,
distinctive contiguous fragment instead of its full wording.
Do not use outside knowledge, correct spelling, expand or abbreviate a name,
infer missing words, return court names, docket numbers, generic legal labels,
query syntax, or quotes. Return an empty list only when case_name has no
distinctive usable words. Give a short reason for the selection.
""".strip()

_TERM_INSTRUCTION = """
case_name:
{{case_name}}
""".strip()


class _TermProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[str] = Field(max_length=3)
    reason: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class MetadataTermPlan:
    """Source-grounded case-name terms and the trace node that records them."""

    terms: tuple[str, ...]
    node: Node


@dataclass(frozen=True, slots=True)
class MetadataQuery:
    """One provider query with stable provenance retained in an attempt node."""

    kind: str
    query: str
    court_id: str | None


class _RetryExhausted(Exception):
    """Keep a retried provider failure inspectable at the stage boundary."""

    def __init__(self, cause: Exception, retry_count: int, throttle_wait_seconds: float) -> None:
        self.cause = cause
        self.retry_count = retry_count
        self.throttle_wait_seconds = throttle_wait_seconds
        super().__init__(str(cause))


async def prepare_case_name_terms(
    record: CitationRecord,
    *,
    stage: str,
    made_by: str,
    source_case_name: str | None,
    session: MelleaSession | None,
) -> MetadataTermPlan:
    """Ask once for source-copied case-name terms, preserving the full IVR run."""
    node_id = f"{record.citation_id}:{stage}:case_name_terms"
    if source_case_name is None:
        return MetadataTermPlan(
            terms=(),
            node=Node(
                node_id=node_id,
                reads=Reads.DOCUMENT,
                stage=stage,
                made_by=made_by,
                outcome="unavailable",
                message="No source-stated case name was available for metadata search terms.",
                details={"source_case_name": None, "terms": [], "reason": None, "ivr": None},
            ),
        )
    try:
        resolved_session = session or start_mellea_session_from_env()
        spec = InstructIvrSpec(
            description=_TERM_INSTRUCTION,
            prefix=_TERM_PREFIX,
            user_variables={"case_name": source_case_name},
            output_format=_TermProposal,
            requirements=[
                req(
                    "Every term must be a non-empty, distinct substring copied from case_name, "
                    "allowing only whitespace relaxation; return at most three terms.",
                    validation_fn=lambda ctx: _validate_terms(ctx, source_case_name),
                )
            ],
        )
        run = await run_instruct_ivr(
            resolved_session,
            spec,
            strategy=MultiTurnStrategy(loop_budget=MAX_TERM_PREPARATION_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(
                max_tokens=MAX_TERM_PREPARATION_TOKENS
            ),
        )
        if not run.success:
            return _failed_term_plan(
                node_id=node_id,
                stage=stage,
                made_by=made_by,
                source_name=source_case_name,
                message="Case-name term preparation exhausted its repair budget.",
                error=run.failure_reason or "Model output did not satisfy the source-grounding contract.",
                run=run,
            )
        proposal = _TermProposal.model_validate_json(run.output)
        terms = tuple(term.strip() for term in proposal.terms)
        return MetadataTermPlan(
            terms=terms,
            node=Node(
                node_id=node_id,
                reads=Reads.DOCUMENT,
                stage=stage,
                made_by=made_by,
                outcome="prepared",
                message="Prepared source-grounded case-name terms for metadata discovery.",
                details={
                    "source_case_name": source_case_name,
                    "terms": list(terms),
                    "reason": proposal.reason,
                    "ivr": serialize_dataclass(run),
                },
            ),
        )
    except Exception as exc:
        return _failed_term_plan(
            node_id=node_id,
            stage=stage,
            made_by=made_by,
            source_name=source_case_name,
            message="Case-name term preparation failed before a usable query was formed.",
            error=f"{type(exc).__name__}: {exc}",
        )


def saved_case_name_terms(record: CitationRecord, *, stage: str) -> MetadataTermPlan | None:
    """Restore exactly one prior term-selection node, if this stage wrote one."""
    node_id = f"{record.citation_id}:{stage}:case_name_terms"
    matches = tuple(node for node in record.trace if node.node_id == node_id)
    if not matches:
        return None
    if len(matches) != 1:
        msg = f"Expected one saved case-name term plan for {record.citation_id!r}"
        raise ValueError(msg)
    terms = matches[0].details.get("terms")
    if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
        msg = f"Saved case-name term plan for {record.citation_id!r} is malformed"
        raise ValueError(msg)
    return MetadataTermPlan(terms=tuple(terms), node=matches[0])


def courtlistener_case_name_query(terms: tuple[str, ...], court_id: str | None) -> str:
    """Build one CourtListener docket-metadata query from grounded name terms."""
    if not terms:
        msg = "CourtListener case-name search needs at least one term"
        raise ValueError(msg)
    clauses = [f"caseName:({' AND '.join(_query_term(term) for term in terms)})"]
    if court_id:
        clauses.append(f"court_id:{court_id}")
    return " AND ".join(clauses)


def deduplicate_queries(queries: Sequence[MetadataQuery]) -> tuple[MetadataQuery, ...]:
    """Keep stable query order while avoiding a duplicate provider request."""
    seen: set[str] = set()
    return tuple(query for query in queries if not (query.query in seen or seen.add(query.query)))


def run_courtlistener_metadata_attempt(
    spec: MetadataQuery,
    client: CourtListenerServiceClient,
) -> MetadataSearchAttempt:
    """Execute one auditable CourtListener docket-corpus metadata query."""
    try:
        first, retry_count, throttle_wait = _request_with_retries(
            lambda: client.search(spec.query, "d"), pacing_key=client
        )
        candidates = tuple(first.results)
        cursor = first.next_cursor
        if first.count <= MAX_DISCOVERY_CANDIDATES_PER_ATTEMPT:
            candidates, cursor, page_retries, page_throttle_wait = _all_courtlistener_candidates(
                client, spec.query, first
            )
            retry_count += page_retries
            throttle_wait += page_throttle_wait
            if len(candidates) != first.count:
                return MetadataSearchAttempt(
                    kind=spec.kind,
                    query=spec.query,
                    court_id=spec.court_id,
                    status=ValidationNodeStatus.FAILED,
                    candidate_count=first.count,
                    candidates=candidates,
                    continuation=cursor,
                    error=f"Search reported {first.count} candidates but returned {len(candidates)} across pages.",
                    retry_count=retry_count,
                    throttle_wait_seconds=throttle_wait,
                )
        return MetadataSearchAttempt(
            kind=spec.kind,
            query=spec.query,
            court_id=spec.court_id,
            status=ValidationNodeStatus.SUCCEEDED,
            candidate_count=first.count,
            candidates=candidates,
            continuation=cursor,
            retry_count=retry_count,
            throttle_wait_seconds=throttle_wait,
        )
    except _RetryExhausted as exc:
        return MetadataSearchAttempt(
            kind=spec.kind,
            query=spec.query,
            court_id=spec.court_id,
            status=ValidationNodeStatus.FAILED,
            candidate_count=None,
            error=f"{type(exc.cause).__name__}: {exc.cause} after {exc.retry_count} retry attempt(s).",
            retry_count=exc.retry_count,
            throttle_wait_seconds=exc.throttle_wait_seconds,
        )
    except Exception as exc:
        return MetadataSearchAttempt(
            kind=spec.kind,
            query=spec.query,
            court_id=spec.court_id,
            status=ValidationNodeStatus.FAILED,
            candidate_count=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_govinfo_metadata_attempt(spec: MetadataQuery, client: GovInfoClient) -> MetadataSearchAttempt:
    """Execute one auditable GovInfo USCOURTS metadata query."""
    try:
        result, retry_count, throttle_wait = _request_with_retries(
            lambda: client.search_uscourts(spec.query, page_size=MAX_DISCOVERY_CANDIDATES_PER_ATTEMPT),
            pacing_key=client,
        )
        candidates = tuple(govinfo_package_candidate(result_item) for result_item in result.results)
        complete = result.count <= MAX_DISCOVERY_CANDIDATES_PER_ATTEMPT and len(candidates) == result.count
        if not complete and result.count <= MAX_DISCOVERY_CANDIDATES_PER_ATTEMPT:
            return MetadataSearchAttempt(
                kind=spec.kind,
                query=result.query,
                court_id=spec.court_id,
                status=ValidationNodeStatus.FAILED,
                candidate_count=result.count,
                candidates=candidates,
                continuation=result.next_offset_mark,
                error=f"Search reported {result.count} packages but returned {len(candidates)}.",
                retry_count=retry_count,
                throttle_wait_seconds=throttle_wait,
            )
        return MetadataSearchAttempt(
            kind=spec.kind,
            query=result.query,
            court_id=spec.court_id,
            status=ValidationNodeStatus.SUCCEEDED,
            candidate_count=result.count,
            candidates=candidates,
            continuation=result.next_offset_mark,
            retry_count=retry_count,
            throttle_wait_seconds=throttle_wait,
        )
    except _RetryExhausted as exc:
        return MetadataSearchAttempt(
            kind=spec.kind,
            query=spec.query,
            court_id=spec.court_id,
            status=ValidationNodeStatus.FAILED,
            candidate_count=None,
            error=f"{type(exc.cause).__name__}: {exc.cause} after {exc.retry_count} retry attempt(s).",
            retry_count=exc.retry_count,
            throttle_wait_seconds=exc.throttle_wait_seconds,
        )
    except Exception as exc:
        return MetadataSearchAttempt(
            kind=spec.kind,
            query=spec.query,
            court_id=spec.court_id,
            status=ValidationNodeStatus.FAILED,
            candidate_count=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def merge_metadata_candidates(
    attempts: tuple[MetadataSearchAttempt, ...],
    *,
    key: str,
) -> tuple[Mapping[str, object], ...]:
    """Merge provider candidates in query order, retaining first occurrence."""
    merged: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for attempt in attempts:
        # An unsuccessful paginated request can retain a partial first page for
        # diagnosis.  Preserve that page on the attempt itself, but never let
        # incomplete provider evidence enter the candidate set used by a later
        # identity decision.
        if attempt.status is not ValidationNodeStatus.SUCCEEDED:
            continue
        for candidate in attempt.candidates:
            identifier = candidate.get(key)
            fingerprint = str(identifier) if identifier is not None else repr(sorted(candidate.items()))
            if fingerprint not in seen:
                seen.add(fingerprint)
                merged.append(candidate)
    return tuple(merged)


def select_narrowest_bounded_metadata_candidates(
    attempts: tuple[MetadataSearchAttempt, ...],
    *,
    key: str,
    limit: int = MAX_CANDIDATES_FOR_LATER_REVIEW,
) -> tuple[Mapping[str, object], ...]:
    """Return candidates from the most selective complete query attempt.

    Metadata discovery deliberately tries a strict case-name conjunction before
    relaxing to one source-grounded name fragment.  A broad fallback can
    legitimately return hundreds of records, while a later fragment can return
    a small complete set containing the same record.  Broad result pages stay
    on their ``MetadataSearchAttempt`` for inspection, but must not prevent the
    bounded candidate resolver from reviewing that narrower evidence.

    This is a retrieval rule, not a case-name normalizer: it only compares
    provider result counts.  Every tied narrowest complete attempt contributes
    its candidates so the selection does not depend on provider query order.
    """
    eligible = tuple(
        attempt
        for attempt in attempts
        if attempt.status is ValidationNodeStatus.SUCCEEDED
        and attempt.candidate_count is not None
        and 0 < attempt.candidate_count < limit
        and len(attempt.candidates) == attempt.candidate_count
    )
    if not eligible:
        return ()
    narrowest_count = min(
        attempt.candidate_count for attempt in eligible if attempt.candidate_count is not None
    )
    narrowest = tuple(attempt for attempt in eligible if attempt.candidate_count == narrowest_count)
    return merge_metadata_candidates(narrowest, key=key)


def _failed_term_plan(
    *,
    node_id: str,
    stage: str,
    made_by: str,
    source_name: str,
    message: str,
    error: str,
    run: object | None = None,
) -> MetadataTermPlan:
    return MetadataTermPlan(
        terms=(),
        node=Node(
            node_id=node_id,
            reads=Reads.DOCUMENT,
            stage=stage,
            made_by=made_by,
            outcome="failed",
            message=message,
            details={
                "source_case_name": source_name,
                "terms": [],
                "reason": None,
                "error": error,
                "ivr": serialize_dataclass(run) if run is not None else None,
            },
        ),
    )


def _validate_terms(ctx: Context, source_name: str) -> ValidationResult:
    try:
        proposal = _TermProposal.model_validate_json(str(ctx.last_output().value))
    except ValidationError as exc:
        return ValidationResult(result=False, reason=f"Output must match the term schema: {exc}")
    terms = tuple(term.strip() for term in proposal.terms)
    if len(terms) > 3:
        return ValidationResult(result=False, reason="Return no more than three terms.")
    if len({term.casefold() for term in terms}) != len(terms):
        return ValidationResult(result=False, reason="Terms must be distinct.")
    if any(not _copied_from(term, source_name) for term in terms):
        return ValidationResult(
            result=False,
            reason="Every term must be copied from case_name with only whitespace layout relaxed.",
        )
    if any(not _is_compact_search_fragment(term) for term in terms):
        return ValidationResult(
            result=False,
            reason="Each search fragment must contain at most three words; return a shorter source-copied fragment.",
        )
    return ValidationResult(result=True)


def _copied_from(value: str, source_name: str) -> bool:
    return (
        bool(value)
        and re.search(
            fuzzy_literal(value, FuzzinessOption.whitespace_relaxation(), newline=True), source_name
        )
        is not None
    )


def _is_compact_search_fragment(value: str) -> bool:
    """Keep one discovery cue short enough for a tolerant pair query."""
    return len(re.findall(r"\w+", value)) <= 3


def _query_term(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _request_with_retries(
    request: Callable[[], _Result],
    *,
    pacing_key: object,
) -> tuple[_Result, int, float]:
    """Retry an explicitly retryable provider request with bounded backoff."""
    retry_count = 0
    throttle_wait = _pace_metadata_request(pacing_key)
    while True:
        try:
            return request(), retry_count, throttle_wait
        except Exception as exc:
            if not bool(getattr(exc, "retryable", False)) or retry_count >= len(_RETRY_DELAYS_SECONDS):
                if retry_count:
                    raise _RetryExhausted(exc, retry_count, throttle_wait) from exc
                raise
            time.sleep(_RETRY_DELAYS_SECONDS[retry_count])
            retry_count += 1


def _pace_metadata_request(pacing_key: object) -> float:
    """Wait only long enough to keep consecutive metadata calls well spaced."""
    key = id(pacing_key)
    now = time.monotonic()
    previous = _LAST_METADATA_REQUEST_AT.get(key)
    wait = max(0.0, _MINIMUM_REQUEST_INTERVAL_SECONDS - (now - previous)) if previous is not None else 0.0
    if wait:
        time.sleep(wait)
    _LAST_METADATA_REQUEST_AT[key] = time.monotonic()
    return wait


def _all_courtlistener_candidates(client: CourtListenerServiceClient, query: str, first_page):
    candidates = list(first_page.results)
    cursor = first_page.next_cursor
    seen: set[str] = set()
    retry_count = 0
    throttle_wait = 0.0
    while cursor is not None:
        if cursor in seen:
            msg = "CourtListener repeated a docket-search cursor"
            raise ValueError(msg)
        seen.add(cursor)
        page, page_retries, page_throttle_wait = _request_with_retries(
            lambda: client.search(query, "d", cursor=cursor), pacing_key=client
        )
        retry_count += page_retries
        throttle_wait += page_throttle_wait
        if page.count != first_page.count:
            msg = "CourtListener changed the docket-search result count while paging"
            raise ValueError(msg)
        candidates.extend(page.results)
        cursor = page.next_cursor
    return tuple(candidates), cursor, retry_count, throttle_wait
