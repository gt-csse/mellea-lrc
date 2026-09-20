"""Shared, source-grounded mechanics for provider metadata discovery.

This module owns the reusable mechanics only: preparing grounded case-name
terms, constructing provider-neutral query provenance, executing bounded
provider requests, and retaining result pages.  Citation-family stages own
eligibility, their query family, and every identity decision.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mellea_lrc.core.fuzziness import FuzzinessOption, fuzzy_literal
from mellea_lrc.core.record import Node, Reads
from mellea_lrc.govinfo import GovInfoClient, govinfo_package_candidate
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.validation.types import MetadataSearchAttempt, ValidationNodeStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient


# Discovery and later model review are deliberately separate bounded work.
MAX_DISCOVERY_CANDIDATES_PER_ATTEMPT = 100
MAX_CANDIDATES_FOR_LATER_REVIEW = 20
MAX_TERM_PREPARATION_REPAIR_TURNS = 2
MAX_TERM_PREPARATION_TOKENS = 256

_TERM_PREFIX = """
You prepare source-grounded search terms for a legal case metadata search.

Choose one to three distinctive text fragments from the supplied case_name.
Each term must be copied from it exactly apart from whitespace layout. Prefer
proper names or distinctive organization names. For an adversarial case, use a
term from each side when useful; for In re, Ex parte, or similar one-party
matters, choose the distinctive title words. Do not use outside knowledge,
correct spelling, expand or abbreviate a name, infer missing words, return
court names, docket numbers, generic legal labels, query syntax, or quotes.
Return an empty list only when case_name has no distinctive usable words. Give
a short reason for the selection.
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
        first = client.search(spec.query, "d")
        candidates = tuple(first.results)
        cursor = first.next_cursor
        if first.count <= MAX_DISCOVERY_CANDIDATES_PER_ATTEMPT:
            candidates, cursor = _all_courtlistener_candidates(client, spec.query, first)
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
                )
        return MetadataSearchAttempt(
            kind=spec.kind,
            query=spec.query,
            court_id=spec.court_id,
            status=ValidationNodeStatus.SUCCEEDED,
            candidate_count=first.count,
            candidates=candidates,
            continuation=cursor,
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
        result = client.search_uscourts(spec.query, page_size=MAX_DISCOVERY_CANDIDATES_PER_ATTEMPT)
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
            )
        return MetadataSearchAttempt(
            kind=spec.kind,
            query=result.query,
            court_id=spec.court_id,
            status=ValidationNodeStatus.SUCCEEDED,
            candidate_count=result.count,
            candidates=candidates,
            continuation=result.next_offset_mark,
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
        for candidate in attempt.candidates:
            identifier = candidate.get(key)
            fingerprint = str(identifier) if identifier is not None else repr(sorted(candidate.items()))
            if fingerprint not in seen:
                seen.add(fingerprint)
                merged.append(candidate)
    return tuple(merged)


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
    return ValidationResult(result=True)


def _copied_from(value: str, source_name: str) -> bool:
    return (
        bool(value)
        and re.search(
            fuzzy_literal(value, FuzzinessOption.whitespace_relaxation(), newline=True), source_name
        )
        is not None
    )


def _query_term(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _all_courtlistener_candidates(client: CourtListenerServiceClient, query: str, first_page):
    candidates = list(first_page.results)
    cursor = first_page.next_cursor
    seen: set[str] = set()
    while cursor is not None:
        if cursor in seen:
            msg = "CourtListener repeated a docket-search cursor"
            raise ValueError(msg)
        seen.add(cursor)
        page = client.search(query, "d", cursor=cursor)
        if page.count != first_page.count:
            msg = "CourtListener changed the docket-search result count while paging"
            raise ValueError(msg)
        candidates.extend(page.results)
        cursor = page.next_cursor
    return tuple(candidates), cursor
