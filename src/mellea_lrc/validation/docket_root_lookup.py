"""Retrieve docket-root search evidence for a later identity review."""

from __future__ import annotations

import json
import re
import time
from contextlib import ExitStack
from typing import Literal, Protocol
from urllib.parse import parse_qs, urlparse

from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerError
from mellea_lrc.courtlistener.models import CourtListenerSearchPage
from mellea_lrc.llm.fuzziness import FuzzinessOption
from mellea_lrc.llm.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupCandidate,
    DocketLookupFailure,
)
from mellea_lrc.model.document import Document

STAGE = "docket_root_lookup"
MINIMUM_SIMILARITY_PERCENT = 40.0
# Each saved page retains its upstream `next` link. Reaching this budget is
# recorded as an attempt failure, so a partial search cannot look complete.
MAX_PAGES_PER_ATTEMPT = 10
MAX_RETRIES_PER_PAGE = 2
MAX_RETRY_DELAY_SECONDS = 60.0

_SEARCH_TYPES: tuple[Literal["d", "o"], ...] = ("d", "o")
_QUERY_SPECIAL = re.compile(r"(\\|&&|\|\||[+!(){}\[\]^\"~*?:/\-])")
_DIGIT_RUN = re.compile(r"[0-9]+")
_SHORTLIST_FUZZINESS = FuzzinessOption.edit_distance(
    similarity_percent=MINIMUM_SIMILARITY_PERCENT, whitespace_relaxation=True
)
_SCORE_FUZZINESS = FuzzinessOption.edit_distance(similarity_percent=1.0, whitespace_relaxation=True)


class DocketSearchClient(Protocol):
    """The narrow CourtListener search contract needed by this stage."""

    def search(
        self, q: str, search_type: Literal["d", "o"], cursor: str | None = None
    ) -> CourtListenerSearchPage: ...


def _query(number: str) -> str:
    """Search the docketNumber field using the source number as query text."""
    escaped = _QUERY_SPECIAL.sub(r"\\\1", number)
    return f"docketNumber:({escaped})"


def _queries(number: str) -> tuple[str, ...]:
    """Add one broad numeric-token query without changing the cited number.

    The final two digit runs can retrieve records stored in shorter docket
    forms. Returned hits still face the full source-number shortlist and
    later identity review.
    """
    full = _query(number)
    runs = _DIGIT_RUN.findall(number)
    if len(runs) < 2:
        return (full,)
    broad = _query(" ".join(runs[-2:]))
    return (full,) if broad == full else (full, broad)


def _similarity(source_number: str, candidate_number: str | None) -> float:
    """Score through the shared fuzzy matcher, including below-cutoff hits."""
    if not candidate_number:
        return 0.0
    evidence = GroundingEvidence((EvidenceCandidate(candidate_number.casefold(), None),))
    for policy in (_SHORTLIST_FUZZINESS, _SCORE_FUZZINESS):
        match = evidence.resolve(source_number.casefold(), policy)
        if match is not None:
            return match.similarity_percent
    return 0.0


def _failure(error: CourtListenerError) -> DocketLookupFailure:
    # Validation errors can contain tuples and other values outside JsonValue.
    detail = json.loads(json.dumps(error.upstream_detail, default=str))
    return DocketLookupFailure(
        failure_type=error.failure_type,
        message=str(error) or type(error).__name__,
        upstream_status_code=error.upstream_status_code,
        url=error.url,
        upstream_detail=detail,
    )


def _next_cursor(url: str) -> str | None:
    values = parse_qs(urlparse(url).query).get("cursor", ())
    return values[0] if len(values) == 1 and values[0] else None


def _retry_after_seconds(error: CourtListenerError) -> float | None:
    """Honor a proxy's bounded retry delay for an exhausted search token pool."""
    if error.upstream_status_code != 429 or not isinstance(error.upstream_detail, str):
        return None
    try:
        detail = json.loads(error.upstream_detail)
    except ValueError:
        return None
    if not isinstance(detail, dict):
        return None
    seconds = detail.get("retry_after_seconds")
    if not isinstance(seconds, int | float) or isinstance(seconds, bool) or seconds < 0:
        return None
    return min(float(seconds), MAX_RETRY_DELAY_SECONDS)


def _shortlist(candidates: list[DocketLookupCandidate]) -> tuple[int, ...]:
    """Present each identified record once while keeping all raw hit pointers."""
    indices: list[int] = []
    seen: set[tuple[str, str]] = set()
    for index, candidate in enumerate(candidates):
        if candidate.docket_similarity < MINIMUM_SIMILARITY_PERCENT:
            continue
        if candidate.record_id is not None:
            key = (candidate.source_type, candidate.record_id)
            if key in seen:
                continue
            seen.add(key)
        indices.append(index)
    return tuple(indices)


def _search_attempt(
    service: DocketSearchClient,
    query: str,
    search_type: Literal["d", "o"],
    source_number: str,
    attempt_index: int,
) -> tuple[DocketLookupAttempt, tuple[DocketLookupCandidate, ...]]:
    pages: list[dict] = []
    candidates: list[DocketLookupCandidate] = []
    retry_failures: list[DocketLookupFailure] = []
    failure: DocketLookupFailure | None = None
    cursor: str | None = None
    seen_cursors: set[str] = set()

    while len(pages) < MAX_PAGES_PER_ATTEMPT:
        for retry_index in range(MAX_RETRIES_PER_PAGE + 1):
            try:
                page = service.search(query, search_type, cursor=cursor)
                break
            except CourtListenerError as error:
                delay = _retry_after_seconds(error)
                if delay is None or retry_index == MAX_RETRIES_PER_PAGE:
                    failure = _failure(error)
                    break
                retry_failures.append(_failure(error))
                time.sleep(delay)
        if failure is not None:
            break

        page_index = len(pages)
        pages.append(page.raw_json)
        for result_index, result in enumerate(page.results):
            record_id = (
                result.docket_id or result.id if search_type == "d" else result.cluster_id or result.id
            )
            candidates.append(
                DocketLookupCandidate(
                    source_type=search_type,
                    record_id=record_id,
                    attempt_index=attempt_index,
                    page_index=page_index,
                    result_index=result_index,
                    docket_number=result.docket_number,
                    docket_similarity=_similarity(source_number, result.docket_number),
                )
            )

        if page.next is None:
            break
        if len(pages) == MAX_PAGES_PER_ATTEMPT:
            failure = DocketLookupFailure(
                failure_type="page_limit_reached",
                message=f"CourtListener search still has results after {MAX_PAGES_PER_ATTEMPT} pages",
                url=page.next,
            )
            break
        next_cursor = _next_cursor(page.next)
        if next_cursor is None or next_cursor in seen_cursors:
            failure = DocketLookupFailure(
                failure_type="invalid_pagination",
                message="CourtListener search returned a missing or repeated cursor",
                url=page.next,
            )
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return (
        DocketLookupAttempt(
            source_type=search_type,
            query=query,
            pages=tuple(pages),
            retry_failures=tuple(retry_failures),
            failure=failure,
        ),
        tuple(candidates),
    )


def docket_root_lookup(document: Document, *, client: DocketSearchClient | None = None) -> Document:
    """Save docket and opinion search traces for every docket root.

    All returned hits remain addressable through their saved page and result
    indices. Court, year, and case name do not remove hits at this stage.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "10_roots" not in document.stage_runs:
        raise ValueError("Form roots before docket lookup")

    roots = tuple(root for root in document.roots if isinstance(root, FullDocketCitation))
    with ExitStack() as stack:
        service = client
        for root in roots:
            recorded = root.record(STAGE)
            span = root.locator[-1].number_span
            source_number = document.text[span.start : span.end]
            if not source_number.strip():
                lookup = DocketLookup(
                    node_id=recorded.nodes[-1].id,
                    failure=DocketLookupFailure(
                        failure_type="missing_docket_number",
                        message="The source docket-number span has no searchable text",
                    ),
                )
            else:
                attempts: list[DocketLookupAttempt] = []
                candidates: list[DocketLookupCandidate] = []
                for query in _queries(source_number):
                    for search_type in _SEARCH_TYPES:
                        if service is None:
                            try:
                                service = stack.enter_context(CourtListenerClient())
                            except CourtListenerError as error:
                                attempts.append(
                                    DocketLookupAttempt(
                                        source_type=search_type, query=query, failure=_failure(error)
                                    )
                                )
                                continue
                        attempt, found = _search_attempt(
                            service, query, search_type, source_number, len(attempts)
                        )
                        attempts.append(attempt)
                        candidates.extend(found)
                lookup = DocketLookup(
                    node_id=recorded.nodes[-1].id,
                    attempts=tuple(attempts),
                    candidates=tuple(candidates),
                    shortlisted_candidate_indices=_shortlist(candidates),
                )
            document = document.replace_citation(recorded.with_docket_lookup(lookup))
    return document.complete(STAGE)
