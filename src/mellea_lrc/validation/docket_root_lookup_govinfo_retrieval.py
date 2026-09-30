"""Retrieve USCOURTS search evidence for docket roots not resolved by CourtListener."""

from __future__ import annotations

import json
import re
import time
from contextlib import ExitStack
from typing import Protocol

from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import DocketLookupFailure
from mellea_lrc.model.citations.govinfo_lookup import (
    GovInfoDocketLookup,
    GovInfoLookupAttempt,
    GovInfoLookupCandidate,
)
from mellea_lrc.model.document import Document
from mellea_lrc.providers.govinfo import (
    GovInfoClient,
    GovInfoError,
    GovInfoSearchPage,
    govinfo_uscourts_docket_query,
)

STAGE = "18_docket_root_lookup_govinfo_retrieval"
MINIMUM_SIMILARITY_PERCENT = 40.0
MAX_PAGES_PER_ATTEMPT = 10
MAX_RETRIES_PER_PAGE = 2

_PACKAGE_ID = re.compile(r"^USCOURTS-([a-z0-9]+)-(.+)$", re.IGNORECASE)
_SHORTLIST_FUZZINESS = FuzzinessOption.edit_distance(
    similarity_percent=MINIMUM_SIMILARITY_PERCENT, whitespace_relaxation=True
)
_SCORE_FUZZINESS = FuzzinessOption.edit_distance(similarity_percent=1.0, whitespace_relaxation=True)


class GovInfoSearchClient(Protocol):
    """The narrow GovInfo search contract needed by this stage."""

    def search(self, query: str, *, offset_mark: str = "*", page_size: int = 100) -> GovInfoSearchPage: ...


def _result_string(result: dict, *keys: str) -> str | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _package_number(package_id: str | None) -> tuple[str | None, str | None]:
    """Return the explicit package court code and its encoded docket number."""
    if package_id is None:
        return None, None
    match = _PACKAGE_ID.fullmatch(package_id)
    if match is None:
        return None, None
    court_code, encoded_number = match.groups()
    if not encoded_number or not re.fullmatch(r"[A-Za-z0-9_.-]+", encoded_number):
        return None, None
    # GovInfo documents colons as underscores in package IDs. Do not try to
    # infer a number from any package ID that does not fit its published form.
    return court_code, encoded_number.replace("_", ":")


def _similarity(source_number: str, candidate_number: str | None) -> float:
    if not candidate_number:
        return 0.0
    evidence = GroundingEvidence((EvidenceCandidate(candidate_number.casefold(), None),))
    for policy in (_SHORTLIST_FUZZINESS, _SCORE_FUZZINESS):
        match = evidence.resolve(source_number.casefold(), policy)
        if match is not None:
            return match.similarity_percent
    return 0.0


def _failure(error: GovInfoError) -> DocketLookupFailure:
    detail = json.loads(json.dumps(error.upstream_detail, default=str))
    return DocketLookupFailure(
        failure_type=error.failure_type,
        message=str(error) or type(error).__name__,
        upstream_status_code=error.upstream_status_code,
        url=error.url,
        upstream_detail=detail,
    )


def _retryable(error: GovInfoError) -> bool:
    return (
        error.failure_type == "transport_error"
        or error.upstream_status_code == 429
        or (error.upstream_status_code is not None and error.upstream_status_code >= 500)
    )


def _shortlist(candidates: list[GovInfoLookupCandidate]) -> tuple[int, ...]:
    indices: list[int] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if candidate.docket_similarity < MINIMUM_SIMILARITY_PERCENT:
            continue
        if candidate.package_id:
            if candidate.package_id in seen:
                continue
            seen.add(candidate.package_id)
        indices.append(index)
    return tuple(indices)


def _query_numbers(number: str) -> tuple[str, ...]:
    """Try a common compact year/sequence spelling without changing the citation."""
    if re.fullmatch(r"\d{7}", number):
        # Federal CM/ECF often writes a two-digit filing year and five-digit
        # sequence with a hyphen. This is a search spelling, not a docket
        # normalization or an assertion that either spelling identifies a case.
        return number, f"{number[:2]}-{number[2:]}"
    return (number,)


def _search(
    service: GovInfoSearchClient, query: str, source_number: str, attempt_index: int
) -> tuple[GovInfoLookupAttempt, tuple[GovInfoLookupCandidate, ...]]:
    pages: list[dict] = []
    count: int | None = None
    next_marks: list[str] = []
    retry_failures: list[DocketLookupFailure] = []
    candidates: list[GovInfoLookupCandidate] = []
    failure: DocketLookupFailure | None = None
    offset_mark = "*"
    seen_marks = {offset_mark}

    while len(pages) < MAX_PAGES_PER_ATTEMPT:
        for retry_index in range(MAX_RETRIES_PER_PAGE + 1):
            try:
                page = service.search(query, offset_mark=offset_mark, page_size=100)
                break
            except GovInfoError as error:
                if not _retryable(error) or retry_index == MAX_RETRIES_PER_PAGE:
                    failure = _failure(error)
                    break
                retry_failures.append(_failure(error))
                time.sleep(2**retry_index)
        if failure is not None:
            break

        page_index = len(pages)
        pages.append(page.raw_json)
        count = page.count
        for result_index, result in enumerate(page.results):
            package_id = _result_string(result, "packageId", "package_id")
            granule_id = _result_string(result, "granuleId", "granule_id")
            court_code, package_number = _package_number(package_id)
            case_number = _result_string(result, "caseNumber", "case_number") or package_number
            candidates.append(
                GovInfoLookupCandidate(
                    attempt_index=attempt_index,
                    page_index=page_index,
                    result_index=result_index,
                    package_id=package_id,
                    granule_id=granule_id,
                    court_code=court_code,
                    docket_number=case_number,
                    docket_similarity=_similarity(source_number, case_number),
                )
            )

        next_mark = page.next_offset_mark
        next_marks.append(next_mark or "")
        if next_mark is None or len(candidates) >= page.count:
            break
        if next_mark in seen_marks:
            failure = DocketLookupFailure(
                failure_type="invalid_pagination",
                message="GovInfo search returned a repeated offset mark",
            )
            break
        if len(pages) == MAX_PAGES_PER_ATTEMPT:
            failure = DocketLookupFailure(
                failure_type="page_limit_reached",
                message=f"GovInfo search still has results after {MAX_PAGES_PER_ATTEMPT} pages",
            )
            break
        seen_marks.add(next_mark)
        offset_mark = next_mark

    attempt = GovInfoLookupAttempt(
        query=query,
        pages=tuple(pages),
        count=count,
        next_offset_marks=tuple(next_marks),
        retry_failures=tuple(retry_failures),
        failure=failure,
    )
    return attempt, tuple(candidates)


def docket_root_lookup_govinfo_retrieval(
    document: Document, *, client: GovInfoSearchClient | None = None
) -> Document:
    """Search GovInfo only for roots without a selected CourtListener case."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "17_docket_root_lookup_courtlistener_llm_review" not in document.stage_runs:
        raise ValueError("Complete CourtListener docket review before GovInfo lookup")

    service = client
    with ExitStack() as stack:
        for root in tuple(item for item in document.roots if isinstance(item, FullDocketCitation)):
            review = root.docket_lookup_review
            if (
                review is not None
                and review.decision is not None
                and review.decision.selected_candidate_index is not None
            ):
                continue

            recorded = root.record(STAGE)
            locator = root.locator[-1]
            source_number = document.text[locator.number_span.start : locator.number_span.end]
            if not source_number.strip():
                lookup = GovInfoDocketLookup(
                    node_id=recorded.nodes[-1].id,
                    failure=DocketLookupFailure(
                        failure_type="missing_docket_number",
                        message="The source docket-number span has no searchable text",
                    ),
                )
            else:
                if service is None:
                    service = stack.enter_context(GovInfoClient())
                attempts: list[GovInfoLookupAttempt] = []
                candidates: list[GovInfoLookupCandidate] = []
                for query_number in _query_numbers(source_number):
                    query = govinfo_uscourts_docket_query(query_number)
                    attempt, found = _search(service, query, source_number, len(attempts))
                    attempts.append(attempt)
                    candidates.extend(found)
                lookup = GovInfoDocketLookup(
                    node_id=recorded.nodes[-1].id,
                    attempts=tuple(attempts),
                    candidates=tuple(candidates),
                    shortlisted_candidate_indices=_shortlist(candidates),
                )
            document = document.replace_citation(recorded.with_govinfo_docket_lookup(lookup))
    return document.complete(STAGE)
