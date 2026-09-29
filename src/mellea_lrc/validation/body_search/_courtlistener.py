"""Bounded CourtListener search and full-body retrieval for two independent stages."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import parse_qs, urlparse

from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerError
from mellea_lrc.courtlistener.models import CourtListenerSearchPage, CourtListenerSearchResult
from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.body_evidence import (
    BodyEvidence,
    BodyEvidenceFailure,
    BodySearch,
    BodySearchAttempt,
    BodySource,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search.common import (
    eligible_on,
    evidence_date,
    locator_text,
    make_body_evidences,
    roots_for_body_search,
)

MAX_PAGES_PER_QUERY = 2
MAX_HITS_PER_QUERY = 40
MAX_FETCHES_PER_CITATION = 8
MAX_TRANSIENT_RETRIES = 2
MAX_RETRY_AFTER_SECONDS = 300
_OPINION_TEXT_FIELDS = (
    "plain_text",
    "html_with_citations",
    "html_lawbox",
    "html_columbia",
    "html_anon_2020",
    "html",
    "xml_harvard",
)


class CourtListenerBodyClient(Protocol):
    """Only the search and detail methods needed by the body stages."""

    def search(
        self, q: str, search_type: Literal["o", "rd"], cursor: str | None = None
    ) -> CourtListenerSearchPage: ...

    def get_opinion(self, opinion_id: str) -> dict[str, Any] | None: ...

    def get_recap_document(self, recap_document_id: str) -> dict[str, Any] | None: ...


_Result = TypeVar("_Result")


def _retry_after(error: CourtListenerError, attempt: int) -> float | None:
    """Retry transient transport/server failures and bounded quota pauses."""
    if error.failure_type == "transport_error" or (
        error.upstream_status_code is not None and error.upstream_status_code >= 500
    ):
        return float(2 ** (attempt + 1))
    if error.upstream_status_code != 429:
        return None
    if isinstance(error.upstream_detail, str):
        try:
            detail = json.loads(error.upstream_detail)
        except ValueError:
            detail = None
        if isinstance(detail, dict) and "retry_after_seconds" in detail:
            seconds = detail["retry_after_seconds"]
            if type(seconds) not in {int, float} or not 0 <= seconds <= MAX_RETRY_AFTER_SECONDS:
                return None
            return min(float(seconds) + 0.25, MAX_RETRY_AFTER_SECONDS)
    return float(2 ** (attempt + 1))


def _retry_transient(action: Callable[[], _Result]) -> _Result:
    """Retry brief provider failures; preserve exhaustion for the run artifact."""
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            return action()
        except CourtListenerError as error:
            delay = _retry_after(error, attempt)
            if delay is None or attempt == MAX_TRANSIENT_RETRIES:
                raise
            time.sleep(delay)
    raise AssertionError("Transient retry loop did not return or raise")


@dataclass(slots=True)
class _CandidateBudget:
    fetched: int = 0
    seen_ids: set[str] = field(default_factory=set)


class _TextOnlyHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.suppressed += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.suppressed:
            self.suppressed -= 1

    def handle_data(self, data: str) -> None:
        if not self.suppressed:
            self.parts.append(data)


def _body_text(record: dict[str, Any], source: BodySource) -> tuple[str, str | None]:
    fields = _OPINION_TEXT_FIELDS if source is BodySource.COURTLISTENER_OPINION else ("plain_text",)
    for name in fields:
        value = record.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        if name == "plain_text":
            return value, name
        parser = _TextOnlyHTML()
        parser.feed(value)
        parser.close()
        text = " ".join(" ".join(parser.parts).split())
        if text:
            return text, name
    return "", None


def _record_id(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return value
    return None


def _opinion_ids(hit: CourtListenerSearchResult) -> tuple[str, ...]:
    """Opinion search IDs belong to clusters; only nested IDs name bodies."""
    found: list[str] = []
    raw = hit.raw_json
    for key in ("opinions", "sibling_ids"):
        values = raw.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict):
                value = value.get("id", value.get("opinion_id"))
            opinion_id = _record_id(value)
            if opinion_id is not None and opinion_id not in found:
                found.append(opinion_id)
    return tuple(found)


def _hit_ids(hit: CourtListenerSearchResult, source: BodySource) -> tuple[str, ...]:
    if source is BodySource.COURTLISTENER_OPINION:
        return _opinion_ids(hit)
    recap_id = _record_id(hit.id)
    return (recap_id,) if recap_id is not None else ()


def _issue_date(
    record: dict[str, Any], hit: CourtListenerSearchResult, source: BodySource
) -> tuple[date | None, str | None]:
    if source is BodySource.COURTLISTENER_OPINION:
        # Search dates belong to clusters. They can date the fetched opinion
        # only when that cluster has a single opinion; multiple subopinions
        # may have different issue dates and cannot safely pass a cutoff.
        fields = ((record, "date_filed", "opinion.date_filed"),)
        if len(_opinion_ids(hit)) == 1:
            fields += (
                (hit.raw_json, "dateFiled", "search.dateFiled"),
                (hit.raw_json, "date_filed", "search.date_filed"),
            )
    else:
        # Search dateFiled is the case's filing date, not the document's.
        fields = (
            (record, "date_filed", "recap_document.date_filed"),
            (record, "entry_date_filed", "recap_document.entry_date_filed"),
            (hit.raw_json, "entry_date_filed", "search.entry_date_filed"),
            (hit.raw_json, "entryDateFiled", "search.entryDateFiled"),
        )
    for data, key, basis in fields:
        issued_on = evidence_date(data.get(key))
        if issued_on is not None:
            return issued_on, basis
    return None, None


def _failure(
    failure_type: str, message: str, *, item_id: str | None = None, status_code: int | None = None
) -> BodyEvidenceFailure:
    return BodyEvidenceFailure(
        failure_type=failure_type, message=message, item_id=item_id, status_code=status_code
    )


def _service_failure(error: CourtListenerError, *, item_id: str | None = None) -> BodyEvidenceFailure:
    parts = [str(error) or type(error).__name__]
    if error.url:
        parts.append(f"url={error.url}")
    if error.upstream_detail is not None:
        parts.append(f"detail={error.upstream_detail!r}")
    return _failure(
        error.failure_type,
        "; ".join(parts),
        item_id=item_id,
        status_code=error.upstream_status_code,
    )


def _next_cursor(url: str) -> str | None:
    values = parse_qs(urlparse(url).query).get("cursor", ())
    return values[0] if len(values) == 1 and values[0] else None


def _locator_query(locator: str) -> str:
    return '"' + locator.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _parent_id(hit: CourtListenerSearchResult, source: BodySource) -> str | None:
    return hit.cluster_id if source is BodySource.COURTLISTENER_OPINION else hit.docket_id


def _linked_cluster_id(record: dict[str, Any]) -> str | None:
    cluster = record.get("cluster")
    if isinstance(cluster, dict):
        return _record_id(cluster.get("id"))
    direct = _record_id(cluster)
    if direct is not None:
        return direct
    if isinstance(cluster, str):
        match = re.search(r"/clusters/([0-9]+)/?$", urlparse(cluster).path)
        return match.group(1) if match else None
    return None


def _url(record: dict[str, Any], hit: CourtListenerSearchResult) -> str | None:
    value = record.get("absolute_url")
    return value if isinstance(value, str) and value else hit.absolute_url


def _fetch_evidence(
    service: CourtListenerBodyClient,
    *,
    item_id: str,
    hit: CourtListenerSearchResult,
    source: BodySource,
    locator: str,
    source_text: str,
    retrospective_date: date | None,
    query: str,
) -> tuple[tuple[BodyEvidence, ...], BodyEvidenceFailure | None]:
    try:
        record = (
            _retry_transient(lambda: service.get_opinion(item_id))
            if source is BodySource.COURTLISTENER_OPINION
            else _retry_transient(lambda: service.get_recap_document(item_id))
        )
    except CourtListenerError as error:
        return (), _service_failure(error, item_id=item_id)
    if record is None:
        return (), _failure("missing_record", "Detail endpoint returned 404", item_id=item_id)
    returned_id = _record_id(record.get("id"))
    if returned_id != item_id:
        return (), _failure(
            "record_id_mismatch",
            f"Detail endpoint returned ID {returned_id!r} for {item_id}",
            item_id=item_id,
        )
    if source is BodySource.COURTLISTENER_OPINION and hit.cluster_id is not None:
        linked_cluster_id = _linked_cluster_id(record)
        if linked_cluster_id is not None and linked_cluster_id != hit.cluster_id:
            return (), _failure(
                "parent_id_mismatch",
                f"Opinion {item_id} belongs to cluster {linked_cluster_id}, not {hit.cluster_id}",
                item_id=item_id,
            )
    issued_on, date_basis = _issue_date(record, hit, source)
    if not eligible_on(issued_on, retrospective_date):
        reason = (
            "Issued item has no exact filing date for the retrospective cutoff"
            if issued_on is None
            else f"Issued item date {issued_on.isoformat()} is after {retrospective_date.isoformat()}"
        )
        return (), _failure("ineligible_issue_date", reason, item_id=item_id)
    body_text, text_field = _body_text(record, source)
    if not body_text:
        return (), _failure("missing_full_text", "Fetched item has no usable full text", item_id=item_id)
    metadata: dict[str, Any] = {
        "text_field": text_field,
        "query": query,
        "detail_id": record["id"],
        "detail_dates": {key: record[key] for key in ("date_filed", "entry_date_filed") if key in record},
    }
    if source is BodySource.COURTLISTENER_OPINION and hit.cluster_id is not None:
        metadata["cluster_id"] = hit.cluster_id
    evidence = make_body_evidences(
        body_id=item_id,
        parent_id=_parent_id(hit, source),
        url=_url(record, hit),
        issued_on=issued_on,
        date_basis=date_basis,
        metadata=metadata,
        body_text=body_text,
        locator=locator,
        source_text=source_text,
    )
    if not evidence:
        return (), _failure(
            "no_grounded_anchor",
            "Fetched body does not contain the cited locator",
            item_id=item_id,
        )
    return evidence, None


def _search_query(
    service: CourtListenerBodyClient,
    *,
    query: str,
    source: BodySource,
    locator: str,
    source_text: str,
    retrospective_date: date | None,
    budget: _CandidateBudget,
) -> tuple[BodySearchAttempt, tuple[BodyEvidence, ...], tuple[BodyEvidenceFailure, ...]]:
    pages: list[dict[str, Any]] = []
    evidence: list[BodyEvidence] = []
    failures: list[BodyEvidenceFailure] = []
    attempt_failure: BodyEvidenceFailure | None = None
    cursor: str | None = None
    seen_cursors: set[str] = set()
    hits_seen = 0
    search_type: Literal["o", "rd"] = "o" if source is BodySource.COURTLISTENER_OPINION else "rd"

    while len(pages) < MAX_PAGES_PER_QUERY:
        try:
            page = _retry_transient(lambda: service.search(query, search_type, cursor=cursor))
        except CourtListenerError as error:
            attempt_failure = _service_failure(error)
            break
        pages.append(page.raw_json)
        exhausted = False
        hit_limit_reached = False
        for hit in page.results:
            if hits_seen == MAX_HITS_PER_QUERY:
                hit_limit_reached = True
                break
            hits_seen += 1
            ids = _hit_ids(hit, source)
            if not ids:
                failures.append(
                    _failure(
                        "missing_body_id",
                        "Search hit did not identify a retrievable opinion or RECAP document",
                        item_id=_parent_id(hit, source),
                    )
                )
                continue
            for item_id in ids:
                if item_id in budget.seen_ids:
                    continue
                if budget.fetched == MAX_FETCHES_PER_CITATION:
                    exhausted = True
                    break
                budget.seen_ids.add(item_id)
                budget.fetched += 1
                found, failure = _fetch_evidence(
                    service,
                    item_id=item_id,
                    hit=hit,
                    source=source,
                    locator=locator,
                    source_text=source_text,
                    retrospective_date=retrospective_date,
                    query=query,
                )
                evidence.extend(found)
                if failure is not None:
                    failures.append(failure)
            if exhausted:
                break
        if exhausted:
            attempt_failure = _failure(
                "candidate_limit_reached",
                f"CourtListener body search stopped after {MAX_FETCHES_PER_CITATION} detail fetches",
            )
            break
        if hit_limit_reached or (hits_seen == MAX_HITS_PER_QUERY and page.next is not None):
            attempt_failure = _failure(
                "hit_limit_reached",
                f"CourtListener body search stopped after {MAX_HITS_PER_QUERY} search hits",
            )
            break
        if budget.fetched == MAX_FETCHES_PER_CITATION and (page.next is not None or not evidence):
            attempt_failure = _failure(
                "candidate_limit_reached",
                f"CourtListener body search stopped after {MAX_FETCHES_PER_CITATION} detail fetches",
            )
            break
        if page.next is None:
            break
        if len(pages) == MAX_PAGES_PER_QUERY:
            attempt_failure = _failure(
                "page_limit_reached",
                f"CourtListener body search still has results after {MAX_PAGES_PER_QUERY} pages",
            )
            break
        next_cursor = _next_cursor(page.next)
        if next_cursor is None or next_cursor in seen_cursors:
            attempt_failure = _failure(
                "invalid_pagination", "CourtListener search returned a missing or repeated cursor"
            )
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return (
        BodySearchAttempt(query=query, pages=tuple(pages), failure=attempt_failure),
        tuple(evidence),
        tuple(failures),
    )


def run_courtlistener_body_search(
    document: Document,
    *,
    stage: str,
    source: BodySource,
    retrospective_date: date | None,
    client: CourtListenerBodyClient | None,
) -> Document:
    """Run one source at a time and attach search evidence to each open root."""
    if stage in document.stage_runs:
        raise ValueError(f"Stage already completed: {stage}")
    if "10_roots" not in document.stage_runs:
        raise ValueError("Form roots before body search")

    with ExitStack() as stack:
        service = client
        for root in roots_for_body_search(document):
            recorded = root.record(stage)
            attempts: list[BodySearchAttempt] = []
            evidence: list[BodyEvidence] = []
            failures: list[BodyEvidenceFailure] = []
            try:
                locator = locator_text(root)
            except ValueError as error:
                locator = ""
                failures.append(_failure("unsearchable_locator", str(error)))
            if not locator and not failures:
                failures.append(_failure("unsearchable_locator", "Citation has no searchable locator"))
            if locator:
                budget = _CandidateBudget()
                query = _locator_query(locator)
                if service is None:
                    try:
                        service = stack.enter_context(CourtListenerClient())
                    except CourtListenerError as error:
                        attempts.append(BodySearchAttempt(query=query, failure=_service_failure(error)))
                if service is not None:
                    attempt, found, failed = _search_query(
                        service,
                        query=query,
                        source=source,
                        locator=locator,
                        source_text=document.text,
                        retrospective_date=retrospective_date,
                        budget=budget,
                    )
                    attempts.append(attempt)
                    evidence.extend(found)
                    failures.extend(failed)
            result = BodySearch(
                node_id=recorded.nodes[-1].id,
                source=source,
                retrospective_date=retrospective_date,
                attempts=tuple(attempts),
                evidence=tuple(evidence),
                failures=tuple(failures),
            )
            document = document.replace_citation(recorded.with_body_search(result))
    return document.complete(stage)
