"""Bounded RECAP docket expansion for not-found citation candidates.

This module collects identity evidence; it does not decide that a docket or
document is the cited authority.  Ranking is deterministic and auditable so a
later deliberation layer can evaluate the evidence without repeating broad
searches or downloading every document on a docket.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
import re
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult
from mellea_lrc.retrieval.types import (
    CourtListenerRequestTrace,
    DocketCandidateEvidence,
    DocketDocumentEvidence,
    DocketEvidenceStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CitationRetrievalClient

HTTP_OK = 200
MAX_DECISIONAL_DOCUMENTS = 8
DATE_FLEXIBILITY_DAYS = 366
_YEAR = re.compile(r"^(\d{4})")
_DECISIONAL_PATTERNS = (
    ("memorandum_opinion", re.compile(r"\bmemorandum\s+(?:opinion|decision)\b", re.IGNORECASE)),
    ("opinion", re.compile(r"\bopinion\b", re.IGNORECASE)),
    (
        "findings_and_recommendation",
        re.compile(r"\bfindings?\b.*\brecommendation\b", re.IGNORECASE),
    ),
    (
        "report_and_recommendation",
        re.compile(r"\breport\b.*\brecommendation\b", re.IGNORECASE),
    ),
    ("decision", re.compile(r"\bdecision\b", re.IGNORECASE)),
    ("judgment", re.compile(r"\bjudgment\b", re.IGNORECASE)),
    ("order", re.compile(r"\border\b", re.IGNORECASE)),
)


def expand_docket_evidence(
    *,
    client: CitationRetrievalClient,
    docket_id: str,
    cited_year: str | None,
    cited_date: str | None = None,
) -> DocketCandidateEvidence:
    """Expand one stable docket ID into bounded document-level evidence."""
    get_docket = getattr(client, "get_docket", None)
    search_entries = getattr(client, "search_docket_entries", None)
    search_recap = getattr(client, "search_recap", None)
    if not callable(get_docket) or not callable(search_recap):
        return DocketCandidateEvidence(status=DocketEvidenceStatus.UNAVAILABLE)

    try:
        docket_payload = get_docket(docket_id)
    except CourtListenerError as exc:
        return _courtlistener_failure(exc)
    except (OSError, TypeError, ValueError) as exc:
        return DocketCandidateEvidence(
            status=DocketEvidenceStatus.FAILED,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    docket_trace = _request_trace(docket_payload)
    if docket_trace.http_status != HTTP_OK:
        return DocketCandidateEvidence(
            status=DocketEvidenceStatus.FAILED,
            docket_request=docket_trace,
            error_message=_error_message(docket_payload, "Docket lookup failed."),
        )

    try:
        target_year = _year(cited_year)
        if cited_date is not None:
            date_range = f"[{cited_date} TO {cited_date}]"
        elif target_year is not None:
            date_range = f"[{target_year}-01-01 TO {target_year}-12-31]"
        else:
            date_range = None
        if date_range is not None:
            entries_payload = search_recap(
                f"docket_id:{docket_id} AND "
                f"entry_date_filed:{date_range} AND "
                '("memorandum opinion" OR opinion OR decision OR judgment OR order)'
            )
        elif callable(search_entries):
            entries_payload = search_entries(docket_id, order_by="-date_filed")
        else:
            return _docket_evidence(
                docket_payload,
                status=DocketEvidenceStatus.UNAVAILABLE,
                docket_request=docket_trace,
                error_message="No cited year or docket-entry search method was available.",
            )
    except CourtListenerError as exc:
        return _docket_with_entries_failure(docket_payload, docket_trace, exc)
    except (OSError, TypeError, ValueError) as exc:
        return _docket_evidence(
            docket_payload,
            status=DocketEvidenceStatus.FAILED,
            docket_request=docket_trace,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    entries_trace = _request_trace(entries_payload)
    if entries_trace.http_status != HTTP_OK:
        return _docket_evidence(
            docket_payload,
            status=DocketEvidenceStatus.FAILED,
            docket_request=docket_trace,
            entries_request=entries_trace,
            error_message=_error_message(entries_payload, "Docket-entry lookup failed."),
        )

    documents = _ranked_documents(
        entries_payload,
        cited_year=cited_year,
        cited_date=cited_date,
    )
    if not documents and (cited_date is not None or target_year is not None):
        flexible = _search_flexible_documents(
            search_recap=search_recap,
            docket_id=docket_id,
            cited_year=cited_year,
            cited_date=cited_date,
            target_year=target_year,
        )
        if flexible is not None:
            entries_payload, entries_trace, documents = flexible
    return _docket_evidence(
        docket_payload,
        status=(
            DocketEvidenceStatus.ENRICHED
            if documents
            else DocketEvidenceStatus.NO_DECISIONAL_DOCUMENTS
        ),
        docket_request=docket_trace,
        entries_request=entries_trace,
        documents=documents,
    )


def _ranked_documents(
    payload: object,
    *,
    cited_year: str | None,
    cited_date: str | None = None,
) -> tuple[DocketDocumentEvidence, ...]:
    if isinstance(payload, CourtListenerSearchResult):
        return _ranked_search_documents(payload, cited_year=cited_year, cited_date=cited_date)
    if not isinstance(payload, Mapping):
        return ()
    entries = _document_containers(payload)
    target_year = _year(cited_year)
    evidence: list[DocketDocumentEvidence] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_description = _string(entry, "description", "snippet")
        entry_date = _string(entry, "date_filed", "entry_date_filed")
        documents = entry.get("recap_documents") or entry.get("recapDocuments")
        if not isinstance(documents, list):
            documents = []
        for document in documents:
            if not isinstance(document, Mapping):
                continue
            document_description = _string(document, "description", "snippet")
            document_date = _string(document, "entry_date_filed", "entryDateFiled") or entry_date
            cues = _decisional_cues(entry_description, document_description)
            if not cues:
                continue
            evidence.append(
                DocketDocumentEvidence(
                    docket_entry_id=_string(entry, "cl_docket_entry_id", "id"),
                    recap_document_id=_string(document, "recap_document_id", "id"),
                    entry_number=_string(entry, "entry_number"),
                    document_number=_string(document, "document_number"),
                    date_filed=document_date,
                    entry_description=entry_description,
                    document_description=document_description,
                    page_count=_nonnegative_int(document.get("page_count")),
                    pacer_doc_id=_string(document, "pacer_doc_id"),
                    available=document.get("available") is True,
                    absolute_url=_string(document, "absolute_url"),
                    decisional_cues=cues,
                    year_distance=_year_distance(document_date, target_year),
                )
            )
    evidence.sort(key=lambda document: _document_rank(document, cited_date=cited_date))
    return tuple(evidence[:MAX_DECISIONAL_DOCUMENTS])


def _ranked_search_documents(
    result: CourtListenerSearchResult,
    *,
    cited_year: str | None,
    cited_date: str | None,
) -> tuple[DocketDocumentEvidence, ...]:
    """Rank documents from the validated CourtListener search boundary."""
    target_year = _year(cited_year)
    evidence: list[DocketDocumentEvidence] = []
    for record in result.records:
        for document in record.recap_documents:
            document_description = document.description or document.snippet
            document_date = document.entry_date_filed or record.date_filed
            cues = _decisional_cues(record.snippet, document_description)
            if not cues:
                continue
            evidence.append(
                DocketDocumentEvidence(
                    recap_document_id=document.recap_document_id,
                    entry_number=document.entry_number,
                    document_number=document.document_number,
                    date_filed=document_date,
                    entry_description=record.snippet,
                    document_description=document_description,
                    page_count=document.page_count,
                    pacer_doc_id=document.pacer_doc_id,
                    available=document.available,
                    absolute_url=document.absolute_url,
                    decisional_cues=cues,
                    year_distance=_year_distance(document_date, target_year),
                )
            )
    evidence.sort(key=lambda document: _document_rank(document, cited_date=cited_date))
    return tuple(evidence[:MAX_DECISIONAL_DOCUMENTS])


def _decisional_query(docket_id: str, date_range: str) -> str:
    return (
        f"docket_id:{docket_id} AND entry_date_filed:{date_range} AND "
        '("memorandum opinion" OR opinion OR decision OR judgment OR order)'
    )


def _flexible_date_range(*, cited_date: str | None, target_year: int | None) -> str:
    if cited_date is not None:
        asserted_date = date.fromisoformat(cited_date)
        lower = asserted_date - timedelta(days=DATE_FLEXIBILITY_DAYS)
        upper = asserted_date + timedelta(days=DATE_FLEXIBILITY_DAYS)
        return f"[{lower.isoformat()} TO {upper.isoformat()}]"
    assert target_year is not None
    return f"[{target_year - 1}-01-01 TO {target_year + 1}-12-31]"


def _search_flexible_documents(
    *,
    search_recap: object,
    docket_id: str,
    cited_year: str | None,
    cited_date: str | None,
    target_year: int | None,
) -> tuple[object, CourtListenerRequestTrace, tuple[DocketDocumentEvidence, ...]] | None:
    if not callable(search_recap):
        return None
    try:
        payload = search_recap(
            _decisional_query(
                docket_id,
                _flexible_date_range(cited_date=cited_date, target_year=target_year),
            )
        )
    except (CourtListenerError, OSError, TypeError, ValueError):
        return None
    trace = _request_trace(payload)
    if trace.http_status != HTTP_OK:
        return None
    documents = _ranked_documents(payload, cited_year=cited_year, cited_date=cited_date)
    return (payload, trace, documents) if documents else None


def _document_rank(
    document: DocketDocumentEvidence,
    *,
    cited_date: str | None,
) -> tuple[object, ...]:
    cue_priority = min((_cue_priority(cue) for cue in document.decisional_cues), default=99)
    return (
        0 if cited_date is not None and document.date_filed == cited_date else 1,
        cue_priority,
        document.year_distance if document.year_distance is not None else 9999,
        0 if document.available else 1,
        document.date_filed or "9999",
        document.entry_number or "",
    )


def _document_containers(payload: Mapping[object, object]) -> list[Mapping[object, object]]:
    for key in ("entries", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    raw = payload.get("raw")
    if isinstance(raw, Mapping):
        results = raw.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, Mapping)]
    return []


def _cue_priority(cue: str) -> int:
    order = {name: index for index, (name, _) in enumerate(_DECISIONAL_PATTERNS)}
    return order.get(cue, len(order))


def _decisional_cues(*descriptions: str | None) -> tuple[str, ...]:
    text = " ".join(description for description in descriptions if description)
    return tuple(name for name, pattern in _DECISIONAL_PATTERNS if pattern.search(text))


def _year_distance(date_filed: str | None, target_year: int | None) -> int | None:
    filed_year = _year(date_filed)
    if filed_year is None or target_year is None:
        return None
    return abs(filed_year - target_year)


def _year(value: str | None) -> int | None:
    if not value:
        return None
    match = _YEAR.match(value)
    return int(match.group(1)) if match else None


def _docket_evidence(
    payload: Mapping[str, object],
    *,
    status: DocketEvidenceStatus,
    docket_request: CourtListenerRequestTrace,
    entries_request: CourtListenerRequestTrace | None = None,
    documents: tuple[DocketDocumentEvidence, ...] = (),
    error_message: str | None = None,
) -> DocketCandidateEvidence:
    return DocketCandidateEvidence(
        status=status,
        docket_request=docket_request,
        entries_request=entries_request or CourtListenerRequestTrace(),
        case_name=_string(payload, "case_name"),
        court_id=_string(payload, "court_id"),
        docket_number=_string(payload, "docket_number"),
        date_filed=_string(payload, "date_filed"),
        date_terminated=_string(payload, "date_terminated"),
        assigned_to=_string(payload, "assigned_to_str"),
        referred_to=_string(payload, "referred_to_str"),
        nature_of_suit=_string(payload, "nature_of_suit"),
        cause=_string(payload, "cause"),
        jurisdiction_type=_string(payload, "jurisdiction_type"),
        documents=documents,
        error_message=error_message,
    )


def _courtlistener_failure(exc: CourtListenerError) -> DocketCandidateEvidence:
    return DocketCandidateEvidence(
        status=DocketEvidenceStatus.FAILED,
        docket_request=CourtListenerRequestTrace(
            http_status=exc.upstream_status_code,
            key=exc.cache_key,
            error_message=f"{type(exc).__name__}: {exc}",
        ),
        error_message=f"{type(exc).__name__}: {exc}",
    )


def _docket_with_entries_failure(
    payload: Mapping[str, object],
    trace: CourtListenerRequestTrace,
    exc: CourtListenerError,
) -> DocketCandidateEvidence:
    return _docket_evidence(
        payload,
        status=DocketEvidenceStatus.FAILED,
        docket_request=trace,
        entries_request=CourtListenerRequestTrace(
            http_status=exc.upstream_status_code,
            key=exc.cache_key,
            error_message=f"{type(exc).__name__}: {exc}",
        ),
        error_message=f"{type(exc).__name__}: {exc}",
    )


def _request_trace(payload: object) -> CourtListenerRequestTrace:
    if isinstance(payload, CourtListenerSearchResult):
        return CourtListenerRequestTrace(
            http_status=payload.http_status,
            cache=payload.cache,
            key=payload.key,
            error_message=payload.error_message,
        )
    if not isinstance(payload, Mapping):
        return CourtListenerRequestTrace(error_message="Response was not an object.")
    status = payload.get("http_status")
    return CourtListenerRequestTrace(
        http_status=status if isinstance(status, int) and not isinstance(status, bool) else None,
        cache=_string(payload, "cache"),
        key=_string(payload, "key"),
        error_message=None,
    )


def _error_message(payload: object, fallback: str) -> str:
    if isinstance(payload, CourtListenerSearchResult):
        return payload.error_message or fallback
    if isinstance(payload, Mapping):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, Mapping):
            message = detail.get("message")
            if isinstance(message, str) and message:
                return message
    return fallback


def _string(payload: Mapping[object, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
