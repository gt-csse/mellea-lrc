"""Normalization and serialization for CourtListener Opinion and RECAP search."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import parse_qs, urlparse

from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.courtlistener.search_models import (
    CourtListenerRecapDocumentRecord,
    CourtListenerSearchRecord,
    CourtListenerSearchResult,
)
from mellea_lrc.courtlistener.search_transport import CourtListenerSearchResponsePayload

HTTP_OK = 200
SEARCH_TYPES = frozenset({"r", "rd", "d", "o"})


def normalize_search_payload(
    payload: object,
    *,
    query: str,
    search_type: str,
    semantic: bool = False,
) -> CourtListenerSearchResult:
    """Validate an external search payload and convert it to domain records."""
    if search_type not in SEARCH_TYPES:
        msg = "type must be one of: r, rd, d, o"
        raise ValueError(msg)

    envelope = payload if isinstance(payload, Mapping) else {}
    response_value = envelope.get("response", envelope)
    response = dict(response_value) if isinstance(response_value, Mapping) else {}
    raw_value = response.pop("raw", None)
    explicit_extra_value = response.pop("extra_data", None)
    if isinstance(raw_value, Mapping):
        response = {**raw_value, **response}

    validated = CourtListenerSearchResponsePayload.model_validate(response)
    http_status = _int_or_none(envelope.get("status"))
    if http_status is None:
        http_status = validated.http_status
    if http_status is None and validated.detail is None:
        http_status = HTTP_OK

    response_extra = validated.collected_extra_data().to_dict()
    envelope_extra = {
        key: value
        for key, value in envelope.items()
        if key not in {"response", "status", "cache", "key"}
    }
    if envelope is response_value:
        envelope_extra = {}
    explicit_extra = (
        dict(explicit_extra_value) if isinstance(explicit_extra_value, Mapping) else {}
    )
    extra_data: dict[str, object] = (
        explicit_extra
        if set(explicit_extra).issubset({"response", "envelope"})
        else ({"response": explicit_extra} if explicit_extra else {})
    )
    if response_extra:
        existing_response_extra = extra_data.get("response")
        extra_data["response"] = {
            **(existing_response_extra if isinstance(existing_response_extra, dict) else {}),
            **response_extra,
        }
    if envelope_extra:
        extra_data["envelope"] = envelope_extra

    return CourtListenerSearchResult(
        query=query,
        search_type=search_type,
        semantic=semantic,
        http_status=http_status,
        count=validated.count,
        records=tuple(item.to_domain(search_type) for item in validated.results),
        next_cursor=validated.next_cursor or _cursor_from_url(validated.next),
        previous_cursor=validated.previous_cursor or _cursor_from_url(validated.previous),
        cache=_string_or_none(envelope.get("cache")) or validated.cache,
        key=_string_or_none(envelope.get("key")) or validated.key,
        error_message=_error_message(validated.detail),
        extra_data=ExtraData(extra_data),
    )


def search_result_dict(result: CourtListenerSearchResult) -> dict[str, object]:
    """Convert a search result into the deployed-service JSON contract."""
    payload: dict[str, object] = {
        "q": result.query,
        "type": result.search_type,
        "semantic": result.semantic,
        "http_status": result.http_status,
        "count": result.count,
        "results": [_search_record_dict(record) for record in result.records],
        "next_cursor": result.next_cursor,
        "previous_cursor": result.previous_cursor,
    }
    if result.cache is not None:
        payload["cache"] = result.cache
    if result.key is not None:
        payload["key"] = result.key
    if result.error_message is not None:
        payload["detail"] = result.error_message
    if result.extra_data:
        payload["extra_data"] = result.extra_data.to_dict()
    return payload


def _search_record_dict(record: CourtListenerSearchRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "cluster_id": record.cluster_id,
        "docket_id": record.docket_id,
        "court_id": record.court_id,
        "docket_number": record.docket_number,
        "case_name": record.case_name,
        "date_filed": record.date_filed,
        "date_terminated": record.date_terminated,
        "absolute_url": record.absolute_url,
        "snippet": record.snippet,
        "resource_uri": record.resource_uri,
        "recap_documents": [_recap_document_dict(item) for item in record.recap_documents],
        "more_docs": record.more_docs,
    }
    if record.extra_data:
        payload["extra_data"] = record.extra_data.to_dict()
    return payload


def _recap_document_dict(document: CourtListenerRecapDocumentRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "recap_document_id": document.recap_document_id,
        "docket_id": document.docket_id,
        "entry_number": document.entry_number,
        "document_number": document.document_number,
        "attachment_number": document.attachment_number,
        "description": document.description,
        "entry_date_filed": document.entry_date_filed,
        "pacer_doc_id": document.pacer_doc_id,
        "filepath_local": document.filepath_local,
        "filepath_ia": document.filepath_ia,
        "absolute_url": document.absolute_url,
        "snippet": document.snippet,
        "page_count": document.page_count,
        "available": document.available,
    }
    if document.extra_data:
        payload["extra_data"] = document.extra_data.to_dict()
    return payload


def _cursor_from_url(value: str | None) -> str | None:
    if not value:
        return None
    cursors = parse_qs(urlparse(value).query).get("cursor")
    return cursors[0] if cursors else None


def _error_message(detail: str | dict[str, object] | None) -> str | None:
    if isinstance(detail, str):
        return detail or None
    if isinstance(detail, dict):
        message = detail.get("message")
        return message if isinstance(message, str) and message else str(detail)
    return None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
