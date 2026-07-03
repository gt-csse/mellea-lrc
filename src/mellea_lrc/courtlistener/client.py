"""CourtListener REST client, response normalization, and rate limiting."""

# ruff: noqa: ANN401, D101, D102, EM101, FBT001, FBT003, PLR2004, TC003, TRY003

from __future__ import annotations

import hashlib
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import gmtime, monotonic, sleep, strftime
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests

from mellea_lrc.courtlistener.cache import CacheEntry, CacheStore, NullCache
from mellea_lrc.courtlistener.lookup import (
    normalize_citation_lookup_payload,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.types import CourtListenerCitationLookup


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class CourtListenerConfig:
    base_url: str = "https://www.courtlistener.com/api/rest/v4/"
    tokens: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> CourtListenerConfig:
        tokens = tuple(
            value.strip()
            for key, value in sorted(os.environ.items())
            if key.startswith("COURTLISTENER_API_TOKEN") and value.strip()
        )
        return cls(
            base_url=os.getenv(
                "COURTLISTENER_BASE_URL",
                os.getenv("COURTLISTENER_API_BASE", "https://www.courtlistener.com/api/rest/v4/"),
            ),
            tokens=tokens,
        )


class CourtListenerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        status_code: int = 502,
        upstream_status_code: int | None = None,
        retryable: bool = False,
        url: str | None = None,
        cache_key: str | None = None,
        retry_after_seconds: float | None = None,
        upstream_detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.failure_type = failure_type
        self.status_code = status_code
        self.upstream_status_code = upstream_status_code
        self.retryable = retryable
        self.url = url
        self.cache_key = cache_key
        self.retry_after_seconds = retry_after_seconds
        self.upstream_detail = upstream_detail

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "failure_type": self.failure_type,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.upstream_status_code is not None:
            payload["upstream_status_code"] = self.upstream_status_code
        if self.cache_key is not None:
            payload["key"] = self.cache_key
        if self.url is not None:
            payload["url"] = self.url
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        if self.upstream_detail is not None:
            payload["upstream_detail"] = self.upstream_detail
        return payload


@dataclass(frozen=True)
class CourtListenerRateLimitConfig:
    enabled: bool = True
    per_minute: int = 5
    per_hour: int = 50
    per_day: int = 125
    max_wait_seconds: float = 115.0

    @classmethod
    def from_env(cls) -> CourtListenerRateLimitConfig:
        return cls(
            enabled=_env_bool("COURTLISTENER_RATE_LIMIT_ENABLED", True),
            per_minute=_env_int("COURTLISTENER_RATE_LIMIT_PER_MINUTE", 5),
            per_hour=_env_int("COURTLISTENER_RATE_LIMIT_PER_HOUR", 50),
            per_day=_env_int("COURTLISTENER_RATE_LIMIT_PER_DAY", 125),
            max_wait_seconds=_env_float("COURTLISTENER_RATE_LIMIT_MAX_WAIT_SECONDS", 115.0),
        )

    def windows(self) -> tuple[tuple[int, float], ...]:
        return tuple(
            (limit, seconds)
            for limit, seconds in (
                (self.per_minute, 60.0),
                (self.per_hour, 3600.0),
                (self.per_day, 86400.0),
            )
            if limit > 0
        )


class CourtListenerRateLimiter:
    def __init__(
        self,
        config: CourtListenerRateLimitConfig | None = None,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self.config = config or CourtListenerRateLimitConfig.from_env()
        self._clock = clock
        self._sleeper = sleeper
        self._lock = Lock()
        self._timestamps: dict[str, dict[float, deque[float]]] = defaultdict(dict)

    def acquire(self, bucket: str, *, url: str, cache_key: str) -> None:
        if not self.config.enabled:
            return

        windows = self.config.windows()
        if not windows:
            return

        while True:
            with self._lock:
                wait_seconds = self._reserve_or_wait(bucket, windows, self._clock())
            if wait_seconds <= 0:
                return
            if wait_seconds > self.config.max_wait_seconds:
                raise CourtListenerError(
                    "CourtListener client-side rate limit is currently exhausted",
                    failure_type="api_limit",
                    status_code=429,
                    retryable=True,
                    url=url,
                    cache_key=cache_key,
                    retry_after_seconds=round(wait_seconds, 3),
                )
            self._sleeper(wait_seconds)

    def _reserve_or_wait(
        self,
        bucket: str,
        windows: tuple[tuple[int, float], ...],
        now: float,
    ) -> float:
        bucket_timestamps = self._timestamps[bucket]
        wait_seconds = 0.0

        for limit, window_seconds in windows:
            timestamps = bucket_timestamps.setdefault(window_seconds, deque())
            cutoff = now - window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= limit:
                wait_seconds = max(wait_seconds, timestamps[0] + window_seconds - now)

        if wait_seconds > 0:
            return wait_seconds

        for _, window_seconds in windows:
            bucket_timestamps[window_seconds].append(now)
        return 0.0


class CourtListenerClient:
    def __init__(
        self,
        config: CourtListenerConfig | None = None,
        cache: CacheStore | None = None,
        session: requests.Session | None = None,
        rate_limiter: CourtListenerRateLimiter | None = None,
    ) -> None:
        self.config = config or CourtListenerConfig.from_env()
        self.cache = cache or NullCache()
        self.session = session or requests.Session()
        self.rate_limiter = rate_limiter or CourtListenerRateLimiter()

    def resolve_docket(
        self,
        court_id: str,
        docket_number: str,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        result = self.get(
            "dockets",
            params={"court": court_id, "docket_number": docket_number, "cursor": cursor},
        )
        raw = result["response"]
        return {
            **_request_metadata(result),
            "court_id": court_id,
            "docket_number": docket_number,
            "candidates": [_normalize_docket(item) for item in _results(raw)],
            **_pagination_metadata(raw),
            "raw": raw,
        }

    def get_docket(self, cl_docket_id: int | str) -> dict[str, Any]:
        result = self.get(f"dockets/{cl_docket_id}")
        raw = result["response"]
        return {
            **_request_metadata(result),
            **_normalize_docket(raw),
            "raw": raw,
        }

    def search_docket_entries(
        self,
        cl_docket_id: int | str,
        entry_number: int | str | None = None,
        cursor: str | None = None,
        order_by: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "docket": cl_docket_id,
            "entry_number": entry_number,
            "cursor": cursor,
            "order_by": order_by,
        }
        result = self.get(
            "docket-entries",
            params=params,
        )
        raw = result["response"]
        response = {
            **_request_metadata(result),
            "cl_docket_id": _coerce_int(cl_docket_id),
            "entries": [_normalize_docket_entry(item) for item in _results(raw)],
            **_pagination_metadata(raw),
            "raw": raw,
        }
        if entry_number is not None:
            response["entry_number"] = str(entry_number)
        return response

    def search_recap_documents(
        self,
        cl_docket_entry_id: int | str | None = None,
        cl_docket_id: int | str | None = None,
        entry_number: int | str | None = None,
        cursor: str | None = None,
        order_by: str | None = None,
    ) -> dict[str, Any]:
        if cl_docket_entry_id is not None:
            result = self.get(
                "recap-documents",
                params={"docket_entry": cl_docket_entry_id, "cursor": cursor},
            )
            raw = result["response"]
            return {
                **_request_metadata(result),
                "cl_docket_entry_id": _coerce_int(cl_docket_entry_id),
                "documents": [_normalize_recap_document(item) for item in _results(raw)],
                **_pagination_metadata(raw),
                "raw": raw,
            }

        if cl_docket_id is None:
            raise ValueError("Provide cl_docket_entry_id or cl_docket_id")

        entries = self.search_docket_entries(
            cl_docket_id,
            entry_number,
            cursor=cursor,
            order_by=order_by,
        )
        documents = [
            document for entry in entries["entries"] for document in entry.get("recap_documents", [])
        ]
        response = {
            "cache": entries["cache"],
            "key": entries["key"],
            "cl_docket_id": entries["cl_docket_id"],
            "documents": documents,
            "next_cursor": entries["next_cursor"],
            "previous_cursor": entries["previous_cursor"],
            "raw": entries["raw"],
        }
        if entry_number is not None:
            response["entry_number"] = str(entry_number)
        return response

    def get_recap_document(self, recap_document_id: int | str) -> dict[str, Any]:
        result = self.get(f"recap-documents/{recap_document_id}")
        raw = result["response"]
        return {
            **_request_metadata(result),
            **_normalize_recap_document(raw),
            "raw": raw,
        }

    def get_recap_document_download_url(self, recap_document_id: int | str) -> dict[str, Any]:
        document = self.get_recap_document(recap_document_id)
        return {
            "cache": document["cache"],
            "key": document["key"],
            **_download_url_payload(document),
            "raw": document["raw"],
        }

    def list_courts(self) -> dict[str, Any]:
        result = self.get("courts")
        raw = result["response"]
        return {
            **_request_metadata(result),
            "courts": [_normalize_court(item) for item in _results(raw)],
            "raw": raw,
        }

    def get_court(self, court_id: str) -> dict[str, Any]:
        result = self.get(f"courts/{court_id}")
        raw = result["response"]
        return {
            **_request_metadata(result),
            **_normalize_court(raw),
            "raw": raw,
        }

    def search(
        self,
        q: str,
        search_type: str,
        cursor: str | None = None,
        *,
        semantic: bool = False,
    ) -> dict[str, Any]:
        if search_type not in {"r", "rd", "d", "o"}:
            raise ValueError("type must be one of: r, rd, d, o")
        params = {"q": q, "type": search_type, "cursor": cursor}
        if semantic:
            params["semantic"] = "true"
        result = self.get("search", params=params)
        raw = result["response"]
        return {
            **_request_metadata(result),
            "http_status": result["status"],
            "q": q,
            "type": search_type,
            "semantic": semantic,
            "count": raw.get("count") if isinstance(raw, dict) else None,
            "results": [_normalize_search_result(item, search_type) for item in _results(raw)],
            **_pagination_metadata(raw),
            "raw": raw,
        }

    def search_opinions(self, q: str) -> dict[str, Any]:
        """Run an opinion (``type=o``) relevance search; response carries ``count``."""
        return self.search(q, "o")

    def search_recap(self, q: str) -> dict[str, Any]:
        """Run a RECAP (``type=r``) relevance search; response carries ``count``."""
        return self.search(q, "r")

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        result = self.post(
            "citation-lookup",
            data={"volume": volume, "reporter": reporter, "page": page},
        )
        return normalize_citation_lookup_payload(result, volume, reporter, page)

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        return self._request("POST", endpoint, data=data)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        endpoint = _normalize_endpoint(endpoint)
        params = _clean(params)
        data = _clean(data)
        key = _cache_key(method, endpoint, params, data)

        cached = self.cache.get(key)
        if cached is not None:
            return {
                "cache": "hit",
                "key": key,
                "status": cached.status_code,
                "response": cached.response,
            }

        response = self._send(method, endpoint, params, data, key)
        try:
            payload = response.json()
        except ValueError as exc:
            raise CourtListenerError(
                "CourtListener returned a non-JSON response",
                failure_type="upstream_invalid_json",
                status_code=502,
                upstream_status_code=response.status_code,
                retryable=True,
                url=response.url,
                cache_key=key,
                upstream_detail=response.text[:500],
            ) from exc
        self.cache.put(
            CacheEntry(
                key=key,
                method=method,
                endpoint=endpoint,
                params=params,
                data=data,
                status_code=response.status_code,
                url=response.url,
                cached_at=strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
                response=payload,
            )
        )
        return {
            "cache": "miss",
            "key": key,
            "status": response.status_code,
            "response": payload,
        }

    def _send(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any],
        data: dict[str, Any],
        key: str,
    ) -> requests.Response:
        url = urljoin(self.config.base_url.rstrip("/") + "/", endpoint)
        attempts = max(1, len(self.config.tokens))
        last_error: CourtListenerError | None = None

        for token_index in range(attempts):
            self.rate_limiter.acquire(
                _rate_limit_bucket(self.config.tokens),
                url=url,
                cache_key=key,
            )
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params or None,
                    data=data or None,
                    headers=self._headers(token_index),
                    timeout=45,
                )
            except requests.Timeout as exc:
                raise CourtListenerError(
                    "CourtListener request timed out",
                    failure_type="upstream_timeout",
                    status_code=504,
                    retryable=True,
                    url=url,
                    cache_key=key,
                ) from exc
            except requests.RequestException as exc:
                raise CourtListenerError(
                    "CourtListener request failed before a response was received",
                    failure_type="upstream_request_error",
                    status_code=502,
                    retryable=True,
                    url=url,
                    cache_key=key,
                    upstream_detail=str(exc),
                ) from exc
            if response.status_code == 429 and token_index + 1 < attempts:
                last_error = _courtlistener_http_error(method, response, key)
                continue
            if response.status_code >= 400:
                raise _courtlistener_http_error(method, response, key)
            return response

        raise last_error or CourtListenerError(
            f"CourtListener {method} failed",
            failure_type="upstream_error",
            status_code=502,
            retryable=True,
            url=url,
            cache_key=key,
        )

    def _headers(self, token_index: int) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
        if self.config.tokens:
            headers["Authorization"] = f"Token {self.config.tokens[token_index % len(self.config.tokens)]}"
        return headers


def _normalize_endpoint(endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        endpoint = urlparse(endpoint).path
    endpoint = endpoint.strip("/")
    if endpoint.startswith("api/rest/v4/"):
        endpoint = endpoint.removeprefix("api/rest/v4/")
    return f"{endpoint}/"


def _clean(values: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (values or {}).items() if value not in (None, "")}


def _cache_key(
    method: str,
    endpoint: str,
    params: dict[str, Any],
    data: dict[str, Any],
) -> str:
    raw = "|".join(
        [
            method.upper(),
            endpoint,
            urlencode(sorted(params.items()), doseq=True),
            urlencode(sorted(data.items()), doseq=True),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _courtlistener_http_error(
    method: str,
    response: requests.Response,
    key: str,
) -> CourtListenerError:
    failure_type = _failure_type_for_status(response.status_code)
    return CourtListenerError(
        f"CourtListener {method} failed with {response.status_code}",
        failure_type=failure_type,
        status_code=_public_status_for_failure(failure_type, response.status_code),
        upstream_status_code=response.status_code,
        retryable=failure_type in {"api_limit", "upstream_error"},
        url=response.url,
        cache_key=key,
        upstream_detail=_response_detail(response),
    )


def _failure_type_for_status(status_code: int) -> str:
    if status_code == 429:
        return "api_limit"
    if status_code in {401, 403}:
        return "upstream_auth"
    if status_code == 404:
        return "upstream_not_found"
    if 400 <= status_code < 500:
        return "upstream_bad_request"
    return "upstream_error"


def _public_status_for_failure(failure_type: str, upstream_status_code: int) -> int:
    if failure_type == "api_limit":
        return 429
    if failure_type == "upstream_bad_request":
        return 400
    if failure_type == "upstream_not_found":
        return 404
    if failure_type == "upstream_auth":
        return 502
    if 500 <= upstream_status_code < 600:
        return 502
    return 502


def _response_detail(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:500]


def _rate_limit_bucket(tokens: tuple[str, ...]) -> str:
    if tokens:
        return "authenticated"
    return "anonymous"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _request_metadata(result: dict[str, Any]) -> dict[str, Any]:
    return {"cache": result["cache"], "key": result["key"]}


def _pagination_metadata(raw: Any) -> dict[str, str | None]:
    return {
        "next_cursor": _cursor_from_url(raw.get("next") if isinstance(raw, dict) else None),
        "previous_cursor": _cursor_from_url(raw.get("previous") if isinstance(raw, dict) else None),
    }


def _cursor_from_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    cursors = parse_qs(urlparse(value).query).get("cursor")
    if not cursors:
        return None
    return cursors[0]


def _results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        return [item for item in raw["results"] if isinstance(item, dict)]
    return []


def _normalize_docket(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "cl_docket_id": raw.get("id"),
        "court_id": raw.get("court_id") or _last_path_part(raw.get("court")),
        "docket_number": raw.get("docket_number"),
        "case_name": raw.get("case_name"),
        "date_filed": raw.get("date_filed"),
        "date_terminated": raw.get("date_terminated"),
        "assigned_to_str": raw.get("assigned_to_str"),
        "referred_to_str": raw.get("referred_to_str"),
        "nature_of_suit": raw.get("nature_of_suit"),
        "cause": raw.get("cause"),
        "jury_demand": raw.get("jury_demand"),
        "jurisdiction_type": raw.get("jurisdiction_type"),
        "absolute_url": raw.get("absolute_url"),
        "resource_uri": raw.get("resource_uri"),
    }


def _normalize_docket_entry(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "cl_docket_entry_id": raw.get("id"),
        "cl_docket_id": _extract_id(raw.get("docket")),
        "entry_number": _string_or_none(raw.get("entry_number")),
        "date_filed": raw.get("date_filed"),
        "description": raw.get("description"),
        "recap_documents": [
            _normalize_recap_document(item)
            for item in raw.get("recap_documents", [])
            if isinstance(item, dict)
        ],
        "resource_uri": raw.get("resource_uri"),
    }


def _normalize_recap_document(raw: dict[str, Any]) -> dict[str, Any]:
    filepath_local = raw.get("filepath_local")
    filepath_ia = raw.get("filepath_ia")
    return {
        "recap_document_id": raw.get("id"),
        "cl_docket_entry_id": _extract_id(raw.get("docket_entry")),
        "document_number": _string_or_none(raw.get("document_number")),
        "attachment_number": _string_or_none(raw.get("attachment_number")),
        "description": raw.get("description"),
        "page_count": raw.get("page_count"),
        "pacer_doc_id": raw.get("pacer_doc_id"),
        "filepath_local": filepath_local,
        "filepath_ia": filepath_ia,
        "filepath_ia_json": raw.get("filepath_ia_json"),
        "absolute_url": raw.get("absolute_url"),
        "resource_uri": raw.get("resource_uri"),
        "available": bool(filepath_local or filepath_ia),
    }


def _normalize_court(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "court_id": raw.get("id"),
        "full_name": raw.get("full_name"),
        "short_name": raw.get("short_name"),
        "citation_string": raw.get("citation_string"),
        "jurisdiction": raw.get("jurisdiction"),
        "start_date": raw.get("start_date"),
        "end_date": raw.get("end_date"),
        "url": raw.get("url"),
        "resource_uri": raw.get("resource_uri"),
    }


def _download_url_payload(document: dict[str, Any]) -> dict[str, Any]:
    filepath_local = document.get("filepath_local")
    filepath_ia = document.get("filepath_ia")
    if filepath_local:
        return {
            "recap_document_id": document.get("recap_document_id"),
            "available": True,
            "source": "courtlistener_storage",
            "download_url": urljoin("https://storage.courtlistener.com/", str(filepath_local)),
            "filepath_local": filepath_local,
            "filepath_ia": filepath_ia,
        }
    if filepath_ia:
        return {
            "recap_document_id": document.get("recap_document_id"),
            "available": True,
            "source": "internet_archive",
            "download_url": filepath_ia,
            "filepath_local": filepath_local,
            "filepath_ia": filepath_ia,
        }
    return {
        "recap_document_id": document.get("recap_document_id"),
        "available": False,
        "source": "none",
        "download_url": None,
        "filepath_local": filepath_local,
        "filepath_ia": filepath_ia,
    }


def _normalize_search_result(raw: dict[str, Any], search_type: str) -> dict[str, Any]:
    if search_type in {"r", "d"}:
        normalized = {
            "cl_docket_id": _first_present(raw, "docket_id", "docketId", "id"),
            "court_id": _first_present(raw, "court_id", "courtId", "court"),
            "docket_number": _first_present(raw, "docketNumber", "docket_number"),
            "case_name": _first_present(raw, "caseName", "case_name"),
            "date_filed": _first_present(raw, "dateFiled", "date_filed"),
            "date_terminated": _first_present(raw, "dateTerminated", "date_terminated"),
            "absolute_url": _first_present(raw, "absolute_url", "absoluteUrl"),
            "snippet": raw.get("snippet"),
            "resource_uri": raw.get("resource_uri"),
        }
        if search_type == "r":
            raw_documents = raw.get("recap_documents") or raw.get("recapDocuments") or []
            normalized["recap_documents"] = [
                _normalize_search_recap_document(item) for item in raw_documents if isinstance(item, dict)
            ]
            normalized["more_docs"] = _first_present(raw, "more_docs", "moreDocs")
        return normalized

    if search_type == "rd":
        return _normalize_search_recap_document(raw)

    return {
        "cluster_id": _first_present(raw, "cluster_id", "clusterId", "id"),
        "case_name": _first_present(raw, "caseName", "case_name"),
        "court_id": _first_present(raw, "court_id", "courtId", "court"),
        "date_filed": _first_present(raw, "dateFiled", "date_filed"),
        "absolute_url": _first_present(raw, "absolute_url", "absoluteUrl"),
        "snippet": raw.get("snippet"),
        "resource_uri": raw.get("resource_uri"),
    }


def _normalize_search_recap_document(raw: dict[str, Any]) -> dict[str, Any]:
    filepath_local = raw.get("filepath_local")
    filepath_ia = raw.get("filepath_ia")
    return {
        "recap_document_id": _first_present(raw, "recap_document_id", "recapDocumentId", "id"),
        "cl_docket_id": _first_present(raw, "docket_id", "docketId"),
        "entry_number": _string_or_none(_first_present(raw, "entry_number", "entryNumber")),
        "document_number": _string_or_none(_first_present(raw, "document_number", "documentNumber")),
        "attachment_number": _string_or_none(_first_present(raw, "attachment_number", "attachmentNumber")),
        "description": raw.get("description"),
        "entry_date_filed": _first_present(raw, "entry_date_filed", "entryDateFiled"),
        "pacer_doc_id": _first_present(raw, "pacer_doc_id", "pacerDocId"),
        "filepath_local": filepath_local,
        "filepath_ia": filepath_ia,
        "absolute_url": _first_present(raw, "absolute_url", "absoluteUrl"),
        "snippet": raw.get("snippet"),
        "available": bool(filepath_local or filepath_ia),
    }


def _first_present(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _extract_id(value: Any) -> int | str | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        part = _last_path_part(value)
        return _coerce_int(part) if part is not None else None
    return None


def _last_path_part(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.rstrip("/").rsplit("/", 1)[-1]


def _coerce_int(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
