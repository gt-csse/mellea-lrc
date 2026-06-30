"""HTTP client for a deployed CourtListener access service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from typing import TYPE_CHECKING
from urllib import error, parse, request

from mellea_lrc.courtlistener.lookup import normalize_citation_lookup_payload

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.types import CourtListenerCitationLookup

CL_ACCESS_URL_ENV = "CL_ACCESS_MODAL_URL"

PostJson = Callable[[str, Mapping[str, str]], object]
GetJson = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class CourtListenerAccessConfig:
    """Configuration for a deployed CourtListener access service."""

    base_url: str

    @classmethod
    def from_env(cls) -> CourtListenerAccessConfig:
        """Load service configuration from environment variables."""
        base_url = os.environ.get(CL_ACCESS_URL_ENV, "").strip()
        if not base_url:
            msg = f"{CL_ACCESS_URL_ENV} must be set to the cl-access service URL"
            raise ValueError(msg)
        return cls(base_url=base_url)


class CourtListenerAccessClient:
    """Small HTTP wrapper around a deployed CourtListener access service."""

    def __init__(
        self,
        config: CourtListenerAccessConfig | None = None,
        *,
        post_json: PostJson | None = None,
        get_json: GetJson | None = None,
    ) -> None:
        self.config = config or CourtListenerAccessConfig.from_env()
        self._post_json = post_json or _post_json
        self._get_json = get_json or _get_json

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Look up a reporter citation through the remote service."""
        url = f"{self.config.base_url.rstrip('/')}/citation-lookup"
        _validate_http_url(url)
        payload = self._post_json(
            url,
            {"volume": volume, "reporter": reporter, "page": page},
        )
        return normalize_citation_lookup_payload(payload, volume, reporter, page)

    def get_docket(self, cl_docket_id: int | str) -> Mapping[str, object]:
        """Retrieve one docket through the remote access service."""
        url = f"{self.config.base_url.rstrip('/')}/dockets/{cl_docket_id}"
        _validate_http_url(url)
        payload = self._get_json(url)
        return payload if isinstance(payload, Mapping) else {}


def _get_json(url: str) -> object:
    req = request.Request(  # noqa: S310
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=45) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"detail": _json_detail(detail) or detail}
    except (OSError, json.JSONDecodeError) as exc:
        return {"detail": str(exc)}


def _post_json(url: str, data: Mapping[str, str]) -> object:
    encoded = parse.urlencode(data).encode("utf-8")
    req = request.Request(  # noqa: S310
        url,
        data=encoded,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=45) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        limit_detail = _json_detail(detail)
        return {
            "response": {
                "citation": " ".join(data[key] for key in ("volume", "reporter", "page")),
                "status": exc.code,
                "error_message": _error_message(limit_detail, detail),
                "limit_detail": limit_detail,
                "clusters": [],
            }
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "response": {
                "citation": " ".join(data[key] for key in ("volume", "reporter", "page")),
                "status": 502,
                "error_message": str(exc),
                "clusters": [],
            }
        }


def _validate_http_url(url: str) -> None:
    scheme = parse.urlparse(url).scheme
    if scheme not in {"http", "https"}:
        msg = f"cl-access URL must be http(s), got: {scheme or '<empty>'}"
        raise ValueError(msg)


def _json_detail(raw: str) -> dict[str, object] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail", payload)
    return detail if isinstance(detail, dict) else payload


def _error_message(detail: dict[str, object] | None, fallback: str) -> str:
    if detail is None:
        return fallback
    message = detail.get("message")
    if isinstance(message, str) and message:
        return message
    failure_type = detail.get("failure_type")
    if isinstance(failure_type, str) and failure_type:
        return failure_type
    return fallback
