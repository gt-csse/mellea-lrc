"""HTTP client for the deployed CourtListener access service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from urllib import error, parse, request

CL_ACCESS_URL_ENV = "CL_ACCESS_MODAL_URL"

JsonObject = dict[str, object]
PostJson = Callable[[str, Mapping[str, str]], object]


@dataclass(frozen=True, slots=True)
class CourtListenerAccessConfig:
    """Configuration for the CourtListener access service."""

    base_url: str

    @classmethod
    def from_env(cls) -> CourtListenerAccessConfig:
        """Load service configuration from environment variables."""
        base_url = os.environ.get(CL_ACCESS_URL_ENV, "").strip()
        if not base_url:
            msg = f"{CL_ACCESS_URL_ENV} must be set to the cl-access service URL"
            raise ValueError(msg)
        return cls(base_url=base_url)


@dataclass(frozen=True, slots=True)
class CourtListenerCitationLookup:
    """Normalized citation lookup response from the cl-access service."""

    citation: str
    status: int
    clusters: tuple[JsonObject, ...]
    cache: str | None = None
    key: str | None = None
    error_message: str | None = None


class CourtListenerAccessClient:
    """Small HTTP wrapper around the deployed cl-access service."""

    def __init__(
        self,
        config: CourtListenerAccessConfig | None = None,
        *,
        post_json: PostJson | None = None,
    ) -> None:
        self.config = config or CourtListenerAccessConfig.from_env()
        self._post_json = post_json or _post_json

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Look up a reporter citation through cl-access."""
        url = f"{self.config.base_url.rstrip('/')}/citation-lookup"
        _validate_http_url(url)
        payload = self._post_json(
            url,
            {"volume": volume, "reporter": reporter, "page": page},
        )
        return _normalize_lookup_payload(payload, volume, reporter, page)


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
        return {
            "response": {
                "citation": " ".join(data[key] for key in ("volume", "reporter", "page")),
                "status": exc.code,
                "error_message": detail,
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


def _normalize_lookup_payload(
    payload: object,
    volume: str,
    reporter: str,
    page: str,
) -> CourtListenerCitationLookup:
    envelope = payload if isinstance(payload, dict) else {}
    raw_response = envelope.get("response", envelope)
    response = raw_response if isinstance(raw_response, dict) else {}
    if isinstance(payload, list) and payload:
        first = payload[0]
        response = first if isinstance(first, dict) else {}

    citation = _string_value(response.get("citation")) or f"{volume} {reporter} {page}"
    status = _int_value(response.get("status")) or 404
    clusters = tuple(
        item for item in response.get("clusters", []) if isinstance(item, dict)
    )
    error_message = _string_value(response.get("error_message"))

    return CourtListenerCitationLookup(
        citation=citation,
        status=status,
        clusters=clusters,
        cache=_string_value(envelope.get("cache")),
        key=_string_value(envelope.get("key")),
        error_message=error_message,
    )


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_value(value: object) -> int | None:
    return value if isinstance(value, int) else None
