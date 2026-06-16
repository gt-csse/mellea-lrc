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
    ) -> None:
        self.config = config or CourtListenerAccessConfig.from_env()
        self._post_json = post_json or _post_json

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
