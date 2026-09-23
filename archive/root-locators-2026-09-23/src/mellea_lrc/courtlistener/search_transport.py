"""Inbound boundary layer for untrusted CourtListener search JSON."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field

from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult


class CourtListenerSearchResponsePayload(BaseModel):
    """External response from the CourtListener v4 search endpoint."""

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    count: int
    results: list[dict[str, Any]] = Field(default_factory=list)
    next: str | None = None
    previous: str | None = None

    def to_domain(
        self,
        *,
        query: str,
        search_type: str,
        semantic: bool,
    ) -> CourtListenerSearchResult:
        """Convert validated transport data to the public domain model."""
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=semantic,
            count=self.count,
            results=self.results,
            next_cursor=_cursor_from_url(self.next),
            previous_cursor=_cursor_from_url(self.previous),
        )


def _cursor_from_url(value: str | None) -> str | None:
    if not value:
        return None
    cursor = parse_qs(urlparse(value).query).get("cursor")
    return cursor[0] if cursor else None
