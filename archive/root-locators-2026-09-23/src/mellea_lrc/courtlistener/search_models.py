"""Domain models for CourtListener search."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class CourtListenerSearchResult:
    """A normalized, immutable CourtListener search response."""

    query: str
    search_type: str
    semantic: bool
    count: int
    results: tuple[Mapping[str, object], ...]
    next_cursor: str | None = None
    previous_cursor: str | None = None

    @classmethod
    def from_payload(
        cls,
        *,
        query: str,
        search_type: str,
        semantic: bool,
        count: int,
        results: list[dict[str, Any]],
        next_cursor: str | None,
        previous_cursor: str | None,
    ) -> CourtListenerSearchResult:
        """Create a result with immutable copies of every upstream record."""
        return cls(
            query=query,
            search_type=search_type,
            semantic=semantic,
            count=count,
            results=tuple(_freeze_record(result) for result in results),
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
        )


def _freeze_record(record: dict[str, Any]) -> Mapping[str, object]:
    """Recursively make an upstream JSON record read-only."""
    return MappingProxyType({key: _freeze_json(value) for key, value in record.items()})


def _freeze_json(value: object) -> object:
    """Convert JSON containers to immutable equivalents."""
    if isinstance(value, dict):
        return _freeze_record(value)
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value
