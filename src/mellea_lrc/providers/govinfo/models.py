"""Response DTOs returned by GovInfo search and granule-list APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GovInfoSearchPage:
    """One complete raw search page and its package or granule results."""

    raw_json: dict[str, Any]
    results: tuple[dict[str, Any], ...]
    count: int
    next_offset_mark: str | None


@dataclass(frozen=True, slots=True)
class GovInfoGranulesPage:
    """One complete raw granule-list page and its entries."""

    raw_json: dict[str, Any]
    granules: tuple[dict[str, Any], ...]
    count: int
    next_offset_mark: str | None
