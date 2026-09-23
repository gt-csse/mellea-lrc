"""Written or reporter-inferred court IDs."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Self

from eyecite.helpers import courts
from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, source_quote
from mellea_lrc.model.span import Span


def _court_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


@lru_cache(maxsize=1)
def _court_index() -> dict[str, frozenset[str]]:
    grouped: dict[str, set[str]] = {}
    for item in courts:
        key = _court_key(item.get("citation_string") or "")
        if key:
            grouped.setdefault(key, set()).add(str(item["id"]))
    return {key: frozenset(ids) for key, ids in grouped.items()}


@lru_cache(maxsize=1)
def _courts_by_id() -> dict[str, Court]:
    return {str(item["id"]): Court(id=str(item["id"]), name=str(item["name"])) for item in courts}


class Court(BaseModel):
    """The normalized court identity used by a citation field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("Court identity is incomplete")
        return self

    @classmethod
    def from_id(cls, court_id: str) -> Self:
        try:
            return _courts_by_id()[court_id]
        except KeyError as exc:
            raise ValueError(f"Unknown normalized court ID: {court_id!r}") from exc


def court_id_if_unique(text: str) -> str | None:
    """Return one mapped court ID; absence is not a normalized reading."""
    found = _court_index().get(_court_key(text))
    return next(iter(found)) if found and len(found) == 1 else None


def normalize_court(quote: str) -> Court:
    """Resolve a written court label or raise rather than store null."""
    court_id = court_id_if_unique(quote)
    if court_id is None:
        raise ValueError(f"Cannot normalize written court: {quote!r}")
    return Court.from_id(court_id)


class CourtField(CitationField[Court]):
    """A normalized Court object, grounded in text or inferred from a reporter."""

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(node_id=node_id, quote=quote, span=span, normalized=normalize_court(quote))

    @classmethod
    def inferred(cls, court_id: str, *, node_id: str) -> Self:
        return cls(node_id=node_id, normalized=Court.from_id(court_id))

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        if self.normalized != Court.from_id(self.normalized.id):
            raise ValueError("Court normalization does not match its court ID")
        if self.quote is not None and self.normalized != normalize_court(self.quote):
            raise ValueError("Court normalization does not match its quote")
        return self
