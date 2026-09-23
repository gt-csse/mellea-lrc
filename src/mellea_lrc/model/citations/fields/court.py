"""Written or reporter-inferred court IDs."""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Self

from courts_db import courts
from pydantic import BaseModel, ConfigDict, model_validator
from reporters_db import STATE_ABBREVIATIONS

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span

_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th|d)$")
_SPLIT_ORDINAL = re.compile(r"\b(\d+)\s+(st|nd|rd|th|d)\b")


def _court_tokens(text: str) -> tuple[str, ...]:
    """Compare citation labels by tokens, folding ordinal spelling and spacing."""
    joined = _SPLIT_ORDINAL.sub(r"\1\2", text.casefold())
    tokens = re.findall(r"[a-z0-9]+", joined)
    return tuple(f"#{match.group(1)}" if (match := _ORDINAL.fullmatch(token)) else token for token in tokens)


@lru_cache(maxsize=1)
def _court_index() -> dict[tuple[str, ...], frozenset[str]]:
    grouped: dict[tuple[str, ...], set[str]] = defaultdict(set)
    location_abbreviations: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for item in courts:
        label = _court_tokens(item.get("citation_string") or "")
        if item.get("system") == "state" and len(label) == 1 and item.get("location"):
            location_abbreviations[str(item["location"])].add(label)

    for item in courts:
        label = _court_tokens(item.get("citation_string") or "")
        if not label:
            continue
        keys = {label}
        location = str(item.get("location") or "")
        location_tokens = _court_tokens(location)
        is_district = item.get("system") == "federal" and str(item.get("name") or "").startswith(
            "District Court"
        )
        if is_district:
            # Derive D. Md. from D. Maryland and the database's Md. label,
            # rather than maintaining a separate state-abbreviation map.
            if label[:1] == ("d",) and label[1:] == location_tokens:
                keys.update(("d", *alias) for alias in location_abbreviations[location])
            elif label[:1] != ("d",):
                # The database sometimes omits the written District prefix.
                keys.add(("d", *label))
        for key in keys:
            grouped[key].add(str(item["id"]))
    return {key: frozenset(ids) for key, ids in grouped.items()}


def _label_ends_in_location(suffix: tuple[str, ...], location: str, bluebook: tuple[str, ...]) -> bool:
    """Verify that a database label's tail denotes its recorded state."""
    full = _court_tokens(location)
    if suffix == full or suffix == bluebook:
        return True
    if len(suffix) != 1 or len(full) != 1 or not full[0].startswith(suffix[0]):
        return False
    # Courts-db sometimes shortens a state differently (Penn. versus Pa.).
    # Accept that spelling only when it identifies exactly one state.
    return sum(_court_tokens(state)[0].startswith(suffix[0]) for state in STATE_ABBREVIATIONS.values()) == 1


@lru_cache(maxsize=1)
def _bluebook_court_index() -> dict[tuple[str, ...], frozenset[str]]:
    """Combine reporters-db state labels with courts-db district court IDs."""
    by_location = {
        location: _court_tokens(abbreviation) for abbreviation, location in STATE_ABBREVIATIONS.items()
    }
    grouped: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for item in courts:
        name = str(item.get("name") or "")
        location = str(item.get("location") or "")
        label = _court_tokens(item.get("citation_string") or "")
        if (
            item.get("system") != "federal"
            or not ("District Court" in name or "Bankruptcy Court" in name)
            or location not in by_location
        ):
            continue
        district_positions = [index for index, token in enumerate(label[:4]) if token == "d"]
        if len(district_positions) != 1 or district_positions[0] == len(label) - 1:
            continue
        prefix_end = district_positions[0] + 1
        if not _label_ends_in_location(label[prefix_end:], location, by_location[location]):
            continue
        grouped[(*label[:prefix_end], *by_location[location])].add(str(item["id"]))
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
    key = _court_tokens(text)
    found = _court_index().get(key)
    if found is None:
        # A generated state abbreviation must not override a direct court label.
        found = _bluebook_court_index().get(key)
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
        return cls(
            node_id=node_id,
            quote=quote,
            span=span,
            **normalization_record(lambda: normalize_court(quote)),
        )

    @classmethod
    def inferred(cls, court_id: str, *, node_id: str) -> Self:
        return cls(
            node_id=node_id,
            normalizable=True,
            normalized=Court.from_id(court_id),
            normalization_error=None,
        )

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        if self.quote is not None:
            self.validate_normalization(lambda: normalize_court(self.quote))
        elif not self.normalizable or self.unchecked_normalized is None:
            raise ValueError("An inferred court needs a normalized court ID")
        elif self.unchecked_normalized != Court.from_id(self.unchecked_normalized.id):
            raise ValueError("Court normalization does not match its court ID")
        return self
