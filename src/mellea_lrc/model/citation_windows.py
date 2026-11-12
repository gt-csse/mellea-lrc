"""Bounded text windows around full citation locators."""

from __future__ import annotations

from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.history import latest
from mellea_lrc.model.document import Document


def _members(document: Document, citation: FullCitationVariant) -> tuple[FullCitationVariant, ...]:
    """Return the current document's citations sharing this parsing group."""
    current = next((item for item in document.full_locators if item.id == citation.id), None)
    if current != citation:
        raise ValueError("Citation must belong to this document state")
    colocation_id = latest(citation.colocation_id)
    if colocation_id is None:
        return (citation,)
    ids = next(group.citation_ids for group in document.colocations if group.id == colocation_id)
    by_id = {item.id: item for item in document.full_locators}
    return tuple(by_id[identifier] for identifier in ids)


def before(document: Document, citation: FullCitationVariant, limit: int) -> tuple[str, int]:
    """Read before the group, bounded by the preceding full locator and limit."""
    if limit < 0:
        raise ValueError("Window limit must be nonnegative")
    members = _members(document, citation)
    first = min(item.locator_span.start for item in members)
    member_ids = {member.id for member in members}
    previous = max(
        (
            item.locator_span.end
            for item in document.full_locators
            if item.id not in member_ids and item.locator_span.end <= first
        ),
        default=0,
    )
    start = max(previous, first - limit)
    return document.text[start:first], start


def after(document: Document, citation: FullCitationVariant, limit: int) -> tuple[str, int]:
    """Read after the group, bounded by the following full locator and limit."""
    if limit < 0:
        raise ValueError("Window limit must be nonnegative")
    members = _members(document, citation)
    last = max(item.locator_span.end for item in members)
    member_ids = {member.id for member in members}
    following = min(
        (
            item.locator_span.start
            for item in document.full_locators
            if item.id not in member_ids and item.locator_span.start >= last
        ),
        default=len(document.text),
    )
    stop = min(following, last + limit)
    return document.text[last:stop], last
