"""Bounded text windows around citation locators."""

from __future__ import annotations

from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.history import latest
from mellea_lrc.model.document import Document


def _members(document: Document, citation: FullCitationVariant) -> tuple[FullCitationVariant, ...]:
    """Return citations in the same colocation as this citation."""
    colocation_id = latest(citation.colocation_id)
    if colocation_id is None:
        return (citation,)
    ids = next(group.citation_ids for group in document.colocations if group.id == colocation_id)
    by_id = {item.id: item for item in document.citations}
    return tuple(by_id[identifier] for identifier in ids)


def before(document: Document, citation: FullCitationVariant, limit: int) -> tuple[str, int]:
    """Return text before the citation's colocated locator group, bounded by adjacent locators."""
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
    """Return text after the citation's colocated locator group, bounded by adjacent locators."""
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
