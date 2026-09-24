"""Shared locator and colocation boundaries for contextual field stages."""

from __future__ import annotations

import re

from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.fields.date import YEAR_RE
from mellea_lrc.model.citations.history import latest
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

_PAREN = re.compile(r"\((?P<body>[^()\r\n]{0,100})\)")


def require_structure(document: Document) -> None:
    if "colocations" not in document.stage_runs:
        raise ValueError("Resolve colocations before reading contextual fields")
    if "roots" in document.stage_runs:
        raise ValueError("Read contextual fields before forming roots")


def _members(document: Document, citation: FullCitationVariant) -> tuple[FullCitationVariant, ...]:
    colocation_id = latest(citation.colocation_id)
    if colocation_id is None:
        return (citation,)
    ids = next(group.citation_ids for group in document.colocations if group.id == colocation_id)
    by_id = {item.id: item for item in document.citations}
    return tuple(by_id[identifier] for identifier in ids)


def site(citation: FullCitationVariant) -> Span:
    return citation.locator_span


def before(document: Document, citation: FullCitationVariant, limit: int) -> tuple[str, int]:
    members = _members(document, citation)
    first = min(site(item).start for item in members)
    previous = max(
        (
            site(item).end
            for item in document.full_locators
            if item.id not in {member.id for member in members} and site(item).end <= first
        ),
        default=0,
    )
    start = max(previous, first - limit)
    return document.text[start:first], start


def after(document: Document, citation: FullCitationVariant, limit: int) -> tuple[str, int]:
    members = _members(document, citation)
    last = max(site(item).end for item in members)
    following = min(
        (
            site(item).start
            for item in document.full_locators
            if item.id not in {member.id for member in members} and site(item).start >= last
        ),
        default=len(document.text),
    )
    stop = min(following, last + limit)
    return document.text[last:stop], last


def dated_parenthetical(
    document: Document, citation: FullCitationVariant, limit: int
) -> tuple[re.Match[str], int] | None:
    after_text, start = after(document, citation, limit)
    for match in _PAREN.finditer(after_text):
        if not YEAR_RE.search(match.group("body")):
            continue
        # A period starting a new sentence before the parenthetical ends the
        # current citation even if no later locator was discovered.
        if re.search(r"\.\s+[A-Z]", after_text[: match.start()]):
            return None
        return match, start
    return None
