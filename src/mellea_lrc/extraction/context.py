"""Bound field-reading windows by the locator and colocation layer."""

from __future__ import annotations

import re
from functools import lru_cache

from eyecite import get_citations
from eyecite.helpers import courts
from eyecite.models import FullCaseCitation

from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.model.citations import CitationDate, FullCitationVariant, FullReporterCitation
from mellea_lrc.model.citations.history import CitationField, latest
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

_CASE = re.compile(r"(?:In re|Ex parte)\s+[^,;\n]{2,100}|[A-Z][^,;\n]{0,100}?\s+v\.\s+[^,;\n]{1,100}")
_SIGNAL = re.compile(r"^(?:See(?: also)?|Cf\.|But see|Accord|Compare)\s+", re.I)
_PAREN = re.compile(r"\((?P<body>[^()\r\n]{0,100})\)")
_YEAR = re.compile(r"(?<!\d)(?:1[6789]\d{2}|20\d{2}|21\d{2})(?!\d)")
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_FULL_DATE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>(?:1[6789]|20|21)\d{2})\b",
    re.I,
)
_PIN = re.compile(r"^\s*,?\s*(?:at\s+)?(?P<pin>\*?\d+(?:[-–]\d+)?)(?![\d:])")


def _require_structure(document: Document) -> None:
    if "colocations" not in document.completed_stages:
        raise ValueError("Resolve colocations before reading contextual fields")
    if "roots" in document.completed_stages:
        raise ValueError("Read contextual fields before forming roots")


def _members(document: Document, citation: FullCitationVariant) -> tuple[FullCitationVariant, ...]:
    colocation_id = latest(citation.colocation_id)
    if colocation_id is None:
        return (citation,)
    ids = next(group.citation_ids for group in document.colocations if group.id == colocation_id)
    by_id = {item.id: item for item in document.citations}
    return tuple(by_id[identifier] for identifier in ids)


def _site(citation: FullCitationVariant) -> Span:
    span = latest(citation.locator_span)
    assert span is not None
    return span


def _before(document: Document, citation: FullCitationVariant, limit: int) -> tuple[str, int]:
    members = _members(document, citation)
    first = min(_site(item).start for item in members)
    previous = max(
        (
            _site(item).end
            for item in document.full_locators
            if item.id not in {member.id for member in members} and _site(item).end <= first
        ),
        default=0,
    )
    start = max(previous, first - limit)
    return document.text[start:first], start


def _after(document: Document, citation: FullCitationVariant, limit: int) -> tuple[str, int]:
    members = _members(document, citation)
    last = max(_site(item).end for item in members)
    following = min(
        (
            _site(item).start
            for item in document.full_locators
            if item.id not in {member.id for member in members} and _site(item).start >= last
        ),
        default=len(document.text),
    )
    stop = min(following, last + limit)
    return document.text[last:stop], last


def _dated_parenthetical(
    document: Document, citation: FullCitationVariant, limit: int
) -> tuple[re.Match[str], int] | None:
    after, start = _after(document, citation, limit)
    for match in _PAREN.finditer(after):
        if not _YEAR.search(match.group("body")):
            continue
        # A period starting a new sentence before the parenthetical ends the
        # current citation even if no later locator was discovered.
        if re.search(r"\.\s+[A-Z]", after[: match.start()]):
            return None
        return match, start
    return None


def resolve_case_names(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read a name before each citation site, never through another locator."""
    stage = "case_names"
    if stage in document.completed_stages:
        return document
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        before, start = _before(document, citation, config.case_name_window)
        matches = tuple(_CASE.finditer(before))
        if not matches:
            continue
        match = matches[-1]
        name = match.group().rstrip(" ,")
        signal = _SIGNAL.match(name)
        offset = signal.end() if signal else 0
        name = name[offset:]
        if not name:
            continue
        span = Span(start + match.start() + offset, start + match.start() + offset + len(name))
        document = document.update_fields(
            stage,
            citation.id,
            {CitationField.CASE_NAME: name, CitationField.CASE_NAME_SPAN: span},
        )
    return document.complete(stage)


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


def _court_from_parenthetical(body: str, date_start: int) -> tuple[str, int] | None:
    written = body[:date_start].strip(" ,;")
    if not written:
        return None
    found = _court_index().get(_court_key(written))
    return (next(iter(found)), len(written)) if found and len(found) == 1 else None


def _court_from_reporter(citation: FullCitationVariant) -> str | None:
    if not isinstance(citation, FullReporterCitation):
        return None
    locator_text = latest(citation.locator_text)
    if not locator_text:
        return None
    # Eyecite's isolated locator may infer a unique reporter court. Running it
    # on this exact span avoids its unbounded post-citation metadata leak.
    isolated = get_citations(locator_text)
    if len(isolated) == 1 and isinstance(isolated[0], FullCaseCitation):
        return isolated[0].metadata.court
    found = _court_index().get(_court_key(latest(citation.reporter) or ""))
    return next(iter(found)) if found and len(found) == 1 else None


def resolve_courts(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an explicit post-site court or infer a unique reporter court."""
    stage = "courts"
    if stage in document.completed_stages:
        return document
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        found = _dated_parenthetical(document, citation, config.post_locator_window)
        court: str | None = None
        span: Span | None = None
        if found:
            parenthetical, start = found
            body = parenthetical.group("body")
            date = _FULL_DATE.search(body) or _YEAR.search(body)
            if date:
                resolved = _court_from_parenthetical(body, date.start())
                if resolved:
                    court, length = resolved
                    body_start = start + parenthetical.start("body")
                    stripped = len(body[: date.start()]) - len(body[: date.start()].lstrip(" ,;"))
                    span = Span(body_start + stripped, body_start + stripped + length)
        if court is None:
            court = _court_from_reporter(citation)
        if court is not None:
            changes = {CitationField.COURT: court}
            if span is not None:
                changes[CitationField.COURT_SPAN] = span
            document = document.update_fields(stage, citation.id, changes)
    return document.complete(stage)


def resolve_dates(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an exact day or year from the bounded post-site parenthetical."""
    stage = "dates"
    if stage in document.completed_stages:
        return document
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        found = _dated_parenthetical(document, citation, config.post_locator_window)
        if not found:
            continue
        parenthetical, start = found
        body = parenthetical.group("body")
        match = _FULL_DATE.search(body) or _YEAR.search(body)
        if match is None:
            continue
        date = CitationDate(
            year=int(match.group("year") if "year" in match.groupdict() else match.group()),
            month=_MONTHS[match.group("month")[:3].lower()] if "month" in match.groupdict() else None,
            day=int(match.group("day")) if "day" in match.groupdict() else None,
        )
        span = Span(
            start + parenthetical.start("body") + match.start(),
            start + parenthetical.start("body") + match.end(),
        )
        document = document.update_fields(
            stage,
            citation.id,
            {CitationField.DATE: date, CitationField.DATE_SPAN: span},
        )
    return document.complete(stage)


def resolve_pin_cites(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read each locator's own immediately adjacent page or star-page pin."""
    stage = "pin_cites"
    if stage in document.completed_stages:
        return document
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        site = _site(citation)
        next_start = min(
            (_site(item).start for item in document.full_locators if _site(item).start >= site.end),
            default=len(document.text),
        )
        region = document.text[site.end : min(next_start, site.end + config.post_locator_window)]
        match = _PIN.match(region)
        if match is None:
            continue
        span = Span(site.end + match.start("pin"), site.end + match.end("pin"))
        document = document.update_fields(
            stage,
            citation.id,
            {CitationField.PIN_CITE: match.group("pin"), CitationField.PIN_CITE_SPAN: span},
        )
    return document.complete(stage)
