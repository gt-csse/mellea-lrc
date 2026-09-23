"""Bound field-reading windows by the locator and colocation layer."""

from __future__ import annotations

import re

from eyecite import get_citations
from eyecite.models import FullCaseCitation

from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.model.citations import CaseName, FullCitationVariant, FullReporterCitation
from mellea_lrc.model.citations.fields.court import court_id_if_unique
from mellea_lrc.model.citations.fields.date import FULL_DATE_RE, YEAR_RE
from mellea_lrc.model.citations.history import latest
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

_CASE = re.compile(r"(?:In re|Ex parte)\s+[^,;\n]{2,100}|[A-Z][^,;\n]{0,100}?\s+v\.\s+[^,;\n]{1,100}")
_SIGNAL = re.compile(r"^(?:See(?: also)?|Cf\.|But see|Accord|Compare)\s+", re.I)
_PAREN = re.compile(r"\((?P<body>[^()\r\n]{0,100})\)")
_PIN = re.compile(r"^\s*,?\s*(?:at\s+)?(?P<pin>\*?\d+(?:[-–]\d+)?)(?![\d:])")
_NAME_TOKEN = re.compile(r"[\w.'’&-]+")
_VERSUS = re.compile(r"\s+v\.\s+")


def _require_structure(document: Document) -> None:
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


def _site(citation: FullCitationVariant) -> Span:
    return citation.locator_span


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
        if not YEAR_RE.search(match.group("body")):
            continue
        # A period starting a new sentence before the parenthetical ends the
        # current citation even if no later locator was discovered.
        if re.search(r"\.\s+[A-Z]", after[: match.start()]):
            return None
        return match, start
    return None


def _reporter_name_span(citation: FullCitationVariant, before: str, start: int) -> Span | None:
    """Use eyecite's parsed parties to anchor a written name, not surrounding prose."""
    if not isinstance(citation, FullReporterCitation):
        return None
    site = citation.locator[-1].quote
    excerpt = before + site
    parsed = next(
        (
            item
            for item in get_citations(excerpt)
            if isinstance(item, FullCaseCitation) and item.span() == (len(before), len(excerpt))
        ),
        None,
    )
    if parsed is None or not parsed.metadata.plaintiff or not parsed.metadata.defendant:
        return None
    plaintiff_tokens = _NAME_TOKEN.findall(parsed.metadata.plaintiff)
    defendant_tokens = _NAME_TOKEN.findall(parsed.metadata.defendant)
    if not plaintiff_tokens or not defendant_tokens:
        return None
    first, last = plaintiff_tokens[0], defendant_tokens[-1]
    for separator in reversed(tuple(_VERSUS.finditer(before))):
        left = before[: separator.start()]
        right = before[separator.end() :]
        starts = tuple(re.finditer(rf"(?<!\w){re.escape(first)}(?!\w)", left, re.I))
        end = re.search(rf"(?<!\w){re.escape(last)}(?!\w)", right, re.I)
        if not starts or end is None:
            continue
        local_start = starts[-1].start()
        local_end = separator.end() + end.end()
        if local_end - local_start > 100:
            continue
        try:
            CaseName.from_quote(before[local_start:local_end])
        except ValueError:
            continue
        return Span(start + local_start, start + local_end)
    return None


def resolve_case_names(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read a name before each citation site, never through another locator."""
    stage = "case_names"
    if stage in document.stage_runs:
        return document
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        before, start = _before(document, citation, config.case_name_window)
        span = _reporter_name_span(citation, before, start)
        if span is not None:
            document = document.replace_citation(citation.record(stage).with_case_name(document.text, span))
            continue
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
        document = document.replace_citation(citation.record(stage).with_case_name(document.text, span))
    return document.complete(stage)


def _court_from_reporter(citation: FullCitationVariant) -> str | None:
    if not isinstance(citation, FullReporterCitation):
        return None
    locator = citation.locator[-1]
    # Eyecite's isolated locator may infer a unique reporter court. Running it
    # on this exact span avoids its unbounded post-citation metadata leak.
    isolated = get_citations(locator.quote)
    if len(isolated) == 1 and isinstance(isolated[0], FullCaseCitation):
        return isolated[0].metadata.court
    return court_id_if_unique(locator.normalized.edition)


def resolve_courts(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an explicit post-site court or infer a unique reporter court."""
    stage = "courts"
    if stage in document.stage_runs:
        return document
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        found = _dated_parenthetical(document, citation, config.post_locator_window)
        span: Span | None = None
        if found:
            parenthetical, start = found
            body = parenthetical.group("body")
            date = FULL_DATE_RE.search(body) or YEAR_RE.search(body)
            if date:
                written = body[: date.start()].strip(" ,;")
                if written:
                    body_start = start + parenthetical.start("body")
                    stripped = len(body[: date.start()]) - len(body[: date.start()].lstrip(" ,;"))
                    span = Span(body_start + stripped, body_start + stripped + len(written))
        if span is not None:
            document = document.replace_citation(citation.record(stage).with_court(document.text, span))
            continue
        court = _court_from_reporter(citation)
        if court is not None:
            document = document.replace_citation(citation.record(stage).with_inferred_court(court))
    return document.complete(stage)


def resolve_dates(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an exact day or year from the bounded post-site parenthetical."""
    stage = "dates"
    if stage in document.stage_runs:
        return document
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        found = _dated_parenthetical(document, citation, config.post_locator_window)
        if not found:
            continue
        parenthetical, start = found
        body = parenthetical.group("body")
        match = FULL_DATE_RE.search(body) or YEAR_RE.search(body)
        if match is None:
            continue
        span = Span(
            start + parenthetical.start("body") + match.start(),
            start + parenthetical.start("body") + match.end(),
        )
        document = document.replace_citation(citation.record(stage).with_date(document.text, span))
    return document.complete(stage)


def resolve_pin_cites(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read each locator's own immediately adjacent page or star-page pin."""
    stage = "pin_cites"
    if stage in document.stage_runs:
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
        document = document.replace_citation(citation.record(stage).with_pin_cite(document.text, span))
    return document.complete(stage)
