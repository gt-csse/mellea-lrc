"""Bound field-reading windows by the locator and colocation layer."""

from __future__ import annotations

import re

from mellea_lrc.extraction.full_reporter_locator import full_reporter_readings
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.model.citations import CaseName, FullCitationVariant, FullReporterCitation
from mellea_lrc.model.citations.fields.court import court_id_if_unique
from mellea_lrc.model.citations.fields.date import FULL_DATE_RE, YEAR_RE
from mellea_lrc.model.citations.fields.pin_cite import PIN_PREFIX
from mellea_lrc.model.citations.history import latest
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

CASE_NAMES_STAGE = "case_names"
COURTS_STAGE = "courts"
DATES_STAGE = "dates"
PIN_CITES_STAGE = "pin_cites"

_CASE = re.compile(r"(?:In re|Ex parte)\s+[^,;\n]{2,100}|[A-Z][^,;\n]{0,100}?\s+v\.\s+[^,;\n]{1,100}")
_SIGNAL = re.compile(r"^(?:See(?: also)?|Cf\.|But see|Accord|Compare)\s+", re.I)
_PAREN = re.compile(r"\((?P<body>[^()\r\n]{0,100})\)")
_BARE_NOTE = re.compile(r"^\s*,?\s*(?:at\s+)?(?P<pin>(?:n{1,2}\.|fn\.?)\s*\d+)", re.I)
_COURT_ORDINAL = re.compile(r"(?:st|nd|rd|th|d)\b\s+(?:Cir\.|Dept\.|Dist\.)", re.I)
# A comma-plus-number may start a parallel citation's volume. Its following
# reporter-like token and page distinguish it from another pinpoint target.
_PARALLEL_REPORTER = re.compile(
    r"\s+(?:[A-Z][A-Za-z0-9.]*\.[A-Za-z0-9.]*|[A-Z]{2,})"
    r"(?:\s+[A-Za-z0-9.]+){0,3}\s+\d+\b"
)
_PIN_CONTINUATION = re.compile(
    r"(?:[A-Za-z0-9]+"
    r"|[^\S\r\n]*[-–]\s*(?:[A-Za-z0-9*¶]+)?"
    r"|\s*,\s*(?:\d[\dA-Za-z]*|\*\d+|¶+\s*\d+|n{1,2}\.\s*\w+|fn\.?\s*\w+)"
    r"|[^\S\r\n]+(?:n{1,2}\.|fn\.?)\s*[^\s;(),]+"
    r"|[^\S\r\n]+(?:and|&)\s*(?:\d+|n{1,2}\.\s*\w+|fn\.?\s*\w+)"
    r"|[^\S\r\n]*&\s*(?:n{1,2}\.\s*\w+|fn\.?\s*\w+))",
    re.I,
)
_NAME_TOKEN = re.compile(r"[\w.'’&-]+")
_VERSUS = re.compile(r"\s+v\.\s+")
_DATE_EVENT = re.compile(r"\s+\b(?:filed|decided|issued)\b\s*$", re.I)


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
        (item for item in full_reporter_readings(excerpt) if item.span == (len(before), len(excerpt))),
        None,
    )
    if parsed is None or not parsed.citation.metadata.plaintiff or not parsed.citation.metadata.defendant:
        return None
    plaintiff_tokens = _NAME_TOKEN.findall(parsed.citation.metadata.plaintiff)
    defendant_tokens = _NAME_TOKEN.findall(parsed.citation.metadata.defendant)
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
    if CASE_NAMES_STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {CASE_NAMES_STAGE}")
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        before, start = _before(document, citation, config.case_name_window)
        span = _reporter_name_span(citation, before, start)
        if span is not None:
            document = document.replace_citation(
                citation.record(CASE_NAMES_STAGE).with_case_name(document.text, span)
            )
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
        document = document.replace_citation(
            citation.record(CASE_NAMES_STAGE).with_case_name(document.text, span)
        )
    return document.complete(CASE_NAMES_STAGE)


def _court_from_reporter(citation: FullCitationVariant) -> str | None:
    if not isinstance(citation, FullReporterCitation):
        return None
    locator = citation.locator[-1]
    # Eyecite's isolated locator may infer a unique reporter court. Running it
    # on this exact span avoids its unbounded post-citation metadata leak.
    isolated = full_reporter_readings(locator.quote)
    if len(isolated) == 1 and isolated[0].span == (0, len(locator.quote)):
        return isolated[0].citation.metadata.court
    if not locator.normalizable:
        return None
    return court_id_if_unique(locator.get_normalized().edition)


def resolve_courts(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an explicit post-site court or infer a unique reporter court."""
    if COURTS_STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {COURTS_STAGE}")
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
                court_region = body[: date.start()]
                if event := _DATE_EVENT.search(court_region):
                    court_region = court_region[: event.start()]
                written = court_region.strip(" ,;")
                if written:
                    body_start = start + parenthetical.start("body")
                    stripped = len(court_region) - len(court_region.lstrip(" ,;"))
                    span = Span(body_start + stripped, body_start + stripped + len(written))
        if span is not None:
            document = document.replace_citation(
                citation.record(COURTS_STAGE).with_court(document.text, span)
            )
            continue
        court = _court_from_reporter(citation)
        if court is not None:
            document = document.replace_citation(citation.record(COURTS_STAGE).with_inferred_court(court))
    return document.complete(COURTS_STAGE)


def resolve_dates(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an exact day or year from the bounded post-site parenthetical."""
    if DATES_STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {DATES_STAGE}")
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
        document = document.replace_citation(citation.record(DATES_STAGE).with_date(document.text, span))
    return document.complete(DATES_STAGE)


def resolve_pin_cites(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an adjacent pin, retaining malformed continuations for later review."""
    if PIN_CITES_STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {PIN_CITES_STAGE}")
    _require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        site = _site(citation)
        next_start = min(
            (_site(item).start for item in document.full_locators if _site(item).start >= site.end),
            default=len(document.text),
        )
        region = document.text[site.end : min(next_start, site.end + config.post_locator_window)]
        match = PIN_PREFIX.match(region) or _BARE_NOTE.match(region)
        if match is None:
            continue
        end = match.end("pin")
        if _COURT_ORDINAL.match(region, end):
            continue
        if region[end : end + 1] == ":":
            # A colon immediately after digits is ordinarily prose numbering,
            # not a pin cite after the locator.
            continue
        pin_text = match.group("pin")
        parallel_reporter = "," in pin_text and _PARALLEL_REPORTER.match(region, end)
        if parallel_reporter:
            end = match.start("pin") + pin_text.rfind(",")
        elif continuation := _PIN_CONTINUATION.match(region, end):
            # Keep the unread token, without swallowing the following prose.
            end = len(region[: continuation.end()].rstrip())
        span = Span(site.end + match.start("pin"), site.end + end)
        document = document.replace_citation(
            citation.record(PIN_CITES_STAGE).with_pin_cite(document.text, span)
        )
    return document.complete(PIN_CITES_STAGE)
