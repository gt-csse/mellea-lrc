"""Read written courts or infer them from unambiguous reporter identity."""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from courts_db import courts

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction._support.context_windows import dated_parenthetical, require_structure
from mellea_lrc.model.citations import FullCitationVariant, FullReporterCitation
from mellea_lrc.model.citations.fields.date import FULL_DATE_RE, YEAR_RE
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "7_courts"

_DATE_EVENT = re.compile(r"\s+\b(?:filed|decided|issued)\b\s*$", re.I)
_REPORTER_SERIES = re.compile(r"\s*\d+\s*(?:st|nd|rd|d|th)\s*$", re.I)
_APPEALS_REPORTER = re.compile(r"\s*App\.?$", re.I)
_ORDINAL = re.compile(r"\b(\d+)[^\S\r\n]*(?:st|nd|rd|d|th)\b", re.I)
_NON_WORD = re.compile(r"[^\w]")


def _court_key(label: str) -> str:
    """Compare reporter and court abbreviations without ordinal spelling or punctuation."""
    return _NON_WORD.sub("", _ORDINAL.sub(r"\1", label)).casefold()


@lru_cache(maxsize=1)
def _courts_by_citation_string() -> dict[str, frozenset[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in courts:
        key = _court_key(item.get("citation_string") or "")
        if key:
            grouped[key].add(str(item["id"]))
    return {key: frozenset(ids) for key, ids in grouped.items()}


def _court_for_reporter_edition(edition: str, cite_type: str, name: str) -> str | None:
    """Map a reporter's normalized database identity to one court, if unique."""
    base = _REPORTER_SERIES.sub("", edition).strip()
    forms = [base]
    if _APPEALS_REPORTER.search(base):
        stem = _APPEALS_REPORTER.sub("", base)
        forms.extend((f"{stem} Ct. App.", f"{stem} App. Ct."))
    for form in forms:
        found = _courts_by_citation_string().get(_court_key(form))
        if found and len(found) == 1:
            return next(iter(found))
    if cite_type == "federal" and ("supreme court" in name.casefold() or "lawyer" in name.casefold()):
        return "scotus"
    return None


def _court_from_reporter(citation: FullCitationVariant) -> str | None:
    if not isinstance(citation, FullReporterCitation):
        return None
    locator = citation.locator[-1]
    if not locator.normalizable:
        return None
    value = locator.get_normalized()
    # The resolved edition distinguishes an early Supreme Court reporter from
    # identically spelled state reporters; an unresolved edition never gets here.
    if value.reporter.is_scotus:
        return "scotus"
    return _court_for_reporter_edition(value.edition, value.reporter.cite_type, value.reporter.name)


def resolve_courts(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an explicit post-site court or infer a unique reporter court."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        found = dated_parenthetical(document, citation, config.post_locator_window)
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
            document = document.replace_citation(citation.record(STAGE).with_court(document.text, span))
            continue
        court = _court_from_reporter(citation)
        if court is not None:
            document = document.replace_citation(citation.record(STAGE).with_inferred_court(court))
    return document.complete(STAGE)
