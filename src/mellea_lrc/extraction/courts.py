"""Read written courts or infer them from unambiguous reporter identity."""

from __future__ import annotations

import re

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction._support.context_windows import dated_parenthetical, require_structure
from mellea_lrc.extraction.full_reporter_locator import full_reporter_readings
from mellea_lrc.model.citations import FullCitationVariant, FullReporterCitation
from mellea_lrc.model.citations.fields.court import court_id_if_unique
from mellea_lrc.model.citations.fields.date import FULL_DATE_RE, YEAR_RE
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "courts"

_DATE_EVENT = re.compile(r"\s+\b(?:filed|decided|issued)\b\s*$", re.I)


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
