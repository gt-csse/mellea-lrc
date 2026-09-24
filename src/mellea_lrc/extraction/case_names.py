"""Read case names before each full locator site."""

from __future__ import annotations

import re

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction._support.context_windows import before as bounded_before
from mellea_lrc.extraction._support.context_windows import require_structure
from mellea_lrc.extraction.full_reporter_locator import full_reporter_readings
from mellea_lrc.model.citations import CaseName, FullCitationVariant, FullReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "case_names"

_CASE = re.compile(r"(?:In re|Ex parte)\s+[^,;\n]{2,100}|[A-Z][^,;\n]{0,100}?\s+v\.\s+[^,;\n]{1,100}")
_SIGNAL = re.compile(r"^(?:See(?: also)?|Cf\.|But see|Accord|Compare)\s+", re.I)
_NAME_TOKEN = re.compile(r"[\w.'’&-]+")
_VERSUS = re.compile(r"\s+v\.\s+")


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
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        before, start = bounded_before(document, citation, config.case_name_window)
        span = _reporter_name_span(citation, before, start)
        if span is not None:
            document = document.replace_citation(citation.record(STAGE).with_case_name(document.text, span))
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
        document = document.replace_citation(citation.record(STAGE).with_case_name(document.text, span))
    return document.complete(STAGE)
