"""Propose unread docket-number sites for optional model review."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mellea_lrc.extraction.docket_locator import DOCKET_PREFIX_PATTERN
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.fields.docket import DOCKET_ENTRY_PATTERN
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

_CONTEXT = 170

# These patterns propose text for review; they do not decide docket validity.
# A labelled identifier is an opaque run rather than a jurisdiction-specific grammar.
_OPAQUE_TOKEN = r"[A-Za-z0-9](?:[A-Za-z0-9:/\\-]|\.(?=[A-Za-z0-9]|[^\S\r\n]+[A-Za-z0-9]))*"
_OPAQUE_IDENTIFIER = rf"{_OPAQUE_TOKEN}(?:[^\S\r\n]+{_OPAQUE_TOKEN}){{0,4}}"
_LABELLED = re.compile(rf"{DOCKET_PREFIX_PATTERN}(?P<number>{_OPAQUE_IDENTIFIER})", re.I)

# Unlabelled identifiers need more citation context before incurring a model call.
_BARE = re.compile(r"(?<![A-Za-z0-9:/\\-])[A-Za-z0-9][A-Za-z0-9:/\\-]{1,45}(?![A-Za-z0-9:/\\-])")
_COMPOUND = re.compile(
    r"(?<![A-Za-z0-9:/\\-])[A-Za-z0-9]{1,16}(?:[:/\\-][^\S\r\n]*[A-Za-z0-9]{1,16}){2,5}(?![A-Za-z0-9:/\\-])"
)
_CASE_CUE = re.compile(r"\bv\.\s|\bin\s+re\b", re.I)
_BARE_JOIN = re.compile(r"[^A-Za-z0-9]{0,20}\Z")


@dataclass(frozen=True, slots=True)
class DocketSiteCandidate:
    """A source span worth reviewing, not a claimed docket citation."""

    locator_span: Span
    number_span: Span
    locator_text: str
    docket_number: str
    context: str


def _masked_text(document: Document) -> str:
    masked = list(document.text)
    for citation in document.full_locators:
        span = citation.locator_span
        masked[span.start : span.end] = " " * (span.end - span.start)
    # An ECF/Doc./Dkt./D.I. entry number identifies a filing within a docket,
    # not the case docket itself. Reuse the entry reader's syntax so an inner
    # "No." cannot be proposed as a full docket locator.
    for entry in DOCKET_ENTRY_PATTERN.finditer(document.text):
        masked[entry.start() : entry.end()] = " " * (entry.end() - entry.start())
    return "".join(masked)


def _candidate(document: Document, start: int, end: int, number_start: int) -> DocketSiteCandidate:
    left = max(0, start - _CONTEXT)
    right = min(len(document.text), end + _CONTEXT)
    return DocketSiteCandidate(
        locator_span=Span(start=start, end=end),
        number_span=Span(start=number_start, end=end),
        locator_text=document.text[start:end],
        docket_number=document.text[number_start:end],
        # The scan is masked, but the review sees the original citation context.
        context=document.text[left:right],
    )


def suspected_dockets(document: Document) -> tuple[DocketSiteCandidate, ...]:
    """Propose unread labelled, parallel, and citation-shaped docket sites."""
    masked = _masked_text(document)
    sites = [
        _candidate(document, *match.span(), match.start("number")) for match in _LABELLED.finditer(masked)
    ]
    reporter_spans = tuple(
        citation.locator_span
        for citation in document.full_locators
        if isinstance(citation, FullReporterCitation)
    )
    held = [site.locator_span for site in sites]
    for reporter in reporter_spans:
        for match in _BARE.finditer(masked, max(0, reporter.start - 80), reporter.start):
            start, end = match.span()
            token = masked[start:end]
            before = masked[:start].rstrip()
            if (
                end > reporter.start
                or not _BARE_JOIN.fullmatch(masked[end : reporter.start])
                or not any(char.isdigit() for char in token)
                or not any(char in ":/\\-" for char in token)
                or (before and before[-1].isalnum())
                or any(start < span.end and end > span.start for span in held)
                or any(
                    prior.end <= start and _BARE_JOIN.fullmatch(masked[prior.end : start])
                    for prior in reporter_spans
                )
            ):
                continue
            sites.append(_candidate(document, start, end, start))
            held.append(Span(start=start, end=end))

    for match in _COMPOUND.finditer(masked):
        start, end = match.span()
        if (
            not any(char.isdigit() for char in masked[start:end])
            or any(start < span.end and end > span.start for span in held)
            or not _CASE_CUE.search(masked[max(0, start - 130) : start])
        ):
            continue
        following = masked[end : end + 70]
        opening = following.find("(")
        if opening < 0 or opening > 35 or ")" not in following[opening:]:
            continue
        sites.append(_candidate(document, start, end, start))
        held.append(Span(start=start, end=end))
    return tuple(sorted(sites, key=lambda site: (site.locator_span.start, site.locator_span.end)))
