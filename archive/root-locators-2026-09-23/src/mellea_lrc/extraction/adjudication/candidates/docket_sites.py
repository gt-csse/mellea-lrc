"""Propose unread docket locator sites for independent review.

The stable reader recognises only the federal CM/ECF family. This generator
does not add another docket grammar: it finds an explicit docket label followed
by a bounded opaque identifier, an identifier immediately beside an already
read reporter locator, or a compound identifier in citation-shaped context.
The reviewer decides whether any candidate names a court case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.extraction.adjudication.masking import mask_locator_spans
from mellea_lrc.extraction.reading.dockets import DOCKET_PREFIX
from mellea_lrc.model.citations import FullCaseCitation
from mellea_lrc.model.spans import Span

if TYPE_CHECKING:
    from mellea_lrc.model.document import Document


@dataclass(frozen=True, slots=True)
class SuspectedDocket:
    """An opaque locator that may identify a court case.

    ``locator_text`` includes a label when one is written. Its span is exactly
    the full locator an accepted root records. ``docket_number`` excludes an
    optional label. Court, date, and case-name readers run after admission.
    """

    locator_span: Span
    locator_text: str
    docket_number: str
    context_span: Span
    context: str


_CONTEXT = 170
# A docket's internal form is court-specific. The candidate grammar therefore
# says only that, after an explicit label, it is a short run of identifier-like
# tokens. Commas, brackets, and sentence punctuation stop the run; the review
# decides what the run means.
# A period is retained only when another token follows it. This keeps local
# forms such as ``19 Civ. 8034`` whole without absorbing the sentence-ending
# period after ``No. 57``.
_OPAQUE_TOKEN = r"[A-Za-z0-9](?:[A-Za-z0-9:/\\-]|\.(?=[A-Za-z0-9]|[^\S\r\n]+[A-Za-z0-9]))*"
_OPAQUE_IDENTIFIER = rf"{_OPAQUE_TOKEN}(?:[^\S\r\n]+{_OPAQUE_TOKEN}){{0,4}}"
_DOCKET_SITE = re.compile(rf"{DOCKET_PREFIX}(?P<docket>{_OPAQUE_IDENTIFIER})", re.IGNORECASE)

# A bare opaque identifier can be a docket citation when it is written as a
# parallel identifier immediately before a reporter locator. This syntax is
# deliberately broad: the reviewer, not a court-specific number grammar,
# decides whether it names a case. Requiring a digit and a joining character
# excludes ordinary words and years while still allowing local docket forms.
_BARE_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9:/\\-])[A-Za-z0-9][A-Za-z0-9:/\\-]{1,45}(?![A-Za-z0-9:/\\-])")
_BARE_JOIN = re.compile(r"[^A-Za-z0-9]{0,20}\Z")
_BARE_LOOKBACK = 80

# Without a parallel reporter, a compound opaque identifier is a useful site
# only in a citation-shaped setting: a nearby case-name cue before it and a
# parenthetical after it. This is still a proposal, not a docket grammar or a
# court/date reading. Multiple joins distinguish it from ordinary page ranges.
_COMPOUND_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9:/\\-])[A-Za-z0-9]{1,16}(?:[:/\\-][^\S\r\n]*[A-Za-z0-9]{1,16}){2,5}(?![A-Za-z0-9:/\\-])"
)
_CASE_NAME_CUE = re.compile(r"\bv\.\s|\bin\s+re\b", re.IGNORECASE)
_CASE_CUE_LOOKBACK = 130
_PAREN_LOOKAHEAD = 70


def _site(document: Document, masked: str, start: int, end: int, docket_start: int) -> SuspectedDocket:
    window_start = max(0, start - _CONTEXT)
    window_end = min(len(document.text), end + _CONTEXT)
    return SuspectedDocket(
        locator_span=Span(start=start, end=end),
        locator_text=document.text[start:end],
        docket_number=document.text[docket_start:end],
        context_span=Span(start=window_start, end=window_end),
        context=masked[window_start:window_end],
    )


def suspected_dockets(document: Document) -> tuple[SuspectedDocket, ...]:
    """Return labelled docket-shaped text the narrow reader did not already read.

    Existing docket locators are blanked before scanning, even when an earlier
    audit withdrew one. Hunting therefore expands the reader's coverage; it
    never retries or contradicts a first-pass result. Bare proposals need
    either a known reporter locator or both case-name and parenthetical cues,
    so they do not scan every numbered range in a filing.
    """
    masked = mask_locator_spans(document)
    reporter_spans = tuple(
        citation.locator_span
        for citation in document.citations
        if isinstance(citation.fields, FullCaseCitation)
    )
    sites: list[SuspectedDocket] = []
    for match in _DOCKET_SITE.finditer(masked):
        start, end = match.span()
        sites.append(_site(document, masked, start, end, match.start("docket")))

    labelled = tuple(site.locator_span for site in sites)
    for reporter in reporter_spans:
        reporter_start = reporter.start
        lookback = max(0, reporter_start - _BARE_LOOKBACK)
        for match in _BARE_IDENTIFIER.finditer(masked, lookback, reporter_start):
            start, end = match.span()
            token = masked[start:end]
            if (
                end > reporter_start
                or not _BARE_JOIN.fullmatch(masked[end:reporter_start])
                or not any(character.isdigit() for character in token)
                or not any(character in ":/\\-" for character in token)
                or any(start < span.end and end > span.start for span in labelled)
                or not _bare_left_boundary(masked, start)
                or any(
                    prior.end <= start and _BARE_JOIN.fullmatch(masked[prior.end : start])
                    for prior in reporter_spans
                )
            ):
                continue
            sites.append(_site(document, masked, start, end, start))

    held = tuple(site.locator_span for site in sites)
    for match in _COMPOUND_IDENTIFIER.finditer(masked):
        start, end = match.span()
        token = masked[start:end]
        if (
            not any(character.isdigit() for character in token)
            or any(start < span.end and end > span.start for span in held)
            or not _CASE_NAME_CUE.search(masked[max(0, start - _CASE_CUE_LOOKBACK) : start])
        ):
            continue
        following = masked[end : end + _PAREN_LOOKAHEAD]
        open_paren = following.find("(")
        if open_paren < 0 or open_paren > 35 or ")" not in following[open_paren:]:
            continue
        sites.append(_site(document, masked, start, end, start))
    return tuple(sorted(sites, key=lambda site: (site.locator_span.start, site.locator_span.end)))


def _bare_left_boundary(text: str, start: int) -> bool:
    """Avoid taking a number from an unpunctuated case-name word as a locator."""
    before = text[:start].rstrip()
    return not before or not before[-1].isalnum()
