"""Propose labelled, non-CM/ECF docket locator sites for independent review.

The stable reader recognises only the federal CM/ECF family. This generator
does not add another docket grammar: it finds an explicit docket label followed
by a bounded opaque identifier and leaves the question of whether that text
names a court case entirely to the reviewer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.adjudication.masking import mask_locator_spans
from mellea_lrc.extraction.reading.dockets import DOCKET_PREFIX

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import Document


@dataclass(frozen=True, slots=True)
class SuspectedDocket:
    """A labelled opaque locator that may identify a court case.

    ``locator_text`` includes the label, so its span is exactly the full docket
    locator that an accepted root records. ``docket_number`` is its opaque
    identifier portion. Site hunting decides only whether this locator is a
    cited case docket; court, date, and case-name readers run later over the
    admitted citation record.
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
_OPAQUE_TOKEN = r"[A-Za-z0-9][A-Za-z0-9:/\\-]*(?:\.(?=[^\S\r\n]+[A-Za-z0-9]))?"
_OPAQUE_IDENTIFIER = rf"{_OPAQUE_TOKEN}(?:[^\S\r\n]+{_OPAQUE_TOKEN}){{0,4}}"
_DOCKET_SITE = re.compile(rf"{DOCKET_PREFIX}(?P<docket>{_OPAQUE_IDENTIFIER})", re.IGNORECASE)


def suspected_dockets(document: Document) -> tuple[SuspectedDocket, ...]:
    """Return labelled docket-shaped text the narrow reader did not already read.

    Existing docket locators are blanked before scanning, even when an earlier
    audit withdrew one. Hunting therefore expands the reader's coverage; it
    never retries or contradicts a first-pass result.
    """
    masked = mask_locator_spans(document)
    sites: list[SuspectedDocket] = []
    for match in _DOCKET_SITE.finditer(masked):
        start, end = match.span()
        window_start = max(0, start - _CONTEXT)
        window_end = min(len(document.text), end + _CONTEXT)
        sites.append(
            SuspectedDocket(
                locator_span=Span(start=start, end=end),
                locator_text=document.text[start:end],
                docket_number=document.text[match.start("docket") : match.end("docket")],
                context_span=Span(start=window_start, end=window_end),
                context=masked[window_start:window_end],
            )
        )
    return tuple(sites)
