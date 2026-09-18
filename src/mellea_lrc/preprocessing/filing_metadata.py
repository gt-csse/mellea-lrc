"""Mask filing metadata that cannot be a citation to another case.

The text passed to extraction is also its coordinate system.  This module
therefore never removes text: each selected character becomes one space.  The
selection is deliberately narrow.  A filing's caption supplies its own docket
number, and a complete CM/ECF stamp has a machine-generated structure prose
does not use.
Every other docket-shaped string remains available to the docket reader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

from mellea_lrc.core.spans import Span

# This reader removes document metadata.  Unlike the general docket reader it
# must never turn an ordinary caption word (for example, "Case law") into a
# removal, so the identifier must contain a digit.
_DOCKET = r"(?=[A-Za-z0-9:/\\.\-]*\d)[A-Za-z0-9][A-Za-z0-9:/\\.\-]*[A-Za-z0-9]"

# This is intentionally a complete stamp, rather than a search for a docket
# near words such as "Filed".  The latter can occur in an argument about a
# filing.  Its complete, machine-generated structure is enough to distinguish
# it from prose, including in a one-page filing where it cannot recur.
_ECF_STAMP = re.compile(
    rf"\bCase(?:[ \t]+No\.?)?[ \t]+(?P<docket>{_DOCKET})"
    rf"[ \t]+Document[ \t]+(?:No\.?[ \t]+)?\d+(?:-\d+)?"
    rf"[ \t]+Filed[ \t]+\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}"
    rf"[ \t]+Page[ \t]+\d+[ \t]+of[ \t]+\d+\b",
    re.IGNORECASE,
)
# A caption is only identified in the document's opening under a court heading.
# This avoids treating a later cited, related, or discussed case as the filing.
_COURT_HEADING = re.compile(
    r"\b(?:IN[ \t]+THE[ \t]+)?(?:UNITED[ \t]+STATES[ \t]+)?"
    r"(?:DISTRICT|BANKRUPTCY|SUPREME)[ \t]+COURT\b",
    re.IGNORECASE,
)
_CAPTION_LABEL_GAP = r"[ \t]*(?:[:#][ \t]*)?(?:\r?\n[ \t]*){0,2}"
_CAPTION_DOCKET = re.compile(
    rf"\b(?:Case[ \t]+(?:No\.?{_CAPTION_LABEL_GAP}|(?=\d))|"
    rf"Civil[ \t]+Action[ \t]+No\.?{_CAPTION_LABEL_GAP})"
    rf"(?P<docket>{_DOCKET})",
    re.IGNORECASE,
)
_CAPTION_SEARCH_LIMIT = 2_500


class FilingMetadataKind(str, Enum):
    """The two kinds of metadata extraction must ignore."""

    CAPTION_DOCKET = "filing_caption_docket"
    ECF_STAMP = "ecf_page_stamp"


@dataclass(frozen=True, slots=True)
class FilingMetadataRemoval:
    """One offset-preserving masking operation, with enough data to reverse it."""

    kind: FilingMetadataKind
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class FilingMetadataMask:
    """Masked text and its auditable, reversible removals."""

    text: str
    removals: tuple[FilingMetadataRemoval, ...]


def filing_metadata_removals(text: str) -> tuple[FilingMetadataRemoval, ...]:
    """Return only the filing caption docket and complete CM/ECF stamp records."""
    stamps = list(_ECF_STAMP.finditer(text))
    removals = [
        FilingMetadataRemoval(FilingMetadataKind.ECF_STAMP, stamp.start(), stamp.end(), stamp.group(0))
        for stamp in stamps
    ]
    stamp_spans = [Span(start=removal.start, end=removal.end) for removal in removals]
    for span in _caption_docket_spans(text, stamp_spans):
        removals.append(
            FilingMetadataRemoval(
                FilingMetadataKind.CAPTION_DOCKET, span.start, span.end, text[span.start : span.end]
            )
        )
    return tuple(sorted(removals, key=lambda removal: (removal.start, removal.end)))


def mask_filing_metadata(text: str) -> FilingMetadataMask:
    """Mask filing metadata and retain the records needed to restore it exactly."""
    characters = list(text)
    removals = filing_metadata_removals(text)
    for removal in removals:
        characters[removal.start : removal.end] = " " * (removal.end - removal.start)
    return FilingMetadataMask(text="".join(characters), removals=removals)


def restore_filing_metadata(masked_text: str, removals: tuple[FilingMetadataRemoval, ...]) -> str:
    """Restore text returned by :func:`mask_filing_metadata` from its removal records."""
    characters = list(masked_text)
    for removal in removals:
        if removal.end - removal.start != len(removal.text):
            msg = "Removal text does not fit its recorded offsets"
            raise ValueError(msg)
        characters[removal.start : removal.end] = removal.text
    return "".join(characters)


def filing_metadata_manifest(source_path: str, original_text: str) -> dict[str, Any]:
    """Build JSON-ready provenance for a masking operation without writing files."""
    masked = mask_filing_metadata(original_text)
    return {
        "source_path": source_path,
        "original_utf8_sha256": sha256(original_text.encode("utf-8")).hexdigest(),
        "masked_utf8_sha256": sha256(masked.text.encode("utf-8")).hexdigest(),
        "text_length": len(original_text),
        "removals": [
            {
                "kind": removal.kind.value,
                "start": removal.start,
                "end": removal.end,
                "text": removal.text,
            }
            for removal in masked.removals
        ],
    }


def _caption_docket_spans(text: str, stamp_spans: list[Span]) -> tuple[Span, ...]:
    heading = _COURT_HEADING.search(text, 0, _CAPTION_SEARCH_LIMIT)
    if heading is None:
        return ()
    candidate = _CAPTION_DOCKET.search(text, heading.end(), _CAPTION_SEARCH_LIMIT)
    if candidate is None:
        return ()

    docket_start, docket_end = candidate.span("docket")
    if any(span.start <= docket_start and docket_end <= span.end for span in stamp_spans):
        return ()
    return (Span(start=docket_start, end=docket_end),)
