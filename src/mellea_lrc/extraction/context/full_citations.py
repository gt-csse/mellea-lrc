"""Extraction-substage guards and contextual text readings."""

from __future__ import annotations

import re

from mellea_lrc.model.citation_windows import after
from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.fields.date import YEAR_RE
from mellea_lrc.model.document import Document

_PAREN = re.compile(r"\((?P<body>[^()\r\n]{0,100})\)")


def require_structure(document: Document) -> None:
    """Require the pipeline position where contextual fields can be read."""
    if "grow_roots.field_reading.colocations" not in document.substage_runs:
        raise ValueError("Resolve colocations before reading contextual fields")
    if "grow_roots.root_formation.rule" in document.substage_runs:
        raise ValueError("Read contextual fields before forming roots")


def dated_parenthetical(
    document: Document, citation: FullCitationVariant, limit: int
) -> tuple[re.Match[str], int] | None:
    """Find a nearby parenthetical containing a year, unless a new sentence intervenes."""
    after_text, start = after(document, citation, limit)
    for match in _PAREN.finditer(after_text):
        if not YEAR_RE.search(match.group("body")):
            continue
        if re.search(r"\.\s+[A-Z]", after_text[: match.start()]):
            return None
        return match, start
    return None
