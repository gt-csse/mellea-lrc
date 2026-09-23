"""Form canonical roots after all first-pass locator fields are readable."""

from __future__ import annotations

import re

from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.operations import CitationField


def _reporter_key(citation: FullReporterCitation) -> tuple[str, ...] | None:
    if not (citation.volume and citation.reporter and citation.page):
        return None
    return (
        "reporter",
        citation.volume.casefold(),
        re.sub(r"\s+", "", citation.reporter.casefold()),
        citation.page.casefold(),
    )


def _docket_key(citation: FullDocketCitation) -> tuple[str, ...] | None:
    # A courtless docket is not globally unique. Preserve its occurrence as a
    # separate root until identity validation or search supplies that context.
    if not citation.court or not citation.docket_number:
        return None
    return ("docket", citation.court, citation.docket_number.casefold())


def form_roots(document: Document) -> Document:
    """Deduplicate exact reporter keys and court-qualified docket keys only.

    Colocation is deliberately absent from the key: proximity helps read fields
    but does not establish shared identity.
    """
    stage = "roots"
    if stage in document.completed_stages:
        return document
    if "colocations" not in document.completed_stages:
        raise ValueError("Resolve colocations before forming roots")
    known: dict[tuple[str, ...], str] = {}
    for citation in document.full_locators:
        key = _reporter_key(citation) if isinstance(citation, FullReporterCitation) else _docket_key(citation)
        root_id = known.setdefault(key, citation.id) if key is not None else citation.id
        document = document.update_fields(stage, citation.id, {CitationField.ROOT_ID: root_id})
    return document.complete(stage)
