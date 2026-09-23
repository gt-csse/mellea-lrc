"""Form canonical roots after all first-pass locator fields are readable."""

from __future__ import annotations

import re

from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.history import CitationField, latest
from mellea_lrc.model.document import Document


def _reporter_key(citation: FullReporterCitation) -> tuple[str, ...] | None:
    volume, reporter, page = latest(citation.volume), latest(citation.reporter), latest(citation.page)
    if not (volume and reporter and page):
        return None
    return (
        "reporter",
        volume.casefold(),
        re.sub(r"\s+", "", reporter.casefold()),
        page.casefold(),
    )


def _docket_key(citation: FullDocketCitation) -> tuple[str, ...] | None:
    # A courtless docket is not globally unique. Preserve its occurrence as a
    # separate root until identity validation or search supplies that context.
    court, docket_number = latest(citation.court), latest(citation.docket_number)
    if not court or not docket_number:
        return None
    return ("docket", court, docket_number.casefold())


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
