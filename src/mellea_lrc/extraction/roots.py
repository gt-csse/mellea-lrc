"""Form canonical roots after all first-pass locator fields are readable."""

from __future__ import annotations

import re

from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.history import latest
from mellea_lrc.model.document import Document


def _reporter_key(citation: FullReporterCitation) -> tuple[str, ...] | None:
    locator = citation.locator[-1].normalized
    return (
        "reporter",
        str(locator.volume),
        re.sub(r"\s+", "", locator.edition.casefold()),
        locator.page.casefold(),
    )


def _docket_key(citation: FullDocketCitation) -> tuple[str, ...] | None:
    # A courtless docket is not globally unique. Preserve its occurrence as a
    # separate root until identity validation or search supplies that context.
    court, docket_number = latest(citation.court), citation.locator[-1].normalized.docket_number
    if not court or not docket_number:
        return None
    return ("docket", court.id, docket_number.casefold())


def form_roots(document: Document) -> Document:
    """Deduplicate exact reporter keys and court-qualified docket keys only.

    Colocation is deliberately absent from the key: proximity helps read fields
    but does not establish shared identity.
    """
    stage = "roots"
    if stage in document.stage_runs:
        return document
    if "colocations" not in document.stage_runs:
        raise ValueError("Resolve colocations before forming roots")
    known: dict[tuple[str, ...], str] = {}
    for citation in document.full_locators:
        key = _reporter_key(citation) if isinstance(citation, FullReporterCitation) else _docket_key(citation)
        root_id = known.setdefault(key, citation.id) if key is not None else citation.id
        document = document.replace_citation(citation.record(stage).with_root(root_id))
    return document.complete(stage)
