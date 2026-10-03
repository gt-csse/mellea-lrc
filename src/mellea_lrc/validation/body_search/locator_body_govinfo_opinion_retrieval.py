"""Find independent USCOURTS opinion citations in individual GovInfo granules."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import date

from mellea_lrc.model.citations.body_evidence import BodySearch
from mellea_lrc.model.document import Document
from mellea_lrc.providers.govinfo import GovInfoClient
from mellea_lrc.validation.body_search._govinfo import GovInfoBodyClient, _search_root
from mellea_lrc.validation.body_search.common import roots_for_body_search

STAGE = "22_locator_body_govinfo_opinion_retrieval"


def locator_body_govinfo_opinion_retrieval(
    document: Document,
    *,
    client: GovInfoBodyClient | None = None,
    retrospective_date: date | None = None,
) -> Document:
    """Save locator matches from fetched USCOURTS opinion granules."""
    if STAGE in document.stage_runs:
        raise ValueError("GovInfo opinion body search has already completed")
    with ExitStack() as stack:
        service = client
        for root in roots_for_body_search(document):
            recorded = root.record(STAGE)
            if service is None:
                service = stack.enter_context(GovInfoClient())
            result = _search_root(document, recorded, service, retrospective_date)
            if not isinstance(result, BodySearch):
                raise ValueError("A locator search must return locator evidence")
            document = document.replace_citation(recorded.with_body_search(result))
    return document.complete(STAGE)
