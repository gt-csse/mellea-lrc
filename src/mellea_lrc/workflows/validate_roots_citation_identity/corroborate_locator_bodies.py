"""Compose the locator body corroboration stage of validate_roots_citation_identity."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.body_search.retrieve_courtlistener_locator_filings import (
    SUBSTAGE as RETRIEVAL_1,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_locator_filings import (
    retrieve_courtlistener_locator_filings,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_locator_opinions import (
    SUBSTAGE as RETRIEVAL_0,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_locator_opinions import (
    retrieve_courtlistener_locator_opinions,
)
from mellea_lrc.validation.body_search.retrieve_govinfo_locator_opinions import SUBSTAGE as RETRIEVAL_2
from mellea_lrc.validation.body_search.retrieve_govinfo_locator_opinions import (
    retrieve_govinfo_locator_opinions,
)
from mellea_lrc.validation.review_locator_bodies_llm import SUBSTAGE as REVIEW
from mellea_lrc.validation.review_locator_bodies_llm import review_locator_bodies_llm
from mellea_lrc.workflows import (
    Checkpoint,
    _finish_stage,
    _require_body_search_cutoff,
    _require_stage_prefix,
    _save_checkpoint,
)

STAGE = "validate_roots_citation_identity.corroborate_locator_bodies"


async def corroborate_locator_bodies(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    _require_body_search_cutoff(document, retrospective_date)
    client_kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
    if RETRIEVAL_0 not in document.substage_runs:
        document = retrieve_courtlistener_locator_opinions(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        _save_checkpoint(document, checkpoint)
    if RETRIEVAL_1 not in document.substage_runs:
        document = retrieve_courtlistener_locator_filings(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        _save_checkpoint(document, checkpoint)
    if RETRIEVAL_2 not in document.substage_runs:
        document = retrieve_govinfo_locator_opinions(document, retrospective_date=retrospective_date)
        _save_checkpoint(document, checkpoint)
    if REVIEW not in document.substage_runs:
        document = await review_locator_bodies_llm(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
