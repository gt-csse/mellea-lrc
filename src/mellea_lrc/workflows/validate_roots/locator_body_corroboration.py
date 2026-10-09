"""Compose the locator body corroboration stage of validate_roots."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.body_search.locator_body_courtlistener_opinion_retrieval import (
    SUBSTAGE as RETRIEVAL_0,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_opinion_retrieval import (
    locator_body_courtlistener_opinion_retrieval,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_recap_retrieval import (
    SUBSTAGE as RETRIEVAL_1,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_recap_retrieval import (
    locator_body_courtlistener_recap_retrieval,
)
from mellea_lrc.validation.body_search.locator_body_govinfo_opinion_retrieval import SUBSTAGE as RETRIEVAL_2
from mellea_lrc.validation.body_search.locator_body_govinfo_opinion_retrieval import (
    locator_body_govinfo_opinion_retrieval,
)
from mellea_lrc.validation.locator_body_llm_judgment import SUBSTAGE as REVIEW
from mellea_lrc.validation.locator_body_llm_judgment import locator_body_llm_judgment
from mellea_lrc.workflows import (
    Checkpoint,
    _finish_stage,
    _require_body_search_cutoff,
    _require_stage_prefix,
    _save_checkpoint,
)

STAGE = "validate_roots.locator_body_corroboration"


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
        document = locator_body_courtlistener_opinion_retrieval(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        _save_checkpoint(document, checkpoint)
    if RETRIEVAL_1 not in document.substage_runs:
        document = locator_body_courtlistener_recap_retrieval(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        _save_checkpoint(document, checkpoint)
    if RETRIEVAL_2 not in document.substage_runs:
        document = locator_body_govinfo_opinion_retrieval(document, retrospective_date=retrospective_date)
        _save_checkpoint(document, checkpoint)
    if REVIEW not in document.substage_runs:
        document = await locator_body_llm_judgment(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
