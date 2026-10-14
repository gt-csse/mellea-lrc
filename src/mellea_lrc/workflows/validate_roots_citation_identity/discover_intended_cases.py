"""Compose the intended case discovery stage of validate_roots_citation_identity."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.body_search.retrieve_courtlistener_intended_case_filings import (
    SUBSTAGE as RETRIEVAL_1,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_intended_case_filings import (
    retrieve_courtlistener_intended_case_filings,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_intended_case_opinions import (
    SUBSTAGE as RETRIEVAL_0,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_intended_case_opinions import (
    retrieve_courtlistener_intended_case_opinions,
)
from mellea_lrc.validation.body_search.retrieve_govinfo_intended_case_opinions import SUBSTAGE as RETRIEVAL_2
from mellea_lrc.validation.body_search.retrieve_govinfo_intended_case_opinions import (
    retrieve_govinfo_intended_case_opinions,
)
from mellea_lrc.validation.review_intended_cases_llm import SUBSTAGE as REVIEW
from mellea_lrc.validation.review_intended_cases_llm import review_intended_cases_llm
from mellea_lrc.workflows import (
    Checkpoint,
    _finish_stage,
    _require_body_search_cutoff,
    _require_stage_prefix,
    _save_checkpoint,
)

STAGE = "validate_roots_citation_identity.discover_intended_cases"


async def discover_intended_cases(
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
        document = retrieve_courtlistener_intended_case_opinions(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        _save_checkpoint(document, checkpoint)
    if RETRIEVAL_1 not in document.substage_runs:
        document = retrieve_courtlistener_intended_case_filings(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        _save_checkpoint(document, checkpoint)
    if RETRIEVAL_2 not in document.substage_runs:
        document = retrieve_govinfo_intended_case_opinions(document, retrospective_date=retrospective_date)
        _save_checkpoint(document, checkpoint)
    if REVIEW not in document.substage_runs:
        document = await review_intended_cases_llm(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
