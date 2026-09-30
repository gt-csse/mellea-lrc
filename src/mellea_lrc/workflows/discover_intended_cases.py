"""Search other citation fields after locator-body review has finished."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.body_search.intended_case_courtlistener_opinion_retrieval import (
    intended_case_courtlistener_opinion_retrieval,
)
from mellea_lrc.validation.body_search.intended_case_courtlistener_recap_retrieval import (
    intended_case_courtlistener_recap_retrieval,
)
from mellea_lrc.validation.body_search.intended_case_govinfo_opinion_retrieval import (
    intended_case_govinfo_opinion_retrieval,
)
from mellea_lrc.validation.intended_case_llm_selection import intended_case_llm_selection


async def discover_intended_cases(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
) -> Document:
    """Gather name-anchored evidence and save a possible intended authority.

    This workflow starts after locator-body review. Its final stage never
    admits the source citation's locator, even when the candidate fields fit.
    Each constituent Document-to-Document stage is independently callable.
    """
    client_kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
    document = intended_case_courtlistener_opinion_retrieval(
        document, retrospective_date=retrospective_date, **client_kwargs
    )
    document = intended_case_courtlistener_recap_retrieval(
        document, retrospective_date=retrospective_date, **client_kwargs
    )
    document = intended_case_govinfo_opinion_retrieval(document, retrospective_date=retrospective_date)
    return await intended_case_llm_selection(document)
