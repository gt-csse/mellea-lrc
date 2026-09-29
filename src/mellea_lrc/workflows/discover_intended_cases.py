"""Search other citation fields after locator-body review has finished."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.body_search.courtlistener_opinion_fields import (
    courtlistener_opinion_field_body_search,
)
from mellea_lrc.validation.body_search.courtlistener_recap_fields import (
    courtlistener_recap_field_body_search,
)
from mellea_lrc.validation.body_search.govinfo_fields import govinfo_opinion_field_body_search
from mellea_lrc.validation.field_body_review import review_intended_case_body_evidence


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
    document = courtlistener_opinion_field_body_search(
        document, retrospective_date=retrospective_date, **client_kwargs
    )
    document = courtlistener_recap_field_body_search(
        document, retrospective_date=retrospective_date, **client_kwargs
    )
    document = govinfo_opinion_field_body_search(document, retrospective_date=retrospective_date)
    return await review_intended_case_body_evidence(document)
