"""Compose locator-only body discovery and its citation-level review."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.body_search.courtlistener_opinion import (
    courtlistener_opinion_locator_body_search,
)
from mellea_lrc.validation.body_search.courtlistener_recap import (
    courtlistener_recap_locator_body_search,
)
from mellea_lrc.validation.body_search.govinfo import govinfo_opinion_locator_body_search
from mellea_lrc.validation.locator_body_review import review_locator_body_evidence


async def corroborate_root_locator_bodies(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
) -> Document:
    """Retrieve locator matches from three sources, then review them.

    The optional cutoff is applied to each individual opinion or filing before
    its excerpt is admitted. Omitting it permits later independent citations.
    Each constituent stage is public and independently checkpointed, so a
    caller can run or inspect any provider before invoking the shared review.
    """
    client_kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
    document = courtlistener_opinion_locator_body_search(
        document, retrospective_date=retrospective_date, **client_kwargs
    )
    document = courtlistener_recap_locator_body_search(
        document, retrospective_date=retrospective_date, **client_kwargs
    )
    document = govinfo_opinion_locator_body_search(document, retrospective_date=retrospective_date)
    return await review_locator_body_evidence(document)
