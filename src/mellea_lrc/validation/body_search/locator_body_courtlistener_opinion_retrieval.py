"""Find independent citation text in fetched CourtListener opinions."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import (
    CourtListenerBodyClient,
    run_courtlistener_body_search,
)

SUBSTAGE = "validate_roots.locator_body_corroboration.courtlistener_opinion_retrieval"


def locator_body_courtlistener_opinion_retrieval(
    document: Document,
    *,
    retrospective_date: date | None = None,
    client: CourtListenerBodyClient | None = None,
) -> Document:
    """Save locator matches from independently fetched opinion bodies."""
    return run_courtlistener_body_search(
        document,
        substage=SUBSTAGE,
        source=BodySource.COURTLISTENER_OPINION,
        retrospective_date=retrospective_date,
        client=client,
    )
