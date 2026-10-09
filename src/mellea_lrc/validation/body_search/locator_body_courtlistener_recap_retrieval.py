"""Find independent citation text in fetched CourtListener RECAP documents."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import (
    CourtListenerBodyClient,
    run_courtlistener_body_search,
)

SUBSTAGE = "validate_roots.locator_body_corroboration.courtlistener_recap_retrieval"


def locator_body_courtlistener_recap_retrieval(
    document: Document,
    *,
    retrospective_date: date | None = None,
    client: CourtListenerBodyClient | None = None,
) -> Document:
    """Save locator matches from independently fetched RECAP filings."""
    return run_courtlistener_body_search(
        document,
        substage=SUBSTAGE,
        source=BodySource.COURTLISTENER_RECAP,
        retrospective_date=retrospective_date,
        client=client,
    )
