"""Find independent citation text in fetched CourtListener RECAP documents."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import (
    CourtListenerBodyClient,
    run_courtlistener_body_search,
)

STAGE = "21_courtlistener_recap_body_search"


def courtlistener_recap_body_search(
    document: Document,
    *,
    retrospective_date: date | None = None,
    client: CourtListenerBodyClient | None = None,
) -> Document:
    """Save bounded RECAP searches and excerpts from fetched document bodies."""
    return run_courtlistener_body_search(
        document,
        stage=STAGE,
        source=BodySource.COURTLISTENER_RECAP,
        retrospective_date=retrospective_date,
        client=client,
    )
