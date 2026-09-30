"""Search fetched CourtListener opinions by the cited case name."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import (
    CourtListenerBodyClient,
    run_courtlistener_field_body_search,
)

STAGE = "24_intended_case_courtlistener_opinion_retrieval"


def intended_case_courtlistener_opinion_retrieval(
    document: Document,
    *,
    retrospective_date: date | None = None,
    client: CourtListenerBodyClient | None = None,
) -> Document:
    """Save case-name matches from independently fetched opinion bodies."""
    return run_courtlistener_field_body_search(
        document,
        stage=STAGE,
        source=BodySource.COURTLISTENER_OPINION,
        retrospective_date=retrospective_date,
        client=client,
    )
