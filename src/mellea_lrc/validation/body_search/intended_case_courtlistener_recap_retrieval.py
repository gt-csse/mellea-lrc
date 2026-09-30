"""Search fetched CourtListener RECAP documents by the cited case name."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import (
    CourtListenerBodyClient,
    run_courtlistener_field_body_search,
)

STAGE = "25_intended_case_courtlistener_recap_retrieval"


def intended_case_courtlistener_recap_retrieval(
    document: Document,
    *,
    retrospective_date: date | None = None,
    client: CourtListenerBodyClient | None = None,
) -> Document:
    """Save case-name matches from independently fetched RECAP bodies."""
    return run_courtlistener_field_body_search(
        document,
        stage=STAGE,
        source=BodySource.COURTLISTENER_RECAP,
        retrospective_date=retrospective_date,
        client=client,
    )
