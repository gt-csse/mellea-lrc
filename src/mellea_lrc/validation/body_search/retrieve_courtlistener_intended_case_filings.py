"""Search fetched CourtListener RECAP documents by the cited case name."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.citations.field_body_evidence import FieldBodySearch
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import (
    CourtListenerBodyClient,
    _search_root,
    body_client,
)

SUBSTAGE = "validate_roots_citation_identity.discover_intended_cases.retrieve_courtlistener_filings"
ROUTE = "validate_roots_citation_identity.discover_intended_cases.retrieve_courtlistener_opinions"


def retrieve_courtlistener_intended_case_filings(
    document: Document,
    *,
    retrospective_date: date | None = None,
    client: CourtListenerBodyClient | None = None,
) -> Document:
    """Save case-name matches from independently fetched RECAP bodies."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if (
        "validate_roots_citation_identity.corroborate_locator_bodies.review_locator_bodies_llm"
        not in document.substage_runs
    ):
        raise ValueError("Complete locator body review before field body search")
    with body_client(client) as get_client:
        for root in document.roots:
            if root.next_substage != ROUTE:
                continue
            recorded = root.record(SUBSTAGE)
            result = _search_root(
                document.text,
                recorded,
                source=BodySource.COURTLISTENER_RECAP,
                retrospective_date=retrospective_date,
                get_client=get_client,
                anchor_kind="case_name",
            )
            if not isinstance(result, FieldBodySearch):
                raise ValueError("A case-name search must return field evidence")
            document = document.replace_citation(recorded.with_field_body_search(result))
    return document.complete_substage(SUBSTAGE)
