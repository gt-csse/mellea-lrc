"""Find independent citation text in fetched CourtListener opinions."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.citations.body_evidence import BodySearch, BodySource
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import (
    CourtListenerBodyClient,
    _search_root,
    body_client,
)
from mellea_lrc.validation.body_search.common import roots_for_body_search

SUBSTAGE = "validate_roots_citation_identity.corroborate_locator_bodies.retrieve_courtlistener_opinions"


def retrieve_courtlistener_locator_opinions(
    document: Document,
    *,
    retrospective_date: date | None = None,
    client: CourtListenerBodyClient | None = None,
) -> Document:
    """Save locator matches from independently fetched opinion bodies."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if "grow_roots.form_root_groups.form_roots" not in document.substage_runs:
        raise ValueError("Form roots before body search")
    with body_client(client) as get_client:
        for root in roots_for_body_search(document):
            recorded = root.record(SUBSTAGE)
            result = _search_root(
                document.text,
                recorded,
                source=BodySource.COURTLISTENER_OPINION,
                retrospective_date=retrospective_date,
                get_client=get_client,
            )
            if not isinstance(result, BodySearch):
                raise ValueError("A locator search must return locator evidence")
            document = document.replace_citation(recorded.with_body_search(result))
    return document.complete_substage(SUBSTAGE)
