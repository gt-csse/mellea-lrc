"""Compose locator-only body discovery and its citation-level review."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.body_search.courtlistener_opinion import STAGE as OPINION_STAGE
from mellea_lrc.validation.body_search.courtlistener_opinion import (
    courtlistener_opinion_locator_body_search,
)
from mellea_lrc.validation.body_search.courtlistener_recap import STAGE as RECAP_STAGE
from mellea_lrc.validation.body_search.courtlistener_recap import (
    courtlistener_recap_locator_body_search,
)
from mellea_lrc.validation.body_search.govinfo import STAGE as GOVINFO_STAGE
from mellea_lrc.validation.body_search.govinfo import govinfo_opinion_locator_body_search
from mellea_lrc.validation.locator_body_review import STAGE as REVIEW_STAGE
from mellea_lrc.validation.locator_body_review import review_locator_body_evidence


async def corroborate_root_locator_bodies(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
    checkpoint: Callable[[Document], None] | None = None,
) -> Document:
    """Retrieve locator matches from three sources, then review them.

    The optional cutoff is applied to each individual opinion or filing before
    its excerpt is admitted. Omitting it permits later independent citations.
    Each constituent stage is public; the optional callback persists its
    completed Document before the next retrieval or review begins.
    """
    stages = (OPINION_STAGE, RECAP_STAGE, GOVINFO_STAGE, REVIEW_STAGE)
    completed = tuple(stage for stage in document.stage_runs if stage in stages)
    if completed != stages[: len(completed)]:
        raise ValueError("Locator-body checkpoint must end at a completed stage boundary")
    client_kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
    if OPINION_STAGE not in document.stage_runs:
        document = courtlistener_opinion_locator_body_search(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        if checkpoint is not None:
            checkpoint(document)
    if RECAP_STAGE not in document.stage_runs:
        document = courtlistener_recap_locator_body_search(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        if checkpoint is not None:
            checkpoint(document)
    if GOVINFO_STAGE not in document.stage_runs:
        document = govinfo_opinion_locator_body_search(document, retrospective_date=retrospective_date)
        if checkpoint is not None:
            checkpoint(document)
    if REVIEW_STAGE not in document.stage_runs:
        document = await review_locator_body_evidence(document)
        if checkpoint is not None:
            checkpoint(document)
    return document
