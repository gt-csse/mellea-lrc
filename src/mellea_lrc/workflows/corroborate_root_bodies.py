"""Compose provider discovery and one citation-level body corroboration review."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_corroboration import body_corroboration_review
from mellea_lrc.validation.body_search.courtlistener_opinion import courtlistener_opinion_body_search
from mellea_lrc.validation.body_search.courtlistener_recap import courtlistener_recap_body_search
from mellea_lrc.validation.body_search.govinfo import govinfo_opinion_body_search


async def corroborate_root_bodies(document: Document, *, retrospective_date: date | None = None) -> Document:
    """Retrieve three body sources, then issue one grounded cross-provider verdict.

    The optional cutoff is applied to each individual opinion or filing before
    its excerpt is admitted. Omitting it permits later independent citations.
    Each constituent stage is public and independently checkpointed, so a
    caller can run or inspect any provider before invoking the shared review.
    """
    document = courtlistener_opinion_body_search(document, retrospective_date=retrospective_date)
    document = courtlistener_recap_body_search(document, retrospective_date=retrospective_date)
    document = govinfo_opinion_body_search(document, retrospective_date=retrospective_date)
    return await body_corroboration_review(document)
