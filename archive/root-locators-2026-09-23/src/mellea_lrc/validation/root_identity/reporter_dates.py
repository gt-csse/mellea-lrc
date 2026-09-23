"""Reserved checkpoint after exact reporter identity's date comparison."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from mellea_lrc.validation.root_identity.reporter import UNIQUE_IDENTITY_STAGE

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.model.document import Document


EXACT_REPORTER_DATE_REVIEW_STAGE = "full_reporter_exact_date_review"


async def review_full_reporter_exact_dates(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    retrospective_date: date | None = None,
) -> Document:
    """Keep a resumable boundary without revising the simple date judgment.

    Exact identity compares complete dates when both sides have them, years
    when that is the shared precision, and makes no date judgment if either
    side lacks a date. This checkpoint is currently a deliberate no-op.
    """
    if UNIQUE_IDENTITY_STAGE not in document.passes:
        raise ValueError("Exact reporter date checkpoint requires unique reporter identity validation")
    if EXACT_REPORTER_DATE_REVIEW_STAGE in document.passes:
        return document
    # TODO: Revisit opinion versions and dated amendments without overriding
    # an already-issued identity decision or assuming the first cluster is final.
    del client, retrospective_date
    return document.evolve(passes=(*document.passes, EXACT_REPORTER_DATE_REVIEW_STAGE))
