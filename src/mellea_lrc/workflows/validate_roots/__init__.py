"""Validate root identity through reporter lookup, docket lookup, and bodies."""

from __future__ import annotations

from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.workflows import Checkpoint, _require_body_search_cutoff, _require_workflow_prefix
from mellea_lrc.workflows.validate_roots.docket_lookup import lookup_docket_roots
from mellea_lrc.workflows.validate_roots.intended_case_discovery import discover_intended_cases
from mellea_lrc.workflows.validate_roots.locator_body_corroboration import corroborate_locator_bodies
from mellea_lrc.workflows.validate_roots.reporter_lookup import lookup_reporter_roots


async def validate_roots(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
    checkpoint: Checkpoint | None = None,
    search_other_fields: bool = False,
) -> Document:
    """Resume lookup and body review, optionally discovering an intended case.

    Every atomic retrieval or judgment and every completed group is sent to
    the checkpoint callback before the next operation begins. Intended-case
    discovery identifies a possible intended case without confirming the
    source locator.
    """
    _require_workflow_prefix(document, "validate_roots")
    _require_body_search_cutoff(document, retrospective_date)
    if "validate_roots.reporter_lookup" not in document.stage_runs:
        document = await lookup_reporter_roots(document, checkpoint=checkpoint)
    if "validate_roots.docket_lookup" not in document.stage_runs:
        document = await lookup_docket_roots(document, checkpoint=checkpoint)
    if "validate_roots.locator_body_corroboration" not in document.stage_runs:
        document = await corroborate_locator_bodies(
            document,
            retrospective_date=retrospective_date,
            courtlistener_client=courtlistener_client,
            checkpoint=checkpoint,
        )
    if search_other_fields and "validate_roots.intended_case_discovery" not in document.stage_runs:
        document = await discover_intended_cases(
            document,
            retrospective_date=retrospective_date,
            courtlistener_client=courtlistener_client,
            checkpoint=checkpoint,
        )
    return document


__all__ = [
    "corroborate_locator_bodies",
    "discover_intended_cases",
    "lookup_docket_roots",
    "lookup_reporter_roots",
    "validate_roots",
]
