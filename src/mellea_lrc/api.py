"""Outer compositional API for Mellea-LRC stages.

This module is the stable boundary for callers composing individual stages.
It intentionally does not choose an end-to-end pipeline: callers begin with a
``Document`` or a serialized checkpoint, then explicitly invoke the
``Document -> Document`` or ``ValidatedDocument -> ValidatedDocument`` stage
they want. The command-line interface is the separate end-to-end entrypoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.extraction.adjudication import hunt_docket_locators
from mellea_lrc.extraction.locator_stages import (
    find_docket_locators,
    find_full_reporter_locators,
    mark_full_reporter_locator_hunting_skipped,
    resolve_colocations,
)
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.stages import (
    resolve_case_names,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)
from mellea_lrc.extraction.types import Document
from mellea_lrc.validation.pipeline import (
    initialize_full_reporter_locator_identity,
)
from mellea_lrc.validation.pipeline import (
    run_full_reporter_locator_identity as _run_full_reporter_locator_identity,
)
from mellea_lrc.validation.types import ValidatedDocument

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient

__all__ = [
    "Document",
    "ExtractionRules",
    "ValidatedDocument",
    "find_docket_locators",
    "find_full_reporter_locators",
    "hunt_docket_locators",
    "mark_full_reporter_locator_hunting_skipped",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_pin_cites",
    "run_full_reporter_locator_identity",
    "stable",
    "start_full_reporter_locator_identity",
]


def start_full_reporter_locator_identity(document: Document) -> ValidatedDocument:
    """Create the serializable checkpoint for full reporter-locator identity."""
    return initialize_full_reporter_locator_identity(document)


async def run_full_reporter_locator_identity(
    checkpoint: ValidatedDocument,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> ValidatedDocument:
    """Run the reporter-only identity stage and return its updated checkpoint."""
    return await _run_full_reporter_locator_identity(checkpoint, client=client, session=session)
