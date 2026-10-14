"""Compose the locator discovery stage of grow_roots."""

from __future__ import annotations

from mellea_lrc.extraction.find_docket_locators import SUBSTAGE as DOCKET
from mellea_lrc.extraction.find_docket_locators import find_docket_locators
from mellea_lrc.extraction.find_full_reporter_locators import SUBSTAGE as REPORTER
from mellea_lrc.extraction.find_full_reporter_locators import find_full_reporter_locators
from mellea_lrc.extraction.hunt_docket_locators import SUBSTAGE as HUNTING
from mellea_lrc.extraction.hunt_docket_locators import hunt_docket_locators
from mellea_lrc.extraction.hunt_docket_locators.review import DocketSiteReviewer
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_roots.discover_root_locators"


async def discover_root_locators(
    document: Document,
    *,
    hunt_dockets: bool = False,
    reviewer: DocketSiteReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=() if hunt_dockets else (HUNTING,))
    if REPORTER not in document.substage_runs:
        document = find_full_reporter_locators(document)
        _save_checkpoint(document, checkpoint)
    if DOCKET not in document.substage_runs:
        document = find_docket_locators(document)
        _save_checkpoint(document, checkpoint)
    if hunt_dockets and HUNTING not in document.substage_runs:
        document = await hunt_docket_locators(document, reviewer=reviewer)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
