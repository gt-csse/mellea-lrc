"""Grow citation roots through locator discovery, field reading, and formation."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules
from mellea_lrc.extraction.docket_root_llm_reassignment.reviewer import DocketRootReviewer
from mellea_lrc.extraction.docket_site_hunting.review import DocketSiteReviewer
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _require_workflow_prefix
from mellea_lrc.workflows.grow_roots.field_reading import read_root_fields
from mellea_lrc.workflows.grow_roots.locator_discovery import discover_root_locators
from mellea_lrc.workflows.grow_roots.root_formation import form_root_groups


async def grow_roots(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
    hunt_dockets: bool = False,
    reviewer: DocketSiteReviewer | None = None,
    review_docket_roots: bool = False,
    docket_root_reviewer: DocketRootReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    """Compose three semantic stages, resuming from each committed boundary."""
    omitted = (() if hunt_dockets else ("grow_roots.locator_discovery.docket_hunting",)) + (
        () if review_docket_roots else ("grow_roots.root_formation.docket_llm_reassignment",)
    )
    _require_workflow_prefix(document, "grow_roots", omitted_substages=omitted)
    if "grow_roots.locator_discovery" not in document.stage_runs:
        document = await discover_root_locators(
            document, hunt_dockets=hunt_dockets, reviewer=reviewer, checkpoint=checkpoint
        )
    if "grow_roots.field_reading" not in document.stage_runs:
        document = read_root_fields(document, rules, checkpoint=checkpoint)
    if "grow_roots.root_formation" not in document.stage_runs:
        document = await form_root_groups(
            document,
            review_docket_roots=review_docket_roots,
            docket_root_reviewer=docket_root_reviewer,
            checkpoint=checkpoint,
        )
    return document


__all__ = ["discover_root_locators", "form_root_groups", "grow_roots", "read_root_fields"]
