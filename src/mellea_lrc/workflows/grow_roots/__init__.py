"""Grow citation roots through locator discovery, field reading, and formation."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.hunt_docket_locators.review import DocketSiteReviewer
from mellea_lrc.extraction.reassign_docket_roots_llm.reviewer import DocketRootReviewer
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, validate_workflow_checkpoint
from mellea_lrc.workflows.grow_roots.discover_root_locators import discover_root_locators
from mellea_lrc.workflows.grow_roots.form_root_groups import form_root_groups
from mellea_lrc.workflows.grow_roots.read_root_fields import read_root_fields


async def grow_roots(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
    field_extraction: FieldExtractionConfig | None = None,
    hunt_dockets: bool = False,
    reviewer: DocketSiteReviewer | None = None,
    review_docket_roots: bool = False,
    docket_root_reviewer: DocketRootReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    """Compose three semantic stages, resuming from each committed boundary."""
    fields = field_extraction or FieldExtractionConfig.from_env()
    omitted = (() if hunt_dockets else ("grow_roots.discover_root_locators.hunt_docket_locators",)) + (
        () if review_docket_roots else ("grow_roots.form_root_groups.reassign_docket_roots_llm",)
    )
    validate_workflow_checkpoint(document, "grow_roots", omitted_substages=omitted)
    config = rules or stable()
    for citation in document.full_locators:
        if citation.rule_field_reading is not None:
            saved = citation.rule_field_reading
            if (saved.case_name_window, saved.post_locator_window) != (
                config.case_name_window,
                config.post_locator_window,
            ):
                raise ValueError("Rule prediction windows changed; rewind before predicting fields")
    if fields.ml_enabled:
        for citation in document.full_locators:
            if citation.ml_field_reading is not None:
                fields.validate_prediction(citation.ml_field_reading.prediction)
    fields.validate_runtime()
    if "grow_roots.discover_root_locators" not in document.stage_runs:
        document = await discover_root_locators(
            document, hunt_dockets=hunt_dockets, reviewer=reviewer, checkpoint=checkpoint
        )
    if "grow_roots.read_root_fields" not in document.stage_runs:
        document = read_root_fields(document, config, field_extraction=fields, checkpoint=checkpoint)
    if "grow_roots.form_root_groups" not in document.stage_runs:
        document = await form_root_groups(
            document,
            review_docket_roots=review_docket_roots,
            docket_root_reviewer=docket_root_reviewer,
            checkpoint=checkpoint,
        )
    return document


__all__ = ["discover_root_locators", "form_root_groups", "grow_roots", "read_root_fields"]
