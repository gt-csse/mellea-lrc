"""Compose the short reporter citations stage of grow_leaves."""

from __future__ import annotations

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.attribute_short_reporter_citations import SUBSTAGE as ATTRIBUTION
from mellea_lrc.extraction.attribute_short_reporter_citations import attribute_short_reporter_citations
from mellea_lrc.extraction.find_short_reporter_citations import SUBSTAGE as DISCOVERY
from mellea_lrc.extraction.find_short_reporter_citations import find_short_reporter_citations
from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewer
from mellea_lrc.extraction.predict_short_reporter_fields_ml import SUBSTAGE as PREDICTION
from mellea_lrc.extraction.predict_short_reporter_fields_ml import predict_short_reporter_fields_ml
from mellea_lrc.extraction.resolve_short_reporter_case_names import SUBSTAGE as CASE_NAMES
from mellea_lrc.extraction.resolve_short_reporter_case_names import resolve_short_reporter_case_names
from mellea_lrc.extraction.resolve_short_reporter_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.extraction.resolve_short_reporter_colocations import resolve_short_reporter_colocations
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_leaves.grow_short_reporter_leaves"


async def grow_short_reporter_leaves(
    document: Document,
    *,
    field_extraction: FieldExtractionConfig | None = None,
    review_leaves: bool = True,
    reviewer: LeafReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    fields = field_extraction or FieldExtractionConfig.from_env()
    _require_stage_prefix(document, STAGE, omitted_substages=() if fields.ml_enabled else (PREDICTION,))
    if fields.ml_enabled:
        for citation in document.short_reporters:
            if citation.ml_field_reading is not None:
                fields.validate_prediction(citation.ml_field_reading.prediction)
    fields.validate_runtime()
    if DISCOVERY not in document.substage_runs:
        document = find_short_reporter_citations(document)
        _save_checkpoint(document, checkpoint)
    if COLOCATIONS not in document.substage_runs:
        document = resolve_short_reporter_colocations(document)
        _save_checkpoint(document, checkpoint)
    if fields.ml_enabled and PREDICTION not in document.substage_runs:
        document = predict_short_reporter_fields_ml(document, fields)
        _save_checkpoint(document, checkpoint)
    if CASE_NAMES not in document.substage_runs:
        document = resolve_short_reporter_case_names(document, field_extraction=fields)
        _save_checkpoint(document, checkpoint)
    if ATTRIBUTION not in document.substage_runs:
        document = await attribute_short_reporter_citations(document, review=review_leaves, reviewer=reviewer)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
