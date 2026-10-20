"""Save one joint ML span prediction per short reporter without applying fields."""

from __future__ import annotations

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.extraction.context.short_field_predictions import with_short_rule_prediction
from mellea_lrc.extraction.resolve_short_reporter_colocations import SUBSTAGE as COLOCATION_SUBSTAGE
from mellea_lrc.field_extraction.ml_cite import MLCiteReader, get_ml_reader
from mellea_lrc.model.citations.ml_field_reading import MLFieldReading
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.grow_short_reporter_leaves.predict_fields_ml"


def predict_short_reporter_fields_ml(
    document: Document, config: FieldExtractionConfig, *, reader: MLCiteReader | None = None
) -> Document:
    """Retain all four raw spans; later short-form readers choose applicable fields."""
    require_leaves(document, SUBSTAGE)
    if not config.ml_enabled:
        raise ValueError("ML field prediction requires ML extraction to be enabled")
    if COLOCATION_SUBSTAGE not in document.substage_runs:
        raise ValueError("Resolve short reporter colocations before predicting contextual fields")
    if "grow_leaves.grow_short_reporter_leaves.resolve_case_names" in document.substage_runs:
        raise ValueError("Predict ML fields before reading short reporter case names")
    citations = document.short_reporters
    if any(citation.ml_field_reading is not None for citation in citations):
        raise ValueError("ML field reading is already recorded")
    if not citations:
        return document.complete_substage(SUBSTAGE)
    model = reader if reader is not None else get_ml_reader(config.ml_reader_config())
    predictions = model.predict_many(
        document.text, tuple(citation.short_locator_span for citation in citations)
    )
    if len(predictions) != len(citations):
        raise ValueError("ML field reader must return one prediction for each locator")
    for citation, prediction in zip(citations, predictions, strict=True):
        prediction.validate_source(document.text, citation.short_locator_span)
        config.validate_prediction(prediction)
    for citation, prediction in zip(citations, predictions, strict=True):
        updated = with_short_rule_prediction(citation.record(SUBSTAGE), document)
        updated = updated.with_ml_field_reading(
            MLFieldReading(node_id=updated.nodes[-1].id, prediction=prediction)
        )
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
