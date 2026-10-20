"""Save one joint ML span prediction per full locator without reading any field."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.field_predictions import with_root_rule_prediction
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.field_extraction.ml_cite import MLCiteReader, get_ml_reader
from mellea_lrc.model.citations.ml_field_reading import MLFieldReading
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.read_root_fields.predict_fields_ml"


def predict_root_fields_ml(
    document: Document,
    config: FieldExtractionConfig,
    *,
    rules: ExtractionRules | None = None,
    reader: MLCiteReader | None = None,
) -> Document:
    """Run all four span heads together and retain their source-relative results."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if not config.ml_enabled:
        raise ValueError("ML field prediction requires ML extraction to be enabled")
    require_structure(document)
    if any(
        name in document.substage_runs
        for name in (
            "grow_roots.read_root_fields.resolve_case_names",
            "grow_roots.read_root_fields.resolve_courts",
            "grow_roots.read_root_fields.resolve_dates",
            "grow_roots.read_root_fields.resolve_pin_cites",
        )
    ):
        raise ValueError("Predict ML fields before reading citation fields")
    citations = document.full_locators
    if any(citation.ml_field_reading is not None for citation in citations):
        raise ValueError("ML field reading is already recorded")
    if not citations:
        return document.complete_substage(SUBSTAGE)
    model = reader if reader is not None else get_ml_reader(config.ml_reader_config())
    predictions = model.predict_many(document.text, tuple(citation.locator_span for citation in citations))
    if len(predictions) != len(citations):
        raise ValueError("ML field reader must return one prediction for each locator")
    # Validate the complete batch before adding a decision node to any citation.
    for citation, prediction in zip(citations, predictions, strict=True):
        prediction.validate_source(document.text, citation.locator_span)
        config.validate_prediction(prediction)
    for citation, prediction in zip(citations, predictions, strict=True):
        updated = with_root_rule_prediction(citation.record(SUBSTAGE), document, rules or stable())
        updated = updated.with_ml_field_reading(
            MLFieldReading(node_id=updated.nodes[-1].id, prediction=prediction)
        )
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
