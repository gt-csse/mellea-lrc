"""Cache one ML inference for full-locator citations, independently of rule reading."""

from __future__ import annotations

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.extraction.field_readers.ml_cite import MLCiteReader, get_ml_reader
from mellea_lrc.model.citations.ml_field_reading import MLFieldReading
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.read_root_fields.predict_fields_ml"


def predict_root_fields_ml(
    document: Document,
    config: FieldExtractionConfig | None = None,
    *,
    reader: MLCiteReader | None = None,
) -> Document:
    """Read all four ML heads once; later field stages select and normalize spans."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
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
        raise ValueError("Predict ML fields before normalizing citation fields")
    fields = config or FieldExtractionConfig.from_env()
    if not fields.ml_enabled:
        raise ValueError("ML field prediction requires ML_ENABLED=true")
    citations = document.full_locators
    if any(citation.ml_field_reading is not None for citation in citations):
        raise ValueError("ML field prediction is already recorded")
    # Resolve the configured checkpoint before inference or any durable update.
    model = reader if reader is not None else get_ml_reader(fields.ml_reader_config())
    if not citations:
        return document.complete_substage(SUBSTAGE)
    locators = tuple(citation.locator_span for citation in citations)
    predictions = model.predict_many(document.text, locators)
    if len(predictions) != len(citations):
        raise ValueError("ML field reader must return one prediction for each locator")
    for locator, prediction in zip(locators, predictions, strict=True):
        prediction.validate_source(document.text, locator)
        fields.validate_prediction(prediction)
    # Validate the complete batch before appending any citation-local history.
    for citation, prediction in zip(citations, predictions, strict=True):
        updated = citation.record(SUBSTAGE)
        updated = updated.with_ml_field_reading(
            MLFieldReading(node_id=updated.nodes[-1].id, prediction=prediction)
        )
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
