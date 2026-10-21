"""Normalize the saved pinpoint span for each full citation."""

from mellea_lrc.config.field_extraction import FieldExtractionConfig, FieldSource
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.extraction.predict_root_fields import SUBSTAGE as PREDICTION
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.read_root_fields.resolve_pin_cites"


def resolve_pin_cites(
    document: Document,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Select a cached span; parsing belongs to the preceding prediction substage."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    require_structure(document)
    if PREDICTION not in document.substage_runs:
        raise ValueError("Predict root fields before normalizing citation fields")
    fields = field_extraction or FieldExtractionConfig.from_env()
    citations = document.full_locators
    predictions = []
    for citation in citations:
        rule_prediction = citation.get_rule_field_prediction()
        predictions.append(
            fields.validate_prediction(citation.get_ml_field_prediction())
            if fields.pin_cite is FieldSource.ML
            else rule_prediction
        )
    for citation, prediction in zip(citations, predictions, strict=True):
        updated = citation.record(SUBSTAGE)
        if prediction.pin_cite is not None:
            updated = updated.with_pin_cite(document.text, prediction.pin_cite)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
