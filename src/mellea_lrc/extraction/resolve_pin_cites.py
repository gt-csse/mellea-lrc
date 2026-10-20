"""Read pinpoint references adjacent to full locator sites."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.field_predictions import with_root_rule_prediction
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.extraction.predict_root_fields_ml import SUBSTAGE as ML_PREDICTION
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.read_root_fields.resolve_pin_cites"


def resolve_pin_cites(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Read an adjacent pin, retaining malformed continuations for later review."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    require_structure(document)
    fields = field_extraction or FieldExtractionConfig.from_env()
    if fields.pin_cite == "ml" and ML_PREDICTION not in document.substage_runs:
        raise ValueError("Predict ML root fields before using ML pinpoint spans")
    config = rules or stable()
    for citation in document.full_locators:
        updated = with_root_rule_prediction(citation.record(SUBSTAGE), document, config)
        prediction = (
            fields.validate_prediction(updated.get_ml_field_prediction())
            if fields.pin_cite == "ml"
            else updated.get_rule_field_prediction()
        )
        if prediction.pin_cite is not None:
            updated = updated.with_pin_cite(document.text, prediction.pin_cite)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
