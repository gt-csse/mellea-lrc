"""Normalize the saved name of each short reporter citation."""

from mellea_lrc.config.field_extraction import FieldExtractionConfig, FieldSource
from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.extraction.predict_short_reporter_fields import SUBSTAGE as PREDICTION
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.grow_short_reporter_leaves.resolve_case_names"


def resolve_short_reporter_case_names(
    document: Document,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Choose the cached backend result without rerunning either span reader."""
    require_leaves(document, SUBSTAGE)
    if PREDICTION not in document.substage_runs:
        raise ValueError("Predict short reporter fields before normalizing case names")
    fields = field_extraction or FieldExtractionConfig.from_env()
    citations = document.short_reporters
    predictions = []
    for citation in citations:
        rule_prediction = citation.get_rule_field_prediction()
        predictions.append(
            fields.validate_prediction(citation.get_ml_field_prediction())
            if fields.case_name is FieldSource.ML
            else rule_prediction
        )
    for citation, prediction in zip(citations, predictions, strict=True):
        updated = citation.record(SUBSTAGE)
        if prediction.case_name is not None:
            updated = updated.with_case_name(document.text, prediction.case_name)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
