"""Read each short reporter's case name before its complete colocation group."""

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.extraction.context.short_field_predictions import with_short_rule_prediction
from mellea_lrc.extraction.predict_short_reporter_fields_ml import SUBSTAGE as ML_PREDICTION
from mellea_lrc.extraction.resolve_short_reporter_colocations import SUBSTAGE as COLOCATION_SUBSTAGE
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.grow_short_reporter_leaves.resolve_case_names"


def resolve_short_reporter_case_names(
    document: Document,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Quote one preceding name for every group member, including singletons.

    Colocation only relaxes the name-reading window. Each citation keeps its
    own locator, pinpoint, normalized reporter and eventual root attachment.
    """
    require_leaves(document, SUBSTAGE)
    if COLOCATION_SUBSTAGE not in document.substage_runs:
        raise ValueError("Resolve short reporter colocations before reading case names")
    fields = field_extraction or FieldExtractionConfig.from_env()
    if fields.case_name == "ml" and ML_PREDICTION not in document.substage_runs:
        raise ValueError("Predict ML short reporter fields before using ML case-name spans")
    for citation in document.short_reporters:
        updated = with_short_rule_prediction(citation.record(SUBSTAGE), document)
        prediction = (
            fields.validate_prediction(updated.get_ml_field_prediction())
            if fields.case_name == "ml"
            else updated.get_rule_field_prediction()
        )
        if prediction.case_name is not None:
            updated = updated.with_case_name(document.text, prediction.case_name)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
