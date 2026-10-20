"""Read case-name fields before each full locator site."""

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.field_predictions import with_root_rule_prediction
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.extraction.predict_root_fields_ml import SUBSTAGE as ML_PREDICTION
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.read_root_fields.resolve_case_names"


def resolve_case_names(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Read a name before each citation site, never through another locator."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    require_structure(document)
    fields = field_extraction or FieldExtractionConfig.from_env()
    if fields.case_name == "ml" and ML_PREDICTION not in document.substage_runs:
        raise ValueError("Predict ML root fields before using ML case-name spans")
    config = rules or stable()
    for citation in document.full_locators:
        updated = with_root_rule_prediction(citation.record(SUBSTAGE), document, config)
        prediction = (
            fields.validate_prediction(updated.get_ml_field_prediction())
            if fields.case_name == "ml"
            else updated.get_rule_field_prediction()
        )
        if prediction.case_name is not None:
            updated = updated.with_case_name(document.text, prediction.case_name)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
