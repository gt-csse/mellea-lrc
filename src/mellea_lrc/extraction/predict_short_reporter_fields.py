"""Save rule and optional ML span predictions for short reporter citations."""

from __future__ import annotations

from mellea_lrc.config.extraction import stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.field_predictions import short_field_contexts
from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.extraction.resolve_short_reporter_colocations import SUBSTAGE as COLOCATION_SUBSTAGE
from mellea_lrc.field_extraction.ml_cite import MLCiteReader, get_ml_reader
from mellea_lrc.field_extraction.rule import RuleFieldReader
from mellea_lrc.model.citations.ml_field_reading import MLFieldReading
from mellea_lrc.model.citations.rule_field_reading import RuleFieldReading
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.grow_short_reporter_leaves.predict_fields"


def predict_short_reporter_fields(
    document: Document, config: FieldExtractionConfig | None = None, *, ml_reader: MLCiteReader | None = None
) -> Document:
    """Run both backends before downstream short-name normalization."""
    require_leaves(document, SUBSTAGE)
    if COLOCATION_SUBSTAGE not in document.substage_runs:
        raise ValueError("Resolve short reporter colocations before predicting contextual fields")
    if "grow_leaves.grow_short_reporter_leaves.resolve_case_names" in document.substage_runs:
        raise ValueError("Predict fields before normalizing short reporter case names")
    fields = config or FieldExtractionConfig.from_env()
    if ml_reader is not None and not fields.ml_enabled:
        raise ValueError("An ML reader requires ML extraction to be enabled")
    citations = document.short_reporters
    if any(
        citation.rule_field_reading is not None or citation.ml_field_reading is not None
        for citation in citations
    ):
        raise ValueError("Field prediction is already recorded")
    # Resolve an enabled model before either backend starts extracting spans.
    model = None
    if fields.ml_enabled:
        model = ml_reader if ml_reader is not None else get_ml_reader(fields.ml_reader_config())
    if not citations:
        return document.complete_substage(SUBSTAGE)

    policy = stable()
    locators = tuple(citation.short_locator_span for citation in citations)
    contexts = short_field_contexts(document)
    rule_predictions = RuleFieldReader(
        case_name_window=policy.case_name_window,
        post_locator_window=policy.post_locator_window,
    ).predict_many(document.text, locators, contexts=contexts)
    if len(rule_predictions) != len(citations):
        raise ValueError("Rule field reader must return one prediction for each locator")
    for locator, prediction in zip(locators, rule_predictions, strict=True):
        prediction.validate_source(document.text, locator)

    ml_predictions = None
    if model is not None:
        ml_predictions = model.predict_many(document.text, locators, contexts=contexts)
        if len(ml_predictions) != len(citations):
            raise ValueError("ML field reader must return one prediction for each locator")
        for locator, prediction in zip(locators, ml_predictions, strict=True):
            prediction.validate_source(document.text, locator)
            fields.validate_prediction(prediction)

    for index, citation in enumerate(citations):
        updated = citation.record(SUBSTAGE)
        updated = updated.with_rule_field_reading(
            RuleFieldReading(
                node_id=updated.nodes[-1].id,
                prediction=rule_predictions[index],
                case_name_window=policy.case_name_window,
                post_locator_window=policy.post_locator_window,
            )
        )
        if ml_predictions is not None:
            updated = updated.with_ml_field_reading(
                MLFieldReading(node_id=updated.nodes[-1].id, prediction=ml_predictions[index])
            )
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
