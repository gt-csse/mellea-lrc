"""Save rule and optional ML span predictions before normalizing root fields."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.field_predictions import _root_field_contexts
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.field_extraction.ml_cite import MLCiteReader, get_ml_reader
from mellea_lrc.field_extraction.rule import RuleFieldReader
from mellea_lrc.model.citations.ml_field_reading import MLFieldReading
from mellea_lrc.model.citations.rule_field_reading import RuleFieldReading
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.read_root_fields.predict_fields"


def predict_root_fields(
    document: Document,
    config: FieldExtractionConfig | None = None,
    *,
    rules: ExtractionRules | None = None,
    ml_reader: MLCiteReader | None = None,
) -> Document:
    """Run each enabled reader once on the same prepared citation contexts."""
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
        raise ValueError("Predict fields before normalizing citation fields")
    fields = config or FieldExtractionConfig.from_env()
    if ml_reader is not None and not fields.ml_enabled:
        raise ValueError("An ML reader requires ML extraction to be enabled")
    citations = document.full_locators
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

    policy = rules or stable()
    contexts = _root_field_contexts(document, policy, fields)
    rule_predictions = RuleFieldReader().predict_many(contexts)
    if len(rule_predictions) != len(citations):
        raise ValueError("Rule field reader must return one prediction for each locator")
    for context, prediction in zip(contexts, rule_predictions, strict=True):
        prediction.validate_source(document.text, context.locator)
        prediction.validate_context(context)

    ml_predictions = None
    if model is not None:
        ml_predictions = model.predict_many(contexts)
        if len(ml_predictions) != len(citations):
            raise ValueError("ML field reader must return one prediction for each locator")
        for context, prediction in zip(contexts, ml_predictions, strict=True):
            prediction.validate_source(document.text, context.locator)
            prediction.validate_context(context)
            fields.validate_prediction(prediction)

    # Neither reader can partially commit a batch. Every raw result is checked
    # before a citation gains a node; field normalization happens only later.
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
