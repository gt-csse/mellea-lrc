"""Compose the field reading stage of grow_roots."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.predict_root_fields import SUBSTAGE as PREDICTION
from mellea_lrc.extraction.predict_root_fields import predict_root_fields
from mellea_lrc.extraction.resolve_case_names import SUBSTAGE as CASE_NAMES
from mellea_lrc.extraction.resolve_case_names import resolve_case_names
from mellea_lrc.extraction.resolve_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.extraction.resolve_colocations import resolve_colocations
from mellea_lrc.extraction.resolve_courts import SUBSTAGE as COURTS
from mellea_lrc.extraction.resolve_courts import resolve_courts
from mellea_lrc.extraction.resolve_dates import SUBSTAGE as DATES
from mellea_lrc.extraction.resolve_dates import resolve_dates
from mellea_lrc.extraction.resolve_docket_entries import SUBSTAGE as DOCKET_ENTRIES
from mellea_lrc.extraction.resolve_docket_entries import resolve_docket_entries
from mellea_lrc.extraction.resolve_pin_cites import SUBSTAGE as PIN_CITES
from mellea_lrc.extraction.resolve_pin_cites import resolve_pin_cites
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_roots.read_root_fields"


def read_root_fields(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    field_extraction: FieldExtractionConfig | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    fields = field_extraction or FieldExtractionConfig.from_env()
    _require_stage_prefix(document, STAGE)
    config = rules or stable()
    for citation in document.full_locators:
        if citation.rule_field_reading is not None:
            saved = citation.rule_field_reading
            if (saved.case_name_window, saved.post_locator_window) != (
                config.case_name_window,
                config.post_locator_window,
            ):
                raise ValueError("Rule prediction windows changed; rewind before predicting fields")
    if fields.ml_enabled:
        for citation in document.full_locators:
            if citation.ml_field_reading is not None:
                fields.validate_prediction(citation.ml_field_reading.prediction)
    fields.validate_runtime()
    if DOCKET_ENTRIES not in document.substage_runs:
        document = resolve_docket_entries(document)
        _save_checkpoint(document, checkpoint)
    if COLOCATIONS not in document.substage_runs:
        document = resolve_colocations(document, config)
        _save_checkpoint(document, checkpoint)
    if PREDICTION not in document.substage_runs:
        document = predict_root_fields(document, fields, rules=config)
        _save_checkpoint(document, checkpoint)
    if CASE_NAMES not in document.substage_runs:
        document = resolve_case_names(document, field_extraction=fields)
        _save_checkpoint(document, checkpoint)
    if COURTS not in document.substage_runs:
        document = resolve_courts(document, field_extraction=fields)
        _save_checkpoint(document, checkpoint)
    if DATES not in document.substage_runs:
        document = resolve_dates(document, field_extraction=fields)
        _save_checkpoint(document, checkpoint)
    if PIN_CITES not in document.substage_runs:
        document = resolve_pin_cites(document, field_extraction=fields)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
