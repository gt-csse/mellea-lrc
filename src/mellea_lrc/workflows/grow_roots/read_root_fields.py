"""Compose the field reading stage of grow_roots."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.predict_root_fields_ml import SUBSTAGE as PREDICTION
from mellea_lrc.extraction.predict_root_fields_ml import predict_root_fields_ml
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
    _require_stage_prefix(document, STAGE, omitted_substages=() if fields.ml_enabled else (PREDICTION,))
    config = rules or stable()
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
    if fields.ml_enabled and PREDICTION not in document.substage_runs:
        document = predict_root_fields_ml(document, fields)
        _save_checkpoint(document, checkpoint)
    if CASE_NAMES not in document.substage_runs:
        document = resolve_case_names(document, config, field_extraction=fields)
        _save_checkpoint(document, checkpoint)
    if COURTS not in document.substage_runs:
        document = resolve_courts(document, config, field_extraction=fields)
        _save_checkpoint(document, checkpoint)
    if DATES not in document.substage_runs:
        document = resolve_dates(document, config, field_extraction=fields)
        _save_checkpoint(document, checkpoint)
    if PIN_CITES not in document.substage_runs:
        document = resolve_pin_cites(document, config, field_extraction=fields)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
