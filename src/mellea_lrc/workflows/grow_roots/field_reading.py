"""Compose the field reading stage of grow_roots."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction.case_names import SUBSTAGE as CASE_NAMES
from mellea_lrc.extraction.case_names import resolve_case_names
from mellea_lrc.extraction.colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.extraction.colocations import resolve_colocations
from mellea_lrc.extraction.courts import SUBSTAGE as COURTS
from mellea_lrc.extraction.courts import resolve_courts
from mellea_lrc.extraction.dates import SUBSTAGE as DATES
from mellea_lrc.extraction.dates import resolve_dates
from mellea_lrc.extraction.docket_entries import SUBSTAGE as DOCKET_ENTRIES
from mellea_lrc.extraction.docket_entries import resolve_docket_entries
from mellea_lrc.extraction.pin_cites import SUBSTAGE as PIN_CITES
from mellea_lrc.extraction.pin_cites import resolve_pin_cites
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_roots.field_reading"


def read_root_fields(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    config = rules or stable()
    if DOCKET_ENTRIES not in document.substage_runs:
        document = resolve_docket_entries(document)
        _save_checkpoint(document, checkpoint)
    if COLOCATIONS not in document.substage_runs:
        document = resolve_colocations(document, config)
        _save_checkpoint(document, checkpoint)
    if CASE_NAMES not in document.substage_runs:
        document = resolve_case_names(document, config)
        _save_checkpoint(document, checkpoint)
    if COURTS not in document.substage_runs:
        document = resolve_courts(document, config)
        _save_checkpoint(document, checkpoint)
    if DATES not in document.substage_runs:
        document = resolve_dates(document, config)
        _save_checkpoint(document, checkpoint)
    if PIN_CITES not in document.substage_runs:
        document = resolve_pin_cites(document, config)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
