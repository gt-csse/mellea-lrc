"""Compose the docket lookup stage of validate_roots."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.docket_root_lookup_courtlistener_llm_review import SUBSTAGE as STEP_1
from mellea_lrc.validation.docket_root_lookup_courtlistener_llm_review import (
    docket_root_lookup_courtlistener_llm_review,
)
from mellea_lrc.validation.docket_root_lookup_courtlistener_retrieval import SUBSTAGE as STEP_0
from mellea_lrc.validation.docket_root_lookup_courtlistener_retrieval import (
    docket_root_lookup_courtlistener_retrieval,
)
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review import SUBSTAGE as STEP_3
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review import docket_root_lookup_govinfo_llm_review
from mellea_lrc.validation.docket_root_lookup_govinfo_retrieval import SUBSTAGE as STEP_2
from mellea_lrc.validation.docket_root_lookup_govinfo_retrieval import docket_root_lookup_govinfo_retrieval
from mellea_lrc.validation.fields_aggregated_identity import SUBSTAGE as STEP_4
from mellea_lrc.validation.fields_aggregated_identity import fields_aggregated_identity
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_roots.docket_lookup"


async def lookup_docket_roots(document: Document, *, checkpoint: Checkpoint | None = None) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if STEP_0 not in document.substage_runs:
        document = docket_root_lookup_courtlistener_retrieval(document)
        _save_checkpoint(document, checkpoint)
    if STEP_1 not in document.substage_runs:
        document = await docket_root_lookup_courtlistener_llm_review(document)
        _save_checkpoint(document, checkpoint)
    if STEP_2 not in document.substage_runs:
        document = docket_root_lookup_govinfo_retrieval(document)
        _save_checkpoint(document, checkpoint)
    if STEP_3 not in document.substage_runs:
        document = await docket_root_lookup_govinfo_llm_review(document)
        _save_checkpoint(document, checkpoint)
    if STEP_4 not in document.substage_runs:
        document = fields_aggregated_identity(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
