"""Compose the docket lookup stage of validate_roots_citation_identity."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.judge_root_identity_from_fields import SUBSTAGE as STEP_4
from mellea_lrc.validation.judge_root_identity_from_fields import judge_root_identity_from_fields
from mellea_lrc.validation.retrieve_courtlistener_docket_roots import SUBSTAGE as STEP_0
from mellea_lrc.validation.retrieve_courtlistener_docket_roots import (
    retrieve_courtlistener_docket_roots,
)
from mellea_lrc.validation.retrieve_govinfo_docket_roots import SUBSTAGE as STEP_2
from mellea_lrc.validation.retrieve_govinfo_docket_roots import retrieve_govinfo_docket_roots
from mellea_lrc.validation.review_courtlistener_docket_roots_llm import SUBSTAGE as STEP_1
from mellea_lrc.validation.review_courtlistener_docket_roots_llm import (
    review_courtlistener_docket_roots_llm,
)
from mellea_lrc.validation.review_govinfo_docket_roots_llm import SUBSTAGE as STEP_3
from mellea_lrc.validation.review_govinfo_docket_roots_llm import review_govinfo_docket_roots_llm
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_roots_citation_identity.lookup_docket_roots"


async def lookup_docket_roots(document: Document, *, checkpoint: Checkpoint | None = None) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if STEP_0 not in document.substage_runs:
        document = retrieve_courtlistener_docket_roots(document)
        _save_checkpoint(document, checkpoint)
    if STEP_1 not in document.substage_runs:
        document = await review_courtlistener_docket_roots_llm(document)
        _save_checkpoint(document, checkpoint)
    if STEP_2 not in document.substage_runs:
        document = retrieve_govinfo_docket_roots(document)
        _save_checkpoint(document, checkpoint)
    if STEP_3 not in document.substage_runs:
        document = await review_govinfo_docket_roots_llm(document)
        _save_checkpoint(document, checkpoint)
    if STEP_4 not in document.substage_runs:
        document = judge_root_identity_from_fields(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
