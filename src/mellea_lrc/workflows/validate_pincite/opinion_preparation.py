"""Compose the opinion preparation stage of validate_pincite."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_root_opinion_page_index import SUBSTAGE as INDEX
from mellea_lrc.validation.reporter_root_opinion_page_index import index_reporter_root_opinion_pages
from mellea_lrc.validation.reporter_root_opinion_retrieval import SUBSTAGE as RETRIEVAL
from mellea_lrc.validation.reporter_root_opinion_retrieval import (
    OpinionClient,
    reporter_root_opinion_retrieval,
)
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_pincite.opinion_preparation"


def prepare_root_opinions(
    document: Document,
    *,
    client: OpinionClient | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if RETRIEVAL not in document.substage_runs:
        document = reporter_root_opinion_retrieval(document, client=client)
        _save_checkpoint(document, checkpoint)
    if INDEX not in document.substage_runs:
        document = index_reporter_root_opinion_pages(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
