"""Compose the opinion preparation stage of validate_all_citation_content."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.index_reporter_root_opinion_pages import SUBSTAGE as INDEX
from mellea_lrc.validation.index_reporter_root_opinion_pages import index_reporter_root_opinion_pages
from mellea_lrc.validation.retrieve_reporter_root_opinions import SUBSTAGE as RETRIEVAL
from mellea_lrc.validation.retrieve_reporter_root_opinions import (
    OpinionClient,
    retrieve_reporter_root_opinions,
)
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_all_citation_content.prepare_root_opinions"


def prepare_root_opinions(
    document: Document,
    *,
    client: OpinionClient | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if RETRIEVAL not in document.substage_runs:
        document = retrieve_reporter_root_opinions(document, client=client)
        _save_checkpoint(document, checkpoint)
    if INDEX not in document.substage_runs:
        document = index_reporter_root_opinion_pages(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
