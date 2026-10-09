"""Compose the reporter lookup stage of validate_roots."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm_judgment import (
    SUBSTAGE as AMBIGUOUS_LLM_JUDGMENT,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm_judgment import (
    reporter_root_lookup_ambiguous_llm_judgment,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_rule_judgment import (
    SUBSTAGE as AMBIGUOUS_RULE_JUDGMENT,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_rule_judgment import (
    reporter_root_lookup_ambiguous_rule_judgment,
)
from mellea_lrc.validation.reporter_root_lookup_cluster_retrieval import SUBSTAGE as CLUSTER_RETRIEVAL
from mellea_lrc.validation.reporter_root_lookup_cluster_retrieval import (
    reporter_root_lookup_cluster_retrieval,
)
from mellea_lrc.validation.reporter_root_lookup_docket_retrieval import SUBSTAGE as DOCKET_RETRIEVAL
from mellea_lrc.validation.reporter_root_lookup_docket_retrieval import reporter_root_lookup_docket_retrieval
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment import SUBSTAGE as UNIQUE_LLM_JUDGMENT
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment import (
    reporter_root_lookup_unique_llm_judgment,
)
from mellea_lrc.validation.reporter_root_lookup_unique_rule_judgment import SUBSTAGE as UNIQUE_RULE_JUDGMENT
from mellea_lrc.validation.reporter_root_lookup_unique_rule_judgment import (
    reporter_root_lookup_unique_rule_judgment,
)
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_roots.reporter_lookup"


async def lookup_reporter_roots(document: Document, *, checkpoint: Checkpoint | None = None) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if CLUSTER_RETRIEVAL not in document.substage_runs:
        document = reporter_root_lookup_cluster_retrieval(document)
        _save_checkpoint(document, checkpoint)
    if DOCKET_RETRIEVAL not in document.substage_runs:
        document = reporter_root_lookup_docket_retrieval(document)
        _save_checkpoint(document, checkpoint)
    if UNIQUE_RULE_JUDGMENT not in document.substage_runs:
        document = reporter_root_lookup_unique_rule_judgment(document)
        _save_checkpoint(document, checkpoint)
    if AMBIGUOUS_RULE_JUDGMENT not in document.substage_runs:
        document = reporter_root_lookup_ambiguous_rule_judgment(document)
        _save_checkpoint(document, checkpoint)
    if UNIQUE_LLM_JUDGMENT not in document.substage_runs:
        document = await reporter_root_lookup_unique_llm_judgment(document)
        _save_checkpoint(document, checkpoint)
    if AMBIGUOUS_LLM_JUDGMENT not in document.substage_runs:
        document = await reporter_root_lookup_ambiguous_llm_judgment(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
