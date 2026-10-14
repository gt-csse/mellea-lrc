"""Compose the reporter lookup stage of validate_roots_citation_identity."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.judge_ambiguous_reporter_roots_rule import (
    SUBSTAGE as AMBIGUOUS_RULE_JUDGMENT,
)
from mellea_lrc.validation.judge_ambiguous_reporter_roots_rule import (
    judge_ambiguous_reporter_roots_rule,
)
from mellea_lrc.validation.judge_unique_reporter_roots_rule import SUBSTAGE as UNIQUE_RULE_JUDGMENT
from mellea_lrc.validation.judge_unique_reporter_roots_rule import (
    judge_unique_reporter_roots_rule,
)
from mellea_lrc.validation.retrieve_reporter_root_clusters import SUBSTAGE as CLUSTER_RETRIEVAL
from mellea_lrc.validation.retrieve_reporter_root_clusters import (
    retrieve_reporter_root_clusters,
)
from mellea_lrc.validation.retrieve_reporter_root_dockets import SUBSTAGE as DOCKET_RETRIEVAL
from mellea_lrc.validation.retrieve_reporter_root_dockets import retrieve_reporter_root_dockets
from mellea_lrc.validation.review_ambiguous_reporter_roots_llm import (
    SUBSTAGE as AMBIGUOUS_LLM_JUDGMENT,
)
from mellea_lrc.validation.review_ambiguous_reporter_roots_llm import (
    review_ambiguous_reporter_roots_llm,
)
from mellea_lrc.validation.review_unique_reporter_roots_llm import SUBSTAGE as UNIQUE_LLM_JUDGMENT
from mellea_lrc.validation.review_unique_reporter_roots_llm import (
    review_unique_reporter_roots_llm,
)
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_roots_citation_identity.lookup_reporter_roots"


async def lookup_reporter_roots(document: Document, *, checkpoint: Checkpoint | None = None) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if CLUSTER_RETRIEVAL not in document.substage_runs:
        document = retrieve_reporter_root_clusters(document)
        _save_checkpoint(document, checkpoint)
    if DOCKET_RETRIEVAL not in document.substage_runs:
        document = retrieve_reporter_root_dockets(document)
        _save_checkpoint(document, checkpoint)
    if UNIQUE_RULE_JUDGMENT not in document.substage_runs:
        document = judge_unique_reporter_roots_rule(document)
        _save_checkpoint(document, checkpoint)
    if AMBIGUOUS_RULE_JUDGMENT not in document.substage_runs:
        document = judge_ambiguous_reporter_roots_rule(document)
        _save_checkpoint(document, checkpoint)
    if UNIQUE_LLM_JUDGMENT not in document.substage_runs:
        document = await review_unique_reporter_roots_llm(document)
        _save_checkpoint(document, checkpoint)
    if AMBIGUOUS_LLM_JUDGMENT not in document.substage_runs:
        document = await review_ambiguous_reporter_roots_llm(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
