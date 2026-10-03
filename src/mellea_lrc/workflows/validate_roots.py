"""Compose the currently implemented root-identity validation stages."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation import (
    docket_root_lookup_courtlistener_llm_review,
    docket_root_lookup_courtlistener_retrieval,
    docket_root_lookup_govinfo_llm_review,
    docket_root_lookup_govinfo_retrieval,
    intended_case_courtlistener_opinion_retrieval,
    intended_case_courtlistener_recap_retrieval,
    intended_case_govinfo_opinion_retrieval,
    intended_case_llm_selection,
    reporter_root_lookup_ambiguous_llm_judgment,
    reporter_root_lookup_ambiguous_rule_judgment,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_lookup_docket_retrieval,
    reporter_root_lookup_unique_llm_judgment,
    reporter_root_lookup_unique_rule_judgment,
)
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.body_search.intended_case_courtlistener_opinion_retrieval import (
    STAGE as FIELD_OPINION_STAGE,
)
from mellea_lrc.validation.body_search.intended_case_courtlistener_recap_retrieval import (
    STAGE as FIELD_RECAP_STAGE,
)
from mellea_lrc.validation.body_search.intended_case_govinfo_opinion_retrieval import (
    STAGE as FIELD_GOVINFO_STAGE,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_opinion_retrieval import (
    STAGE as LOCATOR_OPINION_STAGE,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_opinion_retrieval import (
    locator_body_courtlistener_opinion_retrieval,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_recap_retrieval import (
    STAGE as LOCATOR_RECAP_STAGE,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_recap_retrieval import (
    locator_body_courtlistener_recap_retrieval,
)
from mellea_lrc.validation.body_search.locator_body_govinfo_opinion_retrieval import (
    STAGE as LOCATOR_GOVINFO_STAGE,
)
from mellea_lrc.validation.body_search.locator_body_govinfo_opinion_retrieval import (
    locator_body_govinfo_opinion_retrieval,
)
from mellea_lrc.validation.docket_root_lookup_courtlistener_llm_review import STAGE as DOCKET_REVIEW_STAGE
from mellea_lrc.validation.docket_root_lookup_courtlistener_retrieval import STAGE as DOCKET_LOOKUP_STAGE
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review import STAGE as GOVINFO_REVIEW_STAGE
from mellea_lrc.validation.docket_root_lookup_govinfo_retrieval import STAGE as GOVINFO_LOOKUP_STAGE
from mellea_lrc.validation.intended_case_llm_selection import STAGE as FIELD_REVIEW_STAGE
from mellea_lrc.validation.locator_body_llm_judgment import STAGE as LOCATOR_REVIEW_STAGE
from mellea_lrc.validation.locator_body_llm_judgment import locator_body_llm_judgment
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm_judgment import STAGE as AMBIGUOUS_LLM_STAGE
from mellea_lrc.validation.reporter_root_lookup_ambiguous_rule_judgment import STAGE as AMBIGUOUS_RULE_STAGE
from mellea_lrc.validation.reporter_root_lookup_cluster_retrieval import STAGE as REPORTER_CLUSTER_STAGE
from mellea_lrc.validation.reporter_root_lookup_docket_retrieval import (
    STAGE as REPORTER_DOCKET_STAGE,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment import STAGE as UNIQUE_LLM_STAGE
from mellea_lrc.validation.reporter_root_lookup_unique_rule_judgment import STAGE as UNIQUE_RULE_STAGE


def _require_completed_prefix(document: Document, stages: tuple[str, ...], name: str) -> None:
    completed = tuple(stage for stage in document.stage_runs if stage in stages)
    if completed != stages[: len(completed)]:
        raise ValueError(f"{name} checkpoint must end at a completed stage boundary")


async def validate_roots(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
    checkpoint: Callable[[Document], None] | None = None,
    search_other_fields: bool = False,
) -> Document:
    """Run root lookup and then locator-anchored third-party body review.

    Reporter search and large-candidate review remain future stages. The
    optional cutoff limits each later body-evidence item to material available
    on or before that date. Set search_other_fields=True to continue through
    intended-case discovery (stages 24-27); it cannot confirm the source locator.
    """
    stages = (
        REPORTER_CLUSTER_STAGE,
        REPORTER_DOCKET_STAGE,
        UNIQUE_RULE_STAGE,
        AMBIGUOUS_RULE_STAGE,
        UNIQUE_LLM_STAGE,
        AMBIGUOUS_LLM_STAGE,
        DOCKET_LOOKUP_STAGE,
        DOCKET_REVIEW_STAGE,
        GOVINFO_LOOKUP_STAGE,
        GOVINFO_REVIEW_STAGE,
    )
    _require_completed_prefix(document, stages, "Validation")
    for stage, run in (
        (REPORTER_CLUSTER_STAGE, reporter_root_lookup_cluster_retrieval),
        (REPORTER_DOCKET_STAGE, reporter_root_lookup_docket_retrieval),
        (UNIQUE_RULE_STAGE, reporter_root_lookup_unique_rule_judgment),
        (AMBIGUOUS_RULE_STAGE, reporter_root_lookup_ambiguous_rule_judgment),
    ):
        if stage not in document.stage_runs:
            document = run(document)
            if checkpoint is not None:
                checkpoint(document)
    for stage, run in (
        (UNIQUE_LLM_STAGE, reporter_root_lookup_unique_llm_judgment),
        (AMBIGUOUS_LLM_STAGE, reporter_root_lookup_ambiguous_llm_judgment),
    ):
        if stage not in document.stage_runs:
            document = await run(document)
            if checkpoint is not None:
                checkpoint(document)
    if DOCKET_LOOKUP_STAGE not in document.stage_runs:
        document = docket_root_lookup_courtlistener_retrieval(document)
        if checkpoint is not None:
            checkpoint(document)
    if DOCKET_REVIEW_STAGE not in document.stage_runs:
        document = await docket_root_lookup_courtlistener_llm_review(document)
        if checkpoint is not None:
            checkpoint(document)
    if GOVINFO_LOOKUP_STAGE not in document.stage_runs:
        document = docket_root_lookup_govinfo_retrieval(document)
        if checkpoint is not None:
            checkpoint(document)
    if GOVINFO_REVIEW_STAGE not in document.stage_runs:
        document = await docket_root_lookup_govinfo_llm_review(document)
        if checkpoint is not None:
            checkpoint(document)
    # A selected docket record now routes to fields_aggregated_identity. That
    # decision can run after this checkpoint; body search leaves queued roots
    # alone until the field judgments have been combined.
    # TODO: Review opinion evidence for a securely identified record whose
    # cited court or date still disagrees. The cluster's linked docket may
    # contain imported court metadata that identifies the case but assigns
    # the wrong court. A docket filing date marks case initiation, not an
    # opinion date. The cluster date is shared across grouped opinions, while
    # the specific subopinion or order may print another signed or issued
    # date. Retrieve that opinion/subopinion text or its original court PDF
    # and compare its header and signature before changing field judgments.
    client_kwargs = {"courtlistener_client": courtlistener_client} if courtlistener_client is not None else {}
    if checkpoint is not None:
        client_kwargs["checkpoint"] = checkpoint
    document = await _validate_locator_bodies(
        document, retrospective_date=retrospective_date, **client_kwargs
    )
    if search_other_fields:
        document = await _discover_intended_case(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
    return document


async def _validate_locator_bodies(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
    checkpoint: Callable[[Document], None] | None = None,
) -> Document:
    """Retrieve locator matches from three sources, then review them.

    The optional cutoff is applied to each individual opinion or filing before
    its excerpt is admitted. Omitting it permits later independent citations.
    This is a private part of validate_roots. The callback persists its
    completed Document before the next retrieval or review begins.
    """
    stages = (LOCATOR_OPINION_STAGE, LOCATOR_RECAP_STAGE, LOCATOR_GOVINFO_STAGE, LOCATOR_REVIEW_STAGE)
    _require_completed_prefix(document, stages, "Locator-body")
    client_kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
    if LOCATOR_OPINION_STAGE not in document.stage_runs:
        document = locator_body_courtlistener_opinion_retrieval(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        if checkpoint is not None:
            checkpoint(document)
    if LOCATOR_RECAP_STAGE not in document.stage_runs:
        document = locator_body_courtlistener_recap_retrieval(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        if checkpoint is not None:
            checkpoint(document)
    if LOCATOR_GOVINFO_STAGE not in document.stage_runs:
        document = locator_body_govinfo_opinion_retrieval(document, retrospective_date=retrospective_date)
        if checkpoint is not None:
            checkpoint(document)
    if LOCATOR_REVIEW_STAGE not in document.stage_runs:
        document = await locator_body_llm_judgment(document)
        if checkpoint is not None:
            checkpoint(document)
    return document


async def _discover_intended_case(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
    checkpoint: Callable[[Document], None] | None = None,
) -> Document:
    """Continue root validation by finding the intended case from other fields.

    This evidence cannot establish that the source locator is correct. Each
    completed stage is resumable and belongs to the validate_roots report.
    """
    stages = (FIELD_OPINION_STAGE, FIELD_RECAP_STAGE, FIELD_GOVINFO_STAGE, FIELD_REVIEW_STAGE)
    _require_completed_prefix(document, stages, "Intended-case")
    client_kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
    for stage, run in (
        (FIELD_OPINION_STAGE, intended_case_courtlistener_opinion_retrieval),
        (FIELD_RECAP_STAGE, intended_case_courtlistener_recap_retrieval),
    ):
        if stage not in document.stage_runs:
            document = run(document, retrospective_date=retrospective_date, **client_kwargs)
            if checkpoint is not None:
                checkpoint(document)
    if FIELD_GOVINFO_STAGE not in document.stage_runs:
        document = intended_case_govinfo_opinion_retrieval(document, retrospective_date=retrospective_date)
        if checkpoint is not None:
            checkpoint(document)
    if FIELD_REVIEW_STAGE not in document.stage_runs:
        document = await intended_case_llm_selection(document)
        if checkpoint is not None:
            checkpoint(document)
    return document
