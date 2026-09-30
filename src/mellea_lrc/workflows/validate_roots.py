"""Compose the currently implemented root-identity validation stages."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.validation import (
    docket_root_lookup,
    docket_root_lookup_review,
    govinfo_docket_lookup,
    govinfo_docket_lookup_review,
    reporter_root_lookup,
    reporter_root_lookup_ambiguous,
    reporter_root_lookup_ambiguous_dockets,
    reporter_root_lookup_ambiguous_llm,
    reporter_root_lookup_review,
    reporter_root_lookup_unique_llm,
)
from mellea_lrc.validation.body_search._courtlistener import CourtListenerBodyClient
from mellea_lrc.validation.docket_root_lookup import STAGE as DOCKET_LOOKUP_STAGE
from mellea_lrc.validation.docket_root_lookup_review import STAGE as DOCKET_REVIEW_STAGE
from mellea_lrc.validation.govinfo_docket_lookup import STAGE as GOVINFO_LOOKUP_STAGE
from mellea_lrc.validation.govinfo_docket_lookup_review import STAGE as GOVINFO_REVIEW_STAGE
from mellea_lrc.validation.reporter_root_lookup import STAGE as REPORTER_LOOKUP_STAGE
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE as AMBIGUOUS_REVIEW_STAGE
from mellea_lrc.validation.reporter_root_lookup_ambiguous_dockets import (
    STAGE as AMBIGUOUS_DOCKETS_STAGE,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm import STAGE as AMBIGUOUS_LLM_STAGE
from mellea_lrc.validation.reporter_root_lookup_review import STAGE as UNIQUE_REVIEW_STAGE
from mellea_lrc.validation.reporter_root_lookup_unique_llm import STAGE as UNIQUE_LLM_STAGE
from mellea_lrc.workflows.corroborate_root_locator_bodies import corroborate_root_locator_bodies


async def validate_roots(
    document: Document,
    *,
    retrospective_date: date | None = None,
    courtlistener_client: CourtListenerBodyClient | None = None,
    checkpoint: Callable[[Document], None] | None = None,
) -> Document:
    """Run root lookup and then locator-anchored third-party body review.

    Reporter search and large-candidate review remain future stages. The
    optional cutoff limits each later body-evidence item to material available
    on or before that date.
    """
    stages = (
        REPORTER_LOOKUP_STAGE,
        UNIQUE_REVIEW_STAGE,
        AMBIGUOUS_DOCKETS_STAGE,
        AMBIGUOUS_REVIEW_STAGE,
        UNIQUE_LLM_STAGE,
        AMBIGUOUS_LLM_STAGE,
        DOCKET_LOOKUP_STAGE,
        DOCKET_REVIEW_STAGE,
        GOVINFO_LOOKUP_STAGE,
        GOVINFO_REVIEW_STAGE,
    )
    completed = tuple(stage for stage in document.stage_runs if stage in stages)
    if completed != stages[: len(completed)]:
        raise ValueError("Validation checkpoint must end at a completed stage boundary")
    for stage, run in (
        (REPORTER_LOOKUP_STAGE, reporter_root_lookup),
        (UNIQUE_REVIEW_STAGE, reporter_root_lookup_review),
        (AMBIGUOUS_DOCKETS_STAGE, reporter_root_lookup_ambiguous_dockets),
        (AMBIGUOUS_REVIEW_STAGE, reporter_root_lookup_ambiguous),
    ):
        if stage not in document.stage_runs:
            document = run(document)
            if checkpoint is not None:
                checkpoint(document)
    for stage, run in (
        (UNIQUE_LLM_STAGE, reporter_root_lookup_unique_llm),
        (AMBIGUOUS_LLM_STAGE, reporter_root_lookup_ambiguous_llm),
    ):
        if stage not in document.stage_runs:
            document = await run(document)
            if checkpoint is not None:
                checkpoint(document)
    if DOCKET_LOOKUP_STAGE not in document.stage_runs:
        document = docket_root_lookup(document)
        if checkpoint is not None:
            checkpoint(document)
    if DOCKET_REVIEW_STAGE not in document.stage_runs:
        document = await docket_root_lookup_review(document)
        if checkpoint is not None:
            checkpoint(document)
    if GOVINFO_LOOKUP_STAGE not in document.stage_runs:
        document = govinfo_docket_lookup(document)
        if checkpoint is not None:
            checkpoint(document)
    if GOVINFO_REVIEW_STAGE not in document.stage_runs:
        document = await govinfo_docket_lookup_review(document)
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
    return await corroborate_root_locator_bodies(
        document, retrospective_date=retrospective_date, **client_kwargs
    )
