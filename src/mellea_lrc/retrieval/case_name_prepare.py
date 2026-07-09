"""LLM-backed preparation for not-found case-name candidate search."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mellea import generative
from mellea.core import ValidationResult
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel

from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.assessment.model_options import structured_model_options
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.llm import LlmResponseFormat, llm_api_config_from_env
from mellea_lrc.retrieval.types import (
    CaseNamePreparationStatus,
    CaseNameSearchPreparation,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.extraction.types import ExtractedCitation

PREPARATION_CONTEXT_BEFORE_CHARS = 320
PREPARATION_MAX_TOKENS = 512
COMPLETE_PARTY_COUNT = 2
JSON_OBJECT_HINT = (
    'Return exactly one JSON object with key "result" containing fields: '
    '"classification" ("complete_case_name", "partial_case_name", or "no_case_name"), '
    '"plaintiff" (string copied exactly from local_context or null), '
    '"defendant" (string copied exactly from local_context or null), '
    '"case_name" (optional reconstructed "Plaintiff v. Defendant" when both parties are available, otherwise null), '
    'and "reason" (brief explanation). '
    'Example: {"result": {"classification": "complete_case_name", '
    '"plaintiff": "Smith", "defendant": "Jones", '
    '"case_name": "Smith v. Jones", "reason": "The parties appear immediately before the locator."}}.'
)


class _PreparedCaseName(BaseModel):
    classification: str
    plaintiff: str | None = None
    defendant: str | None = None
    case_name: str | None = None
    reason: str | None = None


@generative
async def _prepare_case_name(
    local_context: str,
    locator: str,
    extracted_plaintiff: str,
    extracted_defendant: str,
    output_format_hint: str,
) -> _PreparedCaseName:
    """Prepare parties for case-name search for the specific locator.

    The locator marks the citation we are trying to recover after exact lookup
    failed. Extract only parties that are bound to this locator. The parties or
    copied case name usually appear before the locator. A local context may
    contain multiple citations; do not borrow parties from a different citation.

    Prefer returning plaintiff and defendant as separate copied strings. The
    party strings must be copied from local_context, but case_name may be a
    reconstruction from those copied parties. Do not invent, normalize, or
    complete party names. If the case name is broken by whitespace or newlines,
    copy the party strings as they appear. Return classification
    "complete_case_name" only when both parties are present. Use
    "partial_case_name" for exactly one party and "no_case_name" when no
    parties tied to this locator are present.

    extracted_plaintiff and extracted_defendant are weak hints from the parser,
    not facts. They may be missing or wrong.

    If output_format_hint is "none", follow the JSON schema passed to you.
    Otherwise, return JSON exactly matching the shape described by
    output_format_hint.
    """


async def prepare_case_name_for_search(
    session: MelleaSession,
    *,
    document_text: str,
    citation: ExtractedCitation,
) -> CaseNameSearchPreparation:
    """Prepare party anchors for a not-found citation's candidate search."""
    if not isinstance(citation.citation, FullCaseCitation):
        return CaseNameSearchPreparation(status=CaseNamePreparationStatus.EMPTY)

    window = DocumentTextWindow.around(
        document_text,
        citation.span,
        before_chars=PREPARATION_CONTEXT_BEFORE_CHARS,
        after_chars=0,
    )
    extracted_plaintiff = citation.citation.plaintiff or ""
    extracted_defendant = citation.citation.defendant or ""
    try:
        proposal, _final_ctx = await _prepare_case_name(
            ChatContext(),
            session.backend,
            local_context=window.text,
            locator=citation.matched_text,
            extracted_plaintiff=extracted_plaintiff,
            extracted_defendant=extracted_defendant,
            output_format_hint=_output_format_hint(),
            requirements=[
                check(
                    "classification must be consistent with party availability",
                    validation_fn=_validate_classification_consistency,
                ),
                req(
                    "plaintiff and defendant must be copied from local_context before the locator",
                    validation_fn=lambda ctx: _validate_grounded_before_locator(ctx, window),
                ),
            ],
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=structured_model_options(max_tokens=PREPARATION_MAX_TOKENS),
        )
    except Exception as exc:
        return CaseNameSearchPreparation(
            status=CaseNamePreparationStatus.FAILED,
            original_case_name=_original_case_name(citation.citation),
            plaintiff=extracted_plaintiff or None,
            defendant=extracted_defendant or None,
            prepared_case_name=None,
            court=citation.citation.court,
            locator=citation.matched_text,
            source="llm",
            error_message=str(exc),
        )

    status = _status_from_classification(proposal.classification)
    return CaseNameSearchPreparation(
        status=status,
        original_case_name=_original_case_name(citation.citation),
        plaintiff=proposal.plaintiff,
        defendant=proposal.defendant,
        prepared_case_name=_prepared_case_name(proposal.plaintiff, proposal.defendant),
        court=citation.citation.court,
        locator=citation.matched_text,
        source="llm",
        llm_classification=proposal.classification,
        llm_reason=proposal.reason,
    )


def _output_format_hint() -> str:
    fmt = llm_api_config_from_env(os.environ).response_format
    return JSON_OBJECT_HINT if fmt is LlmResponseFormat.JSON_OBJECT else "none"


def _validate_classification_consistency(ctx: Context) -> ValidationResult:
    proposal: _PreparedCaseName = ctx.last_output().parsed_repr
    party_count = sum(bool(value) for value in (proposal.plaintiff, proposal.defendant))
    if proposal.classification == "complete_case_name" and party_count == COMPLETE_PARTY_COUNT:
        return ValidationResult(result=True)
    if proposal.classification == "partial_case_name" and party_count == 1:
        return ValidationResult(result=True)
    if proposal.classification == "no_case_name" and party_count == 0:
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason="classification did not match plaintiff/defendant availability",
    )


def _validate_grounded_before_locator(
    ctx: Context,
    window: DocumentTextWindow,
) -> ValidationResult:
    proposal: _PreparedCaseName = ctx.last_output().parsed_repr
    for label, value in (("plaintiff", proposal.plaintiff), ("defendant", proposal.defendant)):
        if not value:
            continue
        grounded = window.locate(value)
        if grounded is None:
            return ValidationResult(
                result=False,
                reason=f"{label}={value!r} was not copied from local_context",
            )
        if grounded.span.end > window.anchor_span.start:
            return ValidationResult(
                result=False,
                reason=f"{label}={value!r} did not appear before the locator",
            )
    return ValidationResult(result=True)


def _status_from_classification(classification: str) -> CaseNamePreparationStatus:
    if classification == "complete_case_name":
        return CaseNamePreparationStatus.ACCEPTED
    if classification in {"partial_case_name", "no_case_name"}:
        return CaseNamePreparationStatus.EMPTY
    return CaseNamePreparationStatus.FAILED


def _original_case_name(citation: FullCaseCitation) -> str | None:
    return _prepared_case_name(citation.plaintiff, citation.defendant)


def _prepared_case_name(plaintiff: str | None, defendant: str | None) -> str | None:
    if plaintiff and defendant:
        return f"{plaintiff} v. {defendant}"
    return None
