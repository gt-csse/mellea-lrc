"""LLM-backed preparation for not-found case-name candidate search."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from mellea.core import ValidationResult
from mellea.core.requirement import Requirement
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.assessment.model_options import structured_model_options
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.llm import (
    InstructIvrSpec,
    RenderedChatMessage,
    render_instruct_chat_messages,
    render_instruct_prompt,
    run_instruct_ivr,
)
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
JSON_OUTPUT_REQUIREMENT = (
    'Return exactly one JSON object with shape '
    '{"classification":"...","plaintiff":"... or null","defendant":"... or null","reason":"..."}.'
)
CASE_NAME_PREPARATION_INSTRUCTION = """
Extract party anchors for the citation marked by locator.

Use only parties bound to this locator. The relevant copied case name usually
appears before the locator. If local_context contains multiple citations, do not
borrow parties from another citation.

Treat the locator string as the boundary marker inside local_context. Prefer the
nearest copied "plaintiff v. defendant" name before that locator over parser
hints. Parser hints may come from a different citation in the same window.

Use classification "complete_case_name" when both parties are present,
"partial_case_name" when exactly one party is present, and "no_case_name" when
no bound party is present.

locator:
{{locator}}

Parser hints, which may be missing or wrong:
plaintiff={{extracted_plaintiff}}
defendant={{extracted_defendant}}
""".strip()


class _PreparedCaseName(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: str
    plaintiff: str | None = None
    defendant: str | None = None
    reason: str | None = None


async def _prepare_case_name(
    session: MelleaSession,
    *,
    local_context: str,
    locator: str,
    extracted_plaintiff: str,
    extracted_defendant: str,
    requirements: list[Requirement],
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> tuple[_PreparedCaseName, object]:
    """Prepare parties for case-name search through Mellea instruct/validate/repair."""
    spec = _case_name_preparation_spec(
        local_context=local_context,
        locator=locator,
        extracted_plaintiff=extracted_plaintiff,
        extracted_defendant=extracted_defendant,
        requirements=requirements,
    )
    result = await run_instruct_ivr(session, spec, strategy=strategy, model_options=model_options)
    proposal = _proposal_from_output(result.result.value)
    return proposal, result.result_ctx


def _case_name_preparation_spec(
    *,
    local_context: str,
    locator: str,
    extracted_plaintiff: str,
    extracted_defendant: str,
    requirements: list[Requirement],
) -> InstructIvrSpec:
    return InstructIvrSpec(
        description=CASE_NAME_PREPARATION_INSTRUCTION,
        grounding_context={"local_context": local_context},
        user_variables={
            "locator": locator,
            "extracted_plaintiff": extracted_plaintiff or "<EMPTY>",
            "extracted_defendant": extracted_defendant or "<EMPTY>",
        },
        requirements=requirements,
    )


def render_case_name_preparation_prompt(
    *,
    local_context: str,
    locator: str,
    extracted_plaintiff: str = "",
    extracted_defendant: str = "",
    window: DocumentTextWindow,
) -> str:
    """Render the raw prompt for a case-name preparation instruction."""
    spec = _case_name_preparation_spec(
        local_context=local_context,
        locator=locator,
        extracted_plaintiff=extracted_plaintiff,
        extracted_defendant=extracted_defendant,
        requirements=_case_name_preparation_requirements(window),
    )
    return render_instruct_prompt(spec)


def render_case_name_preparation_chat_messages(
    *,
    local_context: str,
    locator: str,
    extracted_plaintiff: str = "",
    extracted_defendant: str = "",
    window: DocumentTextWindow,
) -> tuple[RenderedChatMessage, ...]:
    """Render the raw chat messages for a case-name preparation instruction."""
    spec = _case_name_preparation_spec(
        local_context=local_context,
        locator=locator,
        extracted_plaintiff=extracted_plaintiff,
        extracted_defendant=extracted_defendant,
        requirements=_case_name_preparation_requirements(window),
    )
    return render_instruct_chat_messages(spec)


def _case_name_preparation_requirements(window: DocumentTextWindow) -> list[Requirement]:
    return [
        req(JSON_OUTPUT_REQUIREMENT, validation_fn=_validate_output_schema),
        check(
            "classification must be consistent with party availability",
            validation_fn=_validate_classification_consistency,
        ),
        req(
            "plaintiff and defendant must be copied from local_context before the locator; "
            "if a visible 'plaintiff v. defendant' case-name marker appears before the locator, "
            "return complete_case_name for that citation; do not use parties separated from the "
            "target locator by another reporter citation",
            validation_fn=lambda ctx: _validate_grounded_before_locator(ctx, window),
        ),
    ]


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
        requirements = _case_name_preparation_requirements(window)
        proposal, _final_ctx = await _prepare_case_name(
            session,
            local_context=window.text,
            locator=citation.matched_text,
            extracted_plaintiff=extracted_plaintiff,
            extracted_defendant=extracted_defendant,
            requirements=requirements,
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


def _proposal_from_output(output: str | object) -> _PreparedCaseName:
    if not isinstance(output, str):
        msg = f"LLM output was not text: {type(output).__name__}"
        raise ValueError(msg)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        msg = f"LLM output was not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = "LLM output JSON was not an object"
        raise ValueError(msg)
    try:
        return _PreparedCaseName.model_validate(payload)
    except ValidationError as exc:
        msg = f"LLM output did not match case-name preparation schema: {exc}"
        raise ValueError(msg) from exc


def _proposal_from_context(ctx: Context) -> _PreparedCaseName:
    return _proposal_from_output(ctx.last_output().value)


def _validate_output_schema(ctx: Context) -> ValidationResult:
    try:
        _proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _validate_classification_consistency(ctx: Context) -> ValidationResult:
    try:
        proposal = _proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
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
    try:
        proposal = _proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    if (
        proposal.classification != "complete_case_name"
        and _has_case_name_marker_before_locator(window)
    ):
        return ValidationResult(
            result=False,
            reason="a copied v. case-name marker appears before the locator",
        )
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
        between = _window_text_between(window, grounded.span.end, window.anchor_span.start)
        if _contains_reporter_like_locator(between):
            return ValidationResult(
                result=False,
                reason=f"{label}={value!r} appears before another locator, not the target locator",
            )
    return ValidationResult(result=True)


def _has_case_name_marker_before_locator(window: DocumentTextWindow) -> bool:
    before_locator = _window_text_between(window, window.span.start, window.anchor_span.start)
    return bool(re.search(r"\b\S+\s+v\.?\s+\S+", before_locator))


def _contains_reporter_like_locator(text: str) -> bool:
    return bool(re.search(r"\b\d+\s+[A-Z][A-Za-z.0-9]*\s+\d+\b", text))


def _window_text_between(window: DocumentTextWindow, start: int, end: int) -> str:
    local_start = max(0, start - window.span.start)
    local_end = max(local_start, end - window.span.start)
    return window.text[local_start:local_end]


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
