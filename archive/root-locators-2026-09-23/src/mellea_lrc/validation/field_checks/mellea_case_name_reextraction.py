"""Mellea case-name re-extraction from local citation context."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.llm import (
    InstructIvrSpec,
    IvrRun,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.validation.field_checks.source_case_name import ground_source_case_name
from mellea_lrc.validation.types import (
    CitationValidation,
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    MelleaCaseNameCheckNode,
    MelleaCaseNameReextractionNode,
    MelleaCaseNameReextractionOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

CONTEXT_BEFORE_CHARS = 320
CONTEXT_AFTER_CHARS = 160
REEXTRACTION_MAX_TOKENS = 384
REEXTRACTION_MAX_REPAIR_TURNS = 2

REEXTRACTION_PREFIX = """
Extract the full case name copied in local_context for the citation
marked by locator. Treat locator as the boundary marker. Prefer the nearest
copied name before locator, even when a docket number,
parallel citation, or other citation metadata occurs between the name and
locator. If local_context contains multiple citations, do not borrow parties
from another citation.

Copy the entire name as written into case_name_quote, and separately return its
plaintiff and defendant when present. A complete name can have only one party,
such as a proceeding named "In re" or "Ex parte". Minor spacing or punctuation
damage does not erase otherwise literal source text. Do not use outside
knowledge, expand legal abbreviations, or normalize a party toward a known
case. Return null for a field that the source does not state.

Return classification "complete_case_name" when the full written name is
present, including a complete one-party name; "partial_case_name" when only a
fragment is present; and "no_case_name" only when no name bound to locator
appears in local_context.
""".strip()

REEXTRACTION_INSTRUCTION = """
locator:
{{locator}}
""".strip()


class _PartyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["complete_case_name", "partial_case_name", "no_case_name"]
    case_name_quote: str | None
    plaintiff: str | None
    defendant: str | None


async def run_mellea_case_name_reextraction(
    validation: CitationValidation,
    *,
    trigger: ExactLocatorLookupNode | ExactCaseNameCheckNode | MelleaCaseNameCheckNode,
    locator_lookup: ExactLocatorLookupNode,
    document_text: str,
    session: MelleaSession | None = None,
) -> MelleaCaseNameReextractionNode:
    """Re-extract locally grounded parties from an explicit validation trigger."""
    if not locator_lookup.locator:
        return _node(
            validation,
            trigger,
            ValidationNodeStatus.SKIPPED,
            MelleaCaseNameReextractionOutcome.UNAVAILABLE,
            status_message="Skipped local case-name re-extraction because the locator is missing.",
            outcome_message="Local case-name re-extraction is unavailable because the locator is missing.",
        )

    span = validation.citation.locator_span
    start = max(0, span.start - CONTEXT_BEFORE_CHARS)
    end = min(len(document_text), span.end + CONTEXT_AFTER_CHARS)
    local_context = document_text[start:end]
    before_locator = document_text[start : span.start]
    try:
        resolved_session = session or start_mellea_session_from_env()
        options = llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=REEXTRACTION_MAX_TOKENS)
        spec = InstructIvrSpec(
            description=REEXTRACTION_INSTRUCTION,
            prefix=REEXTRACTION_PREFIX,
            grounding_context={"local_context": local_context},
            user_variables={"locator": locator_lookup.locator},
            output_format=_PartyProposal,
            requirements=[
                check(
                    "classification must match party availability",
                    validation_fn=_validate_classification,
                ),
                req(
                    "parties must be copied before locator in local_context",
                    validation_fn=lambda ctx: _validate_grounding(ctx, before_locator),
                ),
            ],
        )
        result = await run_instruct_ivr(
            resolved_session,
            spec,
            strategy=MultiTurnStrategy(loop_budget=REEXTRACTION_MAX_REPAIR_TURNS),
            model_options=options,
        )
        if not result.success:
            return _node(
                validation,
                trigger,
                ValidationNodeStatus.FAILED,
                MelleaCaseNameReextractionOutcome.FAILED,
                status_message="Local case-name re-extraction exhausted its repair attempts.",
                outcome_message="Local case-name re-extraction did not satisfy its grounding requirements.",
                error="Case-name re-extraction exhausted its repair budget",
                run=result,
            )
        proposal = _proposal(result.output)
    except Exception as exc:
        return _node(
            validation,
            trigger,
            ValidationNodeStatus.FAILED,
            MelleaCaseNameReextractionOutcome.FAILED,
            status_message="Local case-name re-extraction failed during execution.",
            outcome_message="Local case-name re-extraction could not be completed.",
            error=f"{type(exc).__name__}: {exc}",
        )

    if not _classification_is_consistent(proposal):
        return _node(
            validation,
            trigger,
            ValidationNodeStatus.FAILED,
            MelleaCaseNameReextractionOutcome.FAILED,
            status_message="Local case-name re-extraction returned inconsistent fields.",
            outcome_message="No source-grounded case name was admitted.",
            error="classification, case_name_quote, and party fields disagree",
            run=result,
        )
    grounded_name = ground_source_case_name(
        proposal.case_name_quote,
        before_locator=before_locator,
        source_start=start,
        plaintiff=proposal.plaintiff,
        defendant=proposal.defendant,
    )
    if proposal.case_name_quote is not None and grounded_name is None:
        return _node(
            validation,
            trigger,
            ValidationNodeStatus.FAILED,
            MelleaCaseNameReextractionOutcome.FAILED,
            status_message="Local case-name re-extraction quoted text outside the target source window.",
            outcome_message="No source-grounded case name was admitted.",
            error="case_name_quote was not found before the target locator",
            run=result,
        )
    if any(
        party is not None and not _is_grounded(party, before_locator)
        for party in (proposal.plaintiff, proposal.defendant)
    ):
        return _node(
            validation,
            trigger,
            ValidationNodeStatus.FAILED,
            MelleaCaseNameReextractionOutcome.FAILED,
            status_message="Local case-name re-extraction returned an ungrounded party.",
            outcome_message="No source-grounded case name was admitted.",
            error="plaintiff or defendant was not copied before the target locator",
            run=result,
        )
    outcome = {
        "complete_case_name": MelleaCaseNameReextractionOutcome.COMPLETE,
        "partial_case_name": MelleaCaseNameReextractionOutcome.PARTIAL,
        "no_case_name": MelleaCaseNameReextractionOutcome.NOT_FOUND,
    }[proposal.classification]
    return _node(
        validation,
        trigger,
        ValidationNodeStatus.SUCCEEDED,
        outcome,
        status_message="Local case-name re-extraction completed.",
        plaintiff=proposal.plaintiff,
        defendant=proposal.defendant,
        case_name=(grounded_name if outcome is MelleaCaseNameReextractionOutcome.COMPLETE else None),
        outcome_message={
            MelleaCaseNameReextractionOutcome.COMPLETE: "Re-extracted a complete case name from local context.",
            MelleaCaseNameReextractionOutcome.PARTIAL: "Re-extracted a case-name fragment from local context.",
            MelleaCaseNameReextractionOutcome.NOT_FOUND: "No case name was found before the locator.",
        }[outcome],
        run=result,
    )


def _node(
    validation: CitationValidation,
    trigger: ExactLocatorLookupNode | ExactCaseNameCheckNode | MelleaCaseNameCheckNode,
    status: ValidationNodeStatus,
    outcome: MelleaCaseNameReextractionOutcome,
    *,
    plaintiff: str | None = None,
    defendant: str | None = None,
    case_name: CaseName | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
    run: IvrRun | None = None,
) -> MelleaCaseNameReextractionNode:
    return MelleaCaseNameReextractionNode(
        node_id=f"{trigger.node_id}:mellea_case_name_reextraction",
        status=status,
        outcome=outcome,
        plaintiff=plaintiff,
        defendant=defendant,
        case_name=case_name,
        depends_on=(trigger.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
        run=run,
    )


def _proposal(output: object) -> _PartyProposal:
    try:
        return _PartyProposal.model_validate_json(output)
    except ValidationError as exc:
        msg = f"Invalid case-name re-extraction output: {exc}"
        raise ValueError(msg) from exc


def _validate_classification(ctx: Context) -> ValidationResult:
    proposal = _proposal(ctx.last_output().value)
    valid = _classification_is_consistent(proposal)
    return ValidationResult(
        result=valid,
        reason=None if valid else "classification, copied case_name_quote, and parties disagree",
    )


def _classification_is_consistent(proposal: _PartyProposal) -> bool:
    has_quote = bool(proposal.case_name_quote and proposal.case_name_quote.strip())
    has_party = proposal.plaintiff is not None or proposal.defendant is not None
    if proposal.classification == "no_case_name":
        return not has_quote and not has_party
    return has_quote


def _validate_grounding(ctx: Context, before_locator: str) -> ValidationResult:
    proposal = _proposal(ctx.last_output().value)
    missing = [
        label
        for label, value in (("plaintiff", proposal.plaintiff), ("defendant", proposal.defendant))
        if value is not None and not _is_grounded(value, before_locator)
    ]
    if (
        proposal.case_name_quote is not None
        and ground_source_case_name(
            proposal.case_name_quote,
            before_locator=before_locator,
            source_start=0,
        )
        is None
    ):
        missing.append("case_name_quote")
    return ValidationResult(
        result=not missing,
        reason=None if not missing else f"not copied before locator: {', '.join(missing)}",
    )


def _is_grounded(value: str, context: str) -> bool:
    """Return whether copied text occurs with exact tokens and flexible whitespace."""
    pattern = r"\s+".join(re.escape(piece) for piece in value.split())
    return re.search(pattern, context) is not None
