"""Semantic comparison of different written forms of a docket number.

This is intentionally not a docket-number normalizer. Deterministic comparison
handles literal and whitespace equality. When those differ, a model considers
only whether the two supplied written forms can identify the same case, keeping
its verdict and prose reasoning as a first-class validation node.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Literal

from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from mellea_lrc.llm import (
    InstructIvrSpec,
    IvrRun,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.root_identity.context import masked_root_context
from mellea_lrc.validation.types import (
    MelleaDocketNumberEquivalenceNode,
    MelleaDocketNumberEquivalenceOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.model.document import Document
    from mellea_lrc.validation.types import CandidateEvaluationNode, CitationValidation, DocketNumberCheckNode


# This comparison can need a little prose reasoning about a jurisdiction's
# display convention. Give it a normal response budget rather than constraining
# the reason to a one-line classifier answer.
MAX_TOKENS = 512
MAX_REPAIR_TURNS = 2

# The research basis is deliberately broad: federal courts document that case
# displays can include filing location, year, category, serial, and assigned
# judge initials, while each court defines its own presentation. This prompt
# exposes that fact without encoding a jurisdiction-specific grammar.
PREFIX = """
Decide whether two written docket-number forms identify the same court case.

Court systems and legal citations can present one docket with different levels
of detail. A displayed form can retain or omit contextual portions such as a
filing location, category, padding, separators, or assigned-official suffixes.
Practices vary by court and era. Do not derive a canonical form, repair either
string, or apply a fixed docket grammar. Judge only the two forms supplied.

Return match only when their relationship is a plausible ordinary presentation
variation of one docket. A shared year, serial fragment, or general shape alone
does not establish equivalence. Return mismatch when the forms conflict or the
evidence is insufficient. Explain the decisive comparison in a short prose
reason.
""".strip()

INSTRUCTION = """
source_locator:
{{source_locator}}

stated_docket_number:
{{extracted_docket_number}}

retrieved_docket_number:
{{retrieved_docket_number}}

stated_court:
{{extracted_court_id}}

retrieved_court:
{{retrieved_court_id}}

The target-only local filing context is below. Other citation locators are
blanked. Use it only as context for the written locator; do not use it to
invent an unshown docket number.

local_context:
{{local_context}}
""".strip()


class _DocketNumberEquivalenceVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["match", "mismatch"]
    reason: Annotated[str, StringConstraints(min_length=1)]


async def run_mellea_docket_number_equivalence_check(
    validation: CitationValidation,
    *,
    deterministic_check: DocketNumberCheckNode,
    candidate: CandidateEvaluationNode,
    document: Document,
    session: MelleaSession | None = None,
) -> MelleaDocketNumberEquivalenceNode:
    """Judge one deterministic docket mismatch without changing either value."""
    extracted = deterministic_check.extracted_docket_number
    retrieved = deterministic_check.retrieved_docket_number
    if extracted is None or retrieved is None:
        return _node(
            deterministic_check=deterministic_check,
            status=ValidationNodeStatus.SKIPPED,
            outcome=MelleaDocketNumberEquivalenceOutcome.UNAVAILABLE,
            extracted=extracted,
            retrieved=retrieved,
            status_message="Skipped semantic docket comparison because a docket number is missing.",
            outcome_message="Semantic docket comparison is unavailable because required evidence is missing.",
        )

    record = validation.citation
    source_locator = document.text[record.locator_span.start : record.locator_span.end]
    context = masked_root_context(document, record, before=800, after=400)
    try:
        result = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            InstructIvrSpec(
                description=INSTRUCTION,
                prefix=PREFIX,
                grounding_context={"local_context": context.text},
                user_variables={
                    "source_locator": source_locator,
                    "extracted_docket_number": extracted,
                    "retrieved_docket_number": retrieved,
                    "extracted_court_id": _string_or_blank(_stated_court(validation)),
                    "retrieved_court_id": _string_or_blank(candidate.court_id),
                },
                output_format=_DocketNumberEquivalenceVerdict,
            ),
            strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
    except Exception as exc:
        return _node(
            deterministic_check=deterministic_check,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketNumberEquivalenceOutcome.FAILED,
            extracted=extracted,
            retrieved=retrieved,
            status_message="Model docket-number equivalence comparison failed during execution.",
            outcome_message="No semantic docket-equivalence judgment was admitted.",
            error=f"{type(exc).__name__}: {exc}",
        )

    if not result.success:
        return _node(
            deterministic_check=deterministic_check,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketNumberEquivalenceOutcome.FAILED,
            extracted=extracted,
            retrieved=retrieved,
            status_message="Model docket-number equivalence comparison exhausted its repair attempts.",
            outcome_message="No semantic docket-equivalence judgment was admitted.",
            error=result.failure_reason or "Model docket-equivalence comparison exhausted its repair budget",
            run=result,
        )
    try:
        verdict = _DocketNumberEquivalenceVerdict.model_validate_json(result.output)
    except ValidationError as exc:
        return _node(
            deterministic_check=deterministic_check,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketNumberEquivalenceOutcome.FAILED,
            extracted=extracted,
            retrieved=retrieved,
            status_message="Model docket-number equivalence comparison returned invalid structured output.",
            outcome_message="No semantic docket-equivalence judgment was admitted.",
            error=str(exc),
            run=result,
        )
    outcome = MelleaDocketNumberEquivalenceOutcome(verdict.verdict)
    return _node(
        deterministic_check=deterministic_check,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        extracted=extracted,
        retrieved=retrieved,
        reason=verdict.reason,
        status_message="Model docket-number equivalence comparison completed.",
        outcome_message=(
            "Mellea judged the stated and retrieved docket forms to identify the same case."
            if outcome is MelleaDocketNumberEquivalenceOutcome.MATCH
            else "Mellea judged the stated and retrieved docket forms not to identify the same case."
        ),
        run=result,
    )


def _node(
    *,
    deterministic_check: DocketNumberCheckNode,
    status: ValidationNodeStatus,
    outcome: MelleaDocketNumberEquivalenceOutcome,
    extracted: str | None,
    retrieved: str | None,
    reason: str | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
    run: IvrRun | None = None,
) -> MelleaDocketNumberEquivalenceNode:
    return MelleaDocketNumberEquivalenceNode(
        node_id=f"{deterministic_check.node_id}:mellea_docket_number_equivalence",
        status=status,
        outcome=outcome,
        extracted_docket_number=extracted,
        retrieved_docket_number=retrieved,
        reason=reason,
        depends_on=(deterministic_check.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
        run=run,
    )


def _stated_court(validation: CitationValidation) -> str | None:
    citation = validation.citation.fields
    court = getattr(citation, "court", None)
    return court if isinstance(court, str) else None


def _string_or_blank(value: str | None) -> str:
    return value or "(not stated)"
