"""Mellea semantic comparison for exact case-name mismatches."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    ExactCaseNameCheckNode,
    MelleaCaseNameCheckNode,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameReextractionNode,
    MelleaReextractedCaseNameCheckNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.validation.types import CitationValidation

MAX_TOKENS = 128
MAX_REPAIR_TURNS = 2
INSTRUCTION = """
Classify whether extracted_case_name is a normal legal citation of retrieved_case_name.

Consider only the names. A match means the same case despite ordinary legal
abbreviation or party shortening, with both sides of "v." represented. Missing,
garbled, materially incomplete, or different-case names are mismatches.

extracted_case_name:
{{extracted_case_name}}

retrieved_case_name:
{{retrieved_case_name}}
""".strip()


class _SemanticVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["match", "mismatch"]


async def run_mellea_case_name_check(
    validation: CitationValidation,
    *,
    case_name_evidence: ExactCaseNameCheckNode | MelleaCaseNameReextractionNode,
    candidate: CandidateEvaluationNode | None = None,
    session: MelleaSession | None = None,
) -> MelleaCaseNameCheckNode | MelleaReextractedCaseNameCheckNode:
    """Compare explicitly supplied case-name evidence against the retrieved name."""
    if isinstance(case_name_evidence, MelleaCaseNameReextractionNode):
        if candidate is None:
            msg = "Re-extracted case-name evidence requires its candidate"
            raise ValueError(msg)
        return await _run_reextracted_check(
            validation,
            case_name_evidence,
            candidate=candidate,
            session=session,
        )

    exact_node = case_name_evidence
    try:
        resolved_session = session or start_mellea_session_from_env()
        spec = InstructIvrSpec(
            description=INSTRUCTION,
            user_variables={
                "extracted_case_name": exact_node.extracted_case_name,
                "retrieved_case_name": exact_node.retrieved_case_name,
            },
            output_format=_SemanticVerdict,
            requirements=[req("Return a valid semantic-verdict object.", validation_fn=_valid_schema)],
        )
        result = await run_instruct_ivr(
            resolved_session,
            spec,
            strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
        if not result.success:
            return _failed_node(
                validation,
                exact_node,
                "Semantic case-name check exhausted its repair budget",
                status_message="Mellea semantic case-name comparison exhausted its repair attempts.",
            )
        verdict = _parse(result.result.value).verdict
    except Exception as exc:
        return _failed_node(
            validation,
            exact_node,
            f"{type(exc).__name__}: {exc}",
            status_message="Mellea semantic case-name comparison failed during execution.",
        )
    return MelleaCaseNameCheckNode(
        node_id=f"{exact_node.node_id}:mellea_case_name_check",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaCaseNameCheckOutcome(verdict),
        extracted_case_name=exact_node.extracted_case_name,
        retrieved_case_name=exact_node.retrieved_case_name,
        depends_on=(exact_node.node_id,),
        status_message="Mellea semantic case-name comparison completed.",
        outcome_message=(
            "Mellea judged the extracted and retrieved case names to identify the same case."
            if verdict == "match"
            else "Mellea judged the extracted and retrieved case names to identify different cases."
        ),
    )


def _parse(value: object) -> _SemanticVerdict:
    try:
        return _SemanticVerdict.model_validate_json(value)
    except ValidationError as exc:
        msg = f"Invalid semantic case-name output: {exc}"
        raise ValueError(msg) from exc


def _valid_schema(ctx: Context) -> ValidationResult:
    try:
        _parse(ctx.last_output().value)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _failed_node(
    validation: CitationValidation,
    exact_node: ExactCaseNameCheckNode,
    error: str,
    *,
    status_message: str,
) -> MelleaCaseNameCheckNode:
    return MelleaCaseNameCheckNode(
        node_id=f"{exact_node.node_id}:mellea_case_name_check",
        status=ValidationNodeStatus.FAILED,
        outcome=MelleaCaseNameCheckOutcome.FAILED,
        extracted_case_name=exact_node.extracted_case_name,
        retrieved_case_name=exact_node.retrieved_case_name,
        depends_on=(exact_node.node_id,),
        status_message=status_message,
        outcome_message="No semantic case-name verdict is available.",
        error=error,
    )


async def _run_reextracted_check(
    validation: CitationValidation,
    reextraction: MelleaCaseNameReextractionNode,
    *,
    candidate: CandidateEvaluationNode,
    session: MelleaSession | None,
) -> MelleaReextractedCaseNameCheckNode:
    extracted = _case_name_from_reextraction(reextraction)
    retrieved = candidate.case_name
    if extracted is None or retrieved is None:
        return MelleaReextractedCaseNameCheckNode(
            node_id=f"{reextraction.node_id}:mellea_reextracted_case_name_check",
            status=ValidationNodeStatus.SKIPPED,
            outcome=MelleaCaseNameCheckOutcome.UNAVAILABLE,
            reextracted_case_name=extracted,
            retrieved_case_name=retrieved,
            depends_on=(reextraction.node_id,),
            status_message="Skipped semantic comparison because re-extracted or retrieved evidence is missing.",
            outcome_message="Semantic case-name comparison is unavailable because one case name is missing.",
        )
    status, outcome, status_message, outcome_message, error = await _semantic_outcome(
        extracted,
        retrieved,
        session,
        debug_context={
            "citation_id": validation.citation_id,
            "node_id": f"{validation.citation_id}:mellea_reextracted_case_name_check",
            "operation": "mellea_reextracted_case_name_check",
        },
    )
    return MelleaReextractedCaseNameCheckNode(
        node_id=f"{reextraction.node_id}:mellea_reextracted_case_name_check",
        status=status,
        outcome=outcome,
        reextracted_case_name=extracted,
        retrieved_case_name=retrieved,
        depends_on=(reextraction.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
    )


def _case_name_from_reextraction(node: MelleaCaseNameReextractionNode) -> str | None:
    if node.plaintiff is None or node.defendant is None:
        return None
    return f"{node.plaintiff} v. {node.defendant}"


async def _semantic_outcome(
    extracted_case_name: str,
    retrieved_case_name: str,
    session: MelleaSession | None,
    *,
    debug_context: dict[str, str],
) -> tuple[
    ValidationNodeStatus,
    MelleaCaseNameCheckOutcome,
    str,
    str,
    str | None,
]:
    try:
        spec = InstructIvrSpec(
            description=INSTRUCTION,
            user_variables={
                "extracted_case_name": extracted_case_name,
                "retrieved_case_name": retrieved_case_name,
            },
            debug_context=debug_context,
            output_format=_SemanticVerdict,
            requirements=[req("Return a valid semantic-verdict object.", validation_fn=_valid_schema)],
        )
        result = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            spec,
            strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
        if result.success:
            verdict = _parse(result.result.value).verdict
            return (
                ValidationNodeStatus.SUCCEEDED,
                MelleaCaseNameCheckOutcome(verdict),
                "Mellea semantic case-name comparison completed.",
                (
                    "Mellea judged the re-extracted and retrieved case names to identify the same case."
                    if verdict == "match"
                    else "Mellea judged the re-extracted and retrieved case names to identify different cases."
                ),
                None,
            )
        return (
            ValidationNodeStatus.FAILED,
            MelleaCaseNameCheckOutcome.FAILED,
            "Mellea semantic case-name comparison exhausted its repair attempts.",
            "No semantic case-name verdict is available.",
            "Semantic case-name check exhausted its repair budget",
        )
    except Exception as exc:
        return (
            ValidationNodeStatus.FAILED,
            MelleaCaseNameCheckOutcome.FAILED,
            "Mellea semantic case-name comparison failed during execution.",
            "No semantic case-name verdict is available.",
            f"{type(exc).__name__}: {exc}",
        )
