"""Grounded model choice among a bounded set of reporter-locator candidates."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Annotated, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import check
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mellea_lrc.llm import (
    InstructIvrSpec,
    IvrRun,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.types import (
    CitationSummaryCandidate,
    CitationValidation,
    LocatorCitationSummaryNode,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

CONTEXT_BEFORE_CHARS = 480
CONTEXT_AFTER_CHARS = 240
CHOICE_MAX_TOKENS = 512
CHOICE_MAX_REPAIR_TURNS = 2

# TODO: Opinion reading could refine a tie, but it belongs to the later opinion
# stage and must not be coupled to this root-identity decision yet.
CHOICE_PREFIX = """
The filing contains one target reporter citation marked by locator. Read only
local_context and the complete list of retrieved candidates. Reparse the
filing's stated case name, court, and date from local_context. Then select the
single candidate that best represents that citation, or return no_match when
none is supportable from the stated fields.

Every candidate shown is a retrieved possibility. Consider every one, including
candidates whose preliminary field assessment says mismatch or partial_match:
those assessments are evidence, not a final selection. Do not use outside
knowledge, change the reporter locator, invent a field, or select multiple
candidates. When a field is absent in local_context, return null for it.
""".strip()

CHOICE_INSTRUCTION = """

locator:
{{locator}}

reviewed_candidates_json:
{{reviewed_candidates_json}}
""".strip()


class _CandidateChoiceProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["select_candidate", "no_match"]
    candidate_index: int | None
    reparsed_case_name: str | None
    reparsed_court: str | None
    reparsed_date: str | None
    rationale: Annotated[str, Field(min_length=1)]


async def run_mellea_locator_candidate_choice(
    validation: CitationValidation,
    *,
    summary: LocatorCitationSummaryNode,
    document_text: str,
    session: MelleaSession | None = None,
) -> MelleaLocatorCandidateChoiceNode:
    """Reparse local fields and select among every bounded, reviewed candidate."""
    candidate_indices = tuple(candidate.candidate_index for candidate in summary.candidates)
    if not candidate_indices:
        msg = "Model candidate choice requires a nonempty locator candidate summary"
        raise ValueError(msg)

    locator = validation.citation.matched_text
    span = validation.citation.locator_span
    start = max(0, span.start - CONTEXT_BEFORE_CHARS)
    end = min(len(document_text), span.end + CONTEXT_AFTER_CHARS)
    local_context = document_text[start:end]
    candidates_json = json.dumps(
        [_candidate_payload(candidate) for candidate in summary.candidates],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        resolved_session = session or start_mellea_session_from_env()
        options = llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=CHOICE_MAX_TOKENS)
        spec = InstructIvrSpec(
            description=CHOICE_INSTRUCTION,
            prefix=CHOICE_PREFIX,
            grounding_context={"local_context": local_context},
            user_variables={
                "locator": locator,
                "reviewed_candidates_json": candidates_json,
            },
            output_format=_CandidateChoiceProposal,
            requirements=[
                check(
                    "decision must select one reviewed candidate or no match",
                    validation_fn=lambda ctx: _validate_choice(ctx, candidate_indices),
                ),
                check(
                    "reparsed fields must be copied from local_context",
                    validation_fn=lambda ctx: _validate_reparsed_fields(ctx, local_context),
                ),
            ],
        )
        result = await run_instruct_ivr(
            resolved_session,
            spec,
            strategy=MultiTurnStrategy(loop_budget=CHOICE_MAX_REPAIR_TURNS),
            model_options=options,
        )
        if not result.success:
            return _node(
                summary,
                candidate_indices,
                status=ValidationNodeStatus.FAILED,
                outcome=MelleaLocatorCandidateChoiceOutcome.FAILED,
                status_message="Model candidate choice exhausted its repair attempts.",
                outcome_message="No model identity choice was admitted from the reviewed candidates.",
                error=result.failure_reason or "Model candidate choice exhausted its repair budget",
                run=result,
            )
        proposal = _proposal(result.output)
    except Exception as exc:
        return _node(
            summary,
            candidate_indices,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaLocatorCandidateChoiceOutcome.FAILED,
            status_message="Model candidate choice failed during execution.",
            outcome_message="No model identity choice was admitted from the reviewed candidates.",
            error=f"{type(exc).__name__}: {exc}",
        )

    if proposal.decision == "select_candidate":
        return _node(
            summary,
            candidate_indices,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            selected_candidate_index=proposal.candidate_index,
            reparsed_case_name=proposal.reparsed_case_name,
            reparsed_court=proposal.reparsed_court,
            reparsed_date=proposal.reparsed_date,
            rationale=proposal.rationale,
            status_message="Model candidate choice completed.",
            outcome_message=f"Model selected reviewed candidate {proposal.candidate_index}.",
            run=result,
        )
    return _node(
        summary,
        candidate_indices,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaLocatorCandidateChoiceOutcome.NO_MATCH,
        reparsed_case_name=proposal.reparsed_case_name,
        reparsed_court=proposal.reparsed_court,
        reparsed_date=proposal.reparsed_date,
        rationale=proposal.rationale,
        status_message="Model candidate choice completed.",
        outcome_message="Model found no supportable representative among the reviewed candidates.",
        run=result,
    )


def _candidate_payload(candidate: CitationSummaryCandidate) -> dict[str, object]:
    """Expose all retained candidate evidence in a compact model-readable form."""
    return {
        "candidate_index": candidate.candidate_index,
        "case_name": candidate.retrieved_case_name,
        "date": candidate.retrieved_year,
        "court_id": candidate.retrieved_court_id,
        "docket_id": candidate.docket_id,
        "preliminary_assessment": candidate.outcome.value,
        "field_assessments": {
            "case_name": candidate.case_name_outcome.value,
            "year": candidate.year_outcome.value,
            "court": candidate.court_outcome.value,
        },
    }


def _node(
    summary: LocatorCitationSummaryNode,
    candidate_indices: tuple[int, ...],
    *,
    status: ValidationNodeStatus,
    outcome: MelleaLocatorCandidateChoiceOutcome,
    selected_candidate_index: int | None = None,
    reparsed_case_name: str | None = None,
    reparsed_court: str | None = None,
    reparsed_date: str | None = None,
    rationale: str | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
    run: IvrRun | None = None,
) -> MelleaLocatorCandidateChoiceNode:
    return MelleaLocatorCandidateChoiceNode(
        node_id=f"{summary.node_id}:mellea_candidate_choice",
        status=status,
        outcome=outcome,
        candidate_indices=candidate_indices,
        selected_candidate_index=selected_candidate_index,
        reparsed_case_name=reparsed_case_name,
        reparsed_court=reparsed_court,
        reparsed_date=reparsed_date,
        rationale=rationale,
        depends_on=(summary.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
        run=run,
    )


def _proposal(output: object) -> _CandidateChoiceProposal:
    try:
        return _CandidateChoiceProposal.model_validate_json(output)
    except ValidationError as exc:
        msg = f"Invalid locator candidate-choice output: {exc}"
        raise ValueError(msg) from exc


def _validate_choice(ctx: Context, candidate_indices: tuple[int, ...]) -> ValidationResult:
    proposal = _proposal(ctx.last_output().value)
    if proposal.decision == "select_candidate":
        valid = proposal.candidate_index in candidate_indices
        reason = None if valid else f"candidate_index must be one of {candidate_indices}"
    else:
        valid = proposal.candidate_index is None
        reason = None if valid else "no_match requires candidate_index to be null"
    return ValidationResult(result=valid, reason=reason)


def _validate_reparsed_fields(ctx: Context, local_context: str) -> ValidationResult:
    proposal = _proposal(ctx.last_output().value)
    missing = [
        label
        for label, value in (
            ("reparsed_case_name", proposal.reparsed_case_name),
            ("reparsed_court", proposal.reparsed_court),
            ("reparsed_date", proposal.reparsed_date),
        )
        if value is not None and not _is_grounded(value, local_context)
    ]
    return ValidationResult(
        result=not missing,
        reason=None if not missing else f"not copied from local_context: {', '.join(missing)}",
    )


def _is_grounded(value: str, context: str) -> bool:
    """Ground copied field text with whitespace relaxation only."""
    pattern = r"\s+".join(re.escape(piece) for piece in value.split())
    return bool(pattern) and re.search(pattern, context) is not None
