"""Grounded model choice among a bounded set of complete-locator candidates."""

from __future__ import annotations

import json
import os
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
    CandidateProvenance,
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
The filing contains one target complete citation marked by locator. Read only
local_context and the complete list of retrieved candidates. Select the single
candidate that best represents that citation, or return no_match when none is
supportable from the stated fields.

Every candidate shown is a retrieved possibility. Consider every one, including
candidates whose preliminary field assessment says mismatch or partial_match:
those assessments are evidence, not a final selection. Do not use outside
knowledge, change the stated locator, invent a field, or select multiple
candidates. Field re-extraction and correction are separate stages, already
completed or deliberately omitted by the caller; do not re-extract fields here.
If a candidate carries a reviewed local case name, use it as evidence of that
separate review; do not assume every candidate was re-extracted. A reviewed
local name confirms what the filing says, not whether the retrieved candidate
has that name. Compare the local and retrieved case names directly. If they
identify different cases, return no_match even when the reporter locator agrees.

A candidate marked selection_eligible is one the caller permits you to choose;
the flag alone does not prove identity or verify the locator, court, or date.
Evaluate whether the stated and retrieved case names plausibly identify the
same matter: ordinary abbreviation, an added or omitted party, or a legal-entity
suffix can be compatible; different parties without supporting evidence are
not. A misspelled party or title is a mismatch, even when the intended case is
recognizable; do not treat a spelling error as an abbreviation. Do not let a
name resemblance override a stated court or date contradiction.

For docket-derived candidates, docket numbers can have different written forms
for the same case. Compare the forms semantically rather than requiring their
strings to be identical.

For a docket-derived candidate, case_filed_year is the date the case began.
It is not the date of an order or opinion cited in local_context, so it cannot
directly verify that citation's decision date. It can only establish whether
the dates are chronologically compatible.
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
    rationale: Annotated[str, Field(min_length=1)]


async def run_mellea_locator_candidate_choice(
    validation: CitationValidation,
    *,
    summary: LocatorCitationSummaryNode,
    document_text: str,
    session: MelleaSession | None = None,
    eligible_candidate_indices: tuple[int, ...] | None = None,
) -> MelleaLocatorCandidateChoiceNode:
    """Select among an explicitly eligible candidate set.

    ``summary`` always carries every retrieved candidate. A caller may narrow
    *selection* to candidates that passed a non-semantic identity anchor, such
    as a docket-number comparison, while keeping non-eligible candidates fully
    visible as evidence.
    """
    candidate_indices = (
        tuple(candidate.candidate_index for candidate in summary.candidates)
        if eligible_candidate_indices is None
        else eligible_candidate_indices
    )
    if not candidate_indices:
        msg = "Model candidate choice requires at least one eligible candidate"
        raise ValueError(msg)
    known_indices = {candidate.candidate_index for candidate in summary.candidates}
    if not set(candidate_indices) <= known_indices:
        msg = "Eligible candidate indices must be present in the complete candidate summary"
        raise ValueError(msg)

    locator = validation.citation.matched_text
    span = validation.citation.locator_span
    start = max(0, span.start - CONTEXT_BEFORE_CHARS)
    end = min(len(document_text), span.end + CONTEXT_AFTER_CHARS)
    local_context = document_text[start:end]
    candidates_json = json.dumps(
        [
            _candidate_payload(candidate, selection_eligible=candidate.candidate_index in candidate_indices)
            for candidate in summary.candidates
        ],
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
                "selection_eligible_indices": json.dumps(candidate_indices),
            },
            output_format=_CandidateChoiceProposal,
            requirements=[
                check(
                    "decision must select one reviewed candidate or no match",
                    validation_fn=lambda ctx: _validate_choice(ctx, candidate_indices),
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
        rationale=proposal.rationale,
        status_message="Model candidate choice completed.",
        outcome_message="Model found no supportable representative among the reviewed candidates.",
        run=result,
    )


def _candidate_payload(
    candidate: CitationSummaryCandidate,
    *,
    selection_eligible: bool,
) -> dict[str, object]:
    """Expose all retained candidate evidence in a compact model-readable form."""
    payload = {
        "candidate_index": candidate.candidate_index,
        "selection_eligible": selection_eligible,
        "case_name": candidate.retrieved_case_name,
        "local_case_name": candidate.extracted_case_name,
        "local_case_name_evidence": candidate.case_name_evidence,
        "court_id": candidate.retrieved_court_id,
        "docket_id": candidate.docket_id,
        "docket_number": candidate.docket_number,
        "preliminary_assessment": candidate.outcome.value,
        "field_assessments": {
            "case_name": candidate.case_name_outcome.value,
            "year": candidate.year_outcome.value if candidate.year_outcome is not None else None,
            "court": candidate.court_outcome.value,
        },
    }
    if candidate.provenance is CandidateProvenance.DOCKET:
        payload["case_filed_year"] = candidate.retrieved_year
    else:
        payload["decision_year"] = candidate.retrieved_year
    return payload


def _node(
    summary: LocatorCitationSummaryNode,
    candidate_indices: tuple[int, ...],
    *,
    status: ValidationNodeStatus,
    outcome: MelleaLocatorCandidateChoiceOutcome,
    selected_candidate_index: int | None = None,
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
