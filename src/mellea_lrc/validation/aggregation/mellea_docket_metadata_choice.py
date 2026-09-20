"""One grounded model choice among a docket-metadata shortlist.

The deterministic shortlist only limits a broad CourtListener docket search. It
never claims that two written docket forms are equivalent. This operation makes
that semantic decision once, over the compact candidate list, and retains the
model's choice as ordinary candidate-selection evidence.
"""

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
    CitationValidation,
    DocketMetadataShortlistNode,
    LocatorCitationSummaryNode,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context


MAX_TOKENS = 512
MAX_REPAIR_TURNS = 2

PREFIX = """
Decide whether one returned docket record represents the filing's stated docket
citation. The candidates have already passed a deterministic court filter when
the filing stated a court, and a deliberately permissive string-similarity
filter. Those filters only make a broad metadata search reviewable; they do
not prove identity.

Compare the stated case name and docket number with every candidate's case name
and docket number. Court systems may display the same docket with ordinary
formatting differences, omitted context, padding, separators, category labels,
or assigned-official suffixes. Do not construct a normalized number, change any
value, or infer an unstated candidate. Choose one candidate only when the case
name and docket form together support the same matter. Reject all when none is
supportable.

Return a concise prose reason for the decision.
""".strip()

INSTRUCTION = """
stated_case_name:
{{stated_case_name}}

stated_docket_number:
{{stated_docket_number}}

candidates_json:
{{candidates_json}}
""".strip()


class _DocketMetadataChoiceProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["select_candidate", "reject_all"]
    candidate_index: int | None
    rationale: Annotated[str, Field(min_length=1)]


async def run_mellea_docket_metadata_choice(
    validation: CitationValidation,
    *,
    summary: LocatorCitationSummaryNode,
    shortlist: DocketMetadataShortlistNode,
    session: MelleaSession | None = None,
) -> MelleaLocatorCandidateChoiceNode:
    """Select or reject a candidate from one persisted metadata shortlist."""
    candidate_indices = tuple(candidate.candidate_index for candidate in shortlist.candidates)
    if not candidate_indices:
        msg = "Docket metadata choice requires at least one shortlisted candidate"
        raise ValueError(msg)
    by_index = {candidate.candidate_index: candidate for candidate in summary.candidates}
    if not set(candidate_indices) <= set(by_index):
        msg = "Every docket-metadata shortlist candidate must be present in the citation summary"
        raise ValueError(msg)

    candidates_json = json.dumps(
        [
            {
                "index": index,
                "case_name": by_index[index].retrieved_case_name,
                "docket_number": by_index[index].docket_number,
            }
            for index in candidate_indices
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    citation = validation.citation.stated
    case_name = citation.case_name.text if citation.case_name is not None else "(not stated)"
    docket_number = getattr(citation, "docket_number", None) or "(not stated)"
    try:
        result = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            InstructIvrSpec(
                description=INSTRUCTION,
                prefix=PREFIX,
                user_variables={
                    "stated_case_name": case_name,
                    "stated_docket_number": docket_number,
                    "candidates_json": candidates_json,
                },
                output_format=_DocketMetadataChoiceProposal,
                requirements=[
                    check(
                        "decision must select one shortlisted candidate or reject all",
                        validation_fn=lambda ctx: _validate_choice(ctx, candidate_indices),
                    ),
                ],
            ),
            strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
    except Exception as exc:
        return _node(
            summary,
            candidate_indices,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaLocatorCandidateChoiceOutcome.FAILED,
            status_message="Model docket-metadata choice failed during execution.",
            outcome_message="No model choice was admitted from the docket-metadata shortlist.",
            error=f"{type(exc).__name__}: {exc}",
        )
    if not result.success:
        return _node(
            summary,
            candidate_indices,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaLocatorCandidateChoiceOutcome.FAILED,
            status_message="Model docket-metadata choice exhausted its repair attempts.",
            outcome_message="No model choice was admitted from the docket-metadata shortlist.",
            error=result.failure_reason or "Model docket-metadata choice exhausted its repair budget",
            run=result,
        )
    try:
        proposal = _DocketMetadataChoiceProposal.model_validate_json(result.output)
    except ValidationError as exc:
        return _node(
            summary,
            candidate_indices,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaLocatorCandidateChoiceOutcome.FAILED,
            status_message="Model docket-metadata choice returned invalid structured output.",
            outcome_message="No model choice was admitted from the docket-metadata shortlist.",
            error=str(exc),
            run=result,
        )
    if proposal.decision == "select_candidate":
        return _node(
            summary,
            candidate_indices,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            selected_candidate_index=proposal.candidate_index,
            rationale=proposal.rationale,
            status_message="Model docket-metadata choice completed.",
            outcome_message=f"Model selected docket-metadata candidate {proposal.candidate_index}.",
            run=result,
        )
    return _node(
        summary,
        candidate_indices,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaLocatorCandidateChoiceOutcome.NO_MATCH,
        rationale=proposal.rationale,
        status_message="Model docket-metadata choice completed.",
        outcome_message="Model rejected every docket-metadata candidate.",
        run=result,
    )


def _validate_choice(ctx: Context, candidate_indices: tuple[int, ...]) -> ValidationResult:
    proposal = _DocketMetadataChoiceProposal.model_validate_json(ctx.last_output().value)
    if proposal.decision == "select_candidate":
        valid = proposal.candidate_index in candidate_indices
        reason = None if valid else f"candidate_index must be one of {candidate_indices}"
    else:
        valid = proposal.candidate_index is None
        reason = None if valid else "reject_all requires candidate_index to be null"
    return ValidationResult(result=valid, reason=reason)


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
        node_id=f"{summary.node_id}:mellea_docket_metadata_choice",
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
