"""One grounded model review of a docket number after docket retrieval misses.

This is a recovery stage for the identifier itself, not an identity decision.
The model receives only the target root's local context and may reproduce a
number that is physically present inside that root's locator.  A different
number is consequently a correction of the parser's stated field, never an
invented identifier.  The caller decides whether to re-run docket search.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Annotated

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.core.fuzziness import FuzzinessOption
from mellea_lrc.extraction.reading.dockets import DOCKET_PREFIX
from mellea_lrc.llm import (
    EvidenceCandidate,
    GroundingEvidence,
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.root_context import masked_root_context
from mellea_lrc.validation.types import (
    MelleaDocketNumberReviewNode,
    MelleaDocketNumberReviewOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.extraction.types import Document

MAX_TOKENS = 640
MAX_REPAIR_TURNS = 2
DOCKET_NUMBER_GROUNDING = FuzzinessOption.whitespace_relaxation()
_LEADING_DOCKET_LABEL = re.compile(DOCKET_PREFIX, re.IGNORECASE)

# This prefix contains the stable recovery contract only. The filing and its
# particular identifier are supplied separately, which keeps a provider prefix
# cache useful without making the prompt depend on a corpus or jurisdiction.
DOCKET_NUMBER_REVIEW_PREFIX = """
Read one docket locator from a legal filing after a CourtListener docket search
found no matching record. Decide only what docket number the filing actually
writes. A docket number is the court-assigned identifier for a case or
proceeding; it is not an ECF entry number, an exhibit number, a statute, or a
page number.

Return the docket-number portion exactly as it appears inside source_locator.
Do not normalize punctuation, add digits, infer a value from outside knowledge,
or use another citation. Do not include a leading label such as ``No.`` or
``Case No.`` in docket_number. Whitespace damage may be preserved exactly as
written.
If source_locator does not contain a docket number, return null. Give one short
reason.
""".strip()

INSTRUCTION = """
source_locator:
{{source_locator}}

The surrounding target-only filing context is below. Other citation locators
are blanked. Use it only to decide whether source_locator is a docket number;
your returned docket_number must still be copied from source_locator.

local_context:
{{local_context}}
""".strip()


class _DocketNumberProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docket_number: str | None
    reason: Annotated[str, StringConstraints(min_length=1)]


def _parse(value: object) -> _DocketNumberProposal:
    return _DocketNumberProposal.model_validate_json(str(value))


def _grounded_candidates(source_locator: str) -> GroundingEvidence[str]:
    """Expose every plausible physical substring as a reusable grounding set.

    The locator is short. Enumerating its digit-bearing, alphanumeric-bounded
    substrings keeps the grounding policy general: the model cannot supply any
    character that the filing did not write, while no jurisdiction-specific
    docket grammar is needed to discover a corrected parse.
    """
    candidates: list[EvidenceCandidate[str]] = []
    source_number = _source_number_region(source_locator)
    for start, first in enumerate(source_number):
        if not first.isalnum():
            continue
        for end in range(start + 1, len(source_number) + 1):
            value = source_number[start:end]
            if not value[-1].isalnum() or not any(character.isdigit() for character in value):
                continue
            candidates.append(EvidenceCandidate(text=value, value=value))
    return GroundingEvidence(candidates)


def _source_number_region(source_locator: str) -> str:
    """Exclude a recognized leading docket label from identifier evidence.

    A label helps a rule or a model recognize the citation shape, but it is not
    part of the court-assigned docket number. Reusing the first-pass reader's
    general label grammar avoids accepting ``No. 16-CV-8607`` as a distinct
    identifier while preserving all later, opaque identifier text for review.
    A locator without one of those labels remains fully available to hunting.
    """
    match = _LEADING_DOCKET_LABEL.match(source_locator)
    return source_locator[match.end() :] if match is not None else source_locator


def _grounded_number(source_locator: str, proposed: str | None) -> str | None:
    """Return one source spelling or refuse an ambiguous/non-source proposal."""
    if proposed is None:
        return None
    matches = _grounded_candidates(source_locator).fuzzy_match(proposed, DOCKET_NUMBER_GROUNDING)
    if len(matches) != 1:
        return None
    return matches[0].candidate.value


def _validate_number(ctx: Context, source_locator: str) -> ValidationResult:
    try:
        proposal = _parse(ctx.last_output().value)
    except ValidationError:
        # The shared IVR wrapper will provide the schema diagnostic first.
        return ValidationResult(result=True)
    if proposal.docket_number is None:
        return ValidationResult(result=True)
    grounded = _grounded_number(source_locator, proposal.docket_number)
    if grounded is not None:
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason=(
            "docket_number must be one unambiguous substring of source_locator, copied with at most "
            "whitespace variation."
        ),
    )


def _node(
    *,
    record: CitationRecord,
    source_locator: str,
    trigger_node_id: str,
    status: ValidationNodeStatus,
    outcome: MelleaDocketNumberReviewOutcome,
    proposed_docket_number: str | None = None,
    grounded_docket_number: str | None = None,
    reason: str | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
    run=None,
) -> MelleaDocketNumberReviewNode:
    citation = record.stated
    extracted = citation.docket_number if isinstance(citation, DocketCitation) else None
    return MelleaDocketNumberReviewNode(
        node_id=f"{record.citation_id}:mellea_docket_number_review",
        status=status,
        outcome=outcome,
        source_locator=source_locator,
        extracted_docket_number=extracted,
        proposed_docket_number=proposed_docket_number,
        grounded_docket_number=grounded_docket_number,
        reason=reason,
        depends_on=(trigger_node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
        run=run,
    )


async def run_mellea_docket_number_review(
    record: CitationRecord,
    *,
    document: Document,
    trigger_node_id: str,
    session: MelleaSession | None = None,
) -> MelleaDocketNumberReviewNode:
    """Re-read a failed docket root once and ground any replacement in source text."""
    source_locator = document.text[record.locator_span.start : record.locator_span.end]
    context = masked_root_context(document, record)
    try:
        result = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            InstructIvrSpec(
                description=INSTRUCTION,
                prefix=DOCKET_NUMBER_REVIEW_PREFIX,
                grounding_context={"local_context": context.text},
                user_variables={"source_locator": source_locator},
                output_format=_DocketNumberProposal,
                requirements=[
                    req(
                        "docket_number must be copied from source_locator",
                        validation_fn=lambda ctx: _validate_number(ctx, source_locator),
                    )
                ],
            ),
            strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
    except Exception as exc:
        return _node(
            record=record,
            source_locator=source_locator,
            trigger_node_id=trigger_node_id,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketNumberReviewOutcome.FAILED,
            status_message="Model docket-number review failed during execution.",
            outcome_message="No docket correction was admitted.",
            error=f"{type(exc).__name__}: {exc}",
        )

    if not result.success:
        return _node(
            record=record,
            source_locator=source_locator,
            trigger_node_id=trigger_node_id,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketNumberReviewOutcome.FAILED,
            status_message="Model docket-number review exhausted its repair attempts.",
            outcome_message="No docket correction was admitted.",
            error=result.failure_reason or "Model docket-number review exhausted its repair budget",
            run=result,
        )
    try:
        proposal = _parse(result.output)
    except ValidationError as exc:
        return _node(
            record=record,
            source_locator=source_locator,
            trigger_node_id=trigger_node_id,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketNumberReviewOutcome.FAILED,
            status_message="Model docket-number review returned invalid structured output.",
            outcome_message="No docket correction was admitted.",
            error=str(exc),
            run=result,
        )

    grounded = _grounded_number(source_locator, proposal.docket_number)
    citation = record.stated
    extracted = citation.docket_number if isinstance(citation, DocketCitation) else None
    if grounded is None:
        return _node(
            record=record,
            source_locator=source_locator,
            trigger_node_id=trigger_node_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketNumberReviewOutcome.NO_DOCKET_NUMBER,
            proposed_docket_number=proposal.docket_number,
            reason=proposal.reason,
            status_message="Model docket-number review completed.",
            outcome_message="The model found no supportable docket number inside the source locator.",
            run=result,
        )
    outcome = (
        MelleaDocketNumberReviewOutcome.UNCHANGED
        if grounded == extracted
        else MelleaDocketNumberReviewOutcome.CORRECTED
    )
    return _node(
        record=record,
        source_locator=source_locator,
        trigger_node_id=trigger_node_id,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        proposed_docket_number=proposal.docket_number,
        grounded_docket_number=grounded,
        reason=proposal.reason,
        status_message="Model docket-number review completed.",
        outcome_message=(
            "The model confirmed the stated docket number."
            if outcome is MelleaDocketNumberReviewOutcome.UNCHANGED
            else "The model recovered a different docket number from the source locator."
        ),
        run=result,
    )
