"""Confirm a suspected docket locator without deciding any citation fields.

The reviewer has one job: decide whether a labelled opaque span is a docket
locator cited for a court case. It quotes the complete locator verbatim, and the
exact string check prevents a tidied or invented span from becoming a record.
Court, date, case name, and pin cite are deliberately absent: after admission,
the ordinary field readers read them from the document just as they do for a
deterministically read root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from mellea_lrc.extraction.adjudication.types import SiteReview
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.extraction.adjudication.candidates.docket_sites import SuspectedDocket
    from mellea_lrc.llm import IvrRun

# GLM 5.3 uses part of the completion budget for reasoning before returning its
# strict JSON response. A smaller budget can end at the reasoning trace with no
# JSON content at all, so this needs room for both.
MAX_TOKENS = 1200
MAX_REPAIR_TURNS = 2

# This invariant review contract is a system-message prefix.  Every site review
# shares it verbatim, allowing the provider's prefix cache to reuse it.  It
# intentionally supplies only the ordinary meaning of a docket number; local
# court conventions belong in a future jurisdictional handbook, not here.
DOCKET_REVIEW_PREFIX = """
Decide whether a labelled opaque span in a legal filing is a docket locator
cited for a court case. A docket number is the court-assigned identifier for
one case or proceeding.

Rules:
- Quote the complete candidate locator exactly as written in the window,
  including its docket label, character for character. Do not repair it.
- Quote the docket number portion exactly as written, without the label. Do
  not repair it.
- Set is_docket_citation=false if the string is not citing a case: an exhibit
  or docket-entry number, a statute, a filing reference, or an internal
  cross-reference.
- Give one short reason for the decision, especially when
  is_docket_citation=false.
""".strip()

# Only the candidate and its bounded document context vary between site
# reviews.  They remain in the instruction message, after the cached prefix.
INSTRUCTION = """
A docket-number-shaped string was found in this filing window: {{locator}}

Return the required structured decision for this candidate.

window:
{{window}}
""".strip()

# OpenRouter uses session IDs to retain provider routing for a repeated prompt
# prefix.  This identifier carries no filing content or decision state.
DOCKET_SITE_HUNTING_SESSION_ID = "mellea-lrc-docket-site-hunting-v1"


class _DocketProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_docket_citation: bool
    locator: str | None
    docket_number: str | None
    reason: Annotated[str, StringConstraints(min_length=1)]


@dataclass(frozen=True, slots=True)
class RecoveredDocketLocator:
    """A cited docket locator the reviewer grounded against one document span."""

    locator_text: str
    docket_number: str
    reason: str


def _parse(value: object) -> _DocketProposal:
    return _DocketProposal.model_validate_json(str(value))


def _validate_schema(ctx: Context) -> ValidationResult:
    try:
        _parse(ctx.last_output().value)
    except ValidationError as error:
        return ValidationResult(result=False, reason=str(error))
    return ValidationResult(result=True)


def _validate_locator(ctx: Context, site: SuspectedDocket) -> ValidationResult:
    """A confirmed citation must quote the complete locator found in the window."""
    try:
        proposal = _parse(ctx.last_output().value)
    except ValidationError as error:
        return ValidationResult(result=False, reason=str(error))
    if not proposal.is_docket_citation:
        return ValidationResult(result=True)
    if proposal.locator != site.locator_text:
        return ValidationResult(
            result=False,
            reason=(
                f"{proposal.locator!r} is not the candidate {site.locator_text!r}. Quote the "
                "candidate exactly as it appears, character for character."
            ),
        )
    if proposal.docket_number != site.docket_number:
        return ValidationResult(
            result=False,
            reason=(
                f"{proposal.docket_number!r} is not the docket portion {site.docket_number!r}. "
                "Quote the candidate's docket number exactly as it appears."
            ),
        )
    return ValidationResult(result=True)


async def adjudicate_docket(
    site: SuspectedDocket,
    *,
    session: MelleaSession | None = None,
) -> SiteReview[RecoveredDocketLocator]:
    """Review one docket site and retain the evidence whether it is accepted or not."""
    resolved_session = session or start_mellea_session_from_env()
    result = await run_instruct_ivr(
        resolved_session,
        InstructIvrSpec(
            description=INSTRUCTION,
            prefix=DOCKET_REVIEW_PREFIX,
            user_variables={"locator": site.locator_text, "window": site.context},
            output_format=_DocketProposal,
            requirements=[
                req("Return a valid docket proposal.", validation_fn=_validate_schema),
                req(
                    "Quote the complete docket locator exactly as written in the window.",
                    validation_fn=lambda ctx: _validate_locator(ctx, site),
                ),
            ],
        ),
        strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
        model_options={
            **llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
            "extra_body": {"session_id": DOCKET_SITE_HUNTING_SESSION_ID},
        },
    )
    return _site_review(result)


def _site_review(result: IvrRun) -> SiteReview[RecoveredDocketLocator]:
    """Keep a failed exact-grounding run as a declined review.

    Mellea retains its last JSON sample after exhausting repair turns. That
    sample can parse while still failing the exact locator/number requirement.
    It is evidence of a decline, never a promotion input: only a successful IVR
    run may create a docket record.
    """
    if not result.success:
        return SiteReview(answer=None, reason=result.failure_reason, run=result)
    try:
        proposal = _parse(result.output)
    except ValidationError:
        return SiteReview(answer=None, reason=result.failure_reason, run=result)
    if not proposal.is_docket_citation or not proposal.locator or not proposal.docket_number:
        return SiteReview(answer=None, reason=proposal.reason, run=result)
    return SiteReview(
        answer=RecoveredDocketLocator(
            locator_text=proposal.locator,
            docket_number=proposal.docket_number,
            reason=proposal.reason,
        ),
        reason=proposal.reason,
        run=result,
    )
