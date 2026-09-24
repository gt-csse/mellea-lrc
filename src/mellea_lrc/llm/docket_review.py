"""Mellea IVR review of one proposed docket locator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import ValidationError

from mellea_lrc.extraction.docket_hunting import (
    DocketReviewOutcome,
    DocketSiteCandidate,
    DocketSiteDecision,
    _grounded,
)
from mellea_lrc.llm.config import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr

if TYPE_CHECKING:
    from mellea import MelleaSession


MAX_TOKENS = 1800
MAX_MODEL_ATTEMPTS = 3  # Initial answer plus at most two repairs.
SESSION_ID = "mellea-lrc-docket-site-hunting-v1"

# The invariant instruction is a prefix so repeated candidate reviews can use
# the provider's prompt cache. It describes the task without jurisdictional or
# corpus-specific docket conventions.
_PREFIX = """Decide whether a proposed span in a legal filing is a docket locator cited for a court case. A docket number is a court-assigned identifier for a case or proceeding.

Quote the complete proposed locator and its docket-number portion as written. Do not add nearby court, date, pinpoint, or case-name text. An exhibit number, docket-entry number, statute, filing reference, or internal cross-reference is not a case docket citation. Give a short reason for either decision.

Return only the required structured fields: is_docket_citation, locator, docket_number, reason. If this is not a docket citation, set the two quoted fields to null."""

_INSTRUCTION = """Proposed span: {{locator}}

Decide whether that span cites a case docket. Keep the quoted fields within the proposed span.

Filing window:
{{window}}"""


def _validate_grounding(ctx: object, candidate: DocketSiteCandidate) -> ValidationResult:
    """Let the wrapper handle schema repair, then check source evidence."""
    try:
        answer = DocketSiteDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    if not answer.is_docket_citation or _grounded(candidate, answer):
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason=(
            "The locator and docket_number must reproduce the proposed span and its "
            "docket-number portion. Copy their source characters; whitespace variation is permitted."
        ),
    )


@dataclass(frozen=True, slots=True)
class IvrDocketReviewer:
    """One bounded IVR decision, with every attempt retained on the result."""

    session: MelleaSession
    model_options: dict[str, object]
    max_attempts: int = MAX_MODEL_ATTEMPTS

    @classmethod
    def from_env(cls) -> IvrDocketReviewer:
        load_dotenv(override=False)
        config = llm_api_config_from_env(os.environ)
        return cls(
            session=start_mellea_session_from_env(),
            model_options={
                **config.mellea_call_options(max_tokens=MAX_TOKENS),
                "extra_body": {"session_id": SESSION_ID},
            },
        )

    async def __call__(self, candidate: DocketSiteCandidate) -> DocketReviewOutcome:
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=_PREFIX,
                user_variables={"locator": candidate.locator_text, "window": candidate.context},
                output_format=DocketSiteDecision,
                requirements=(
                    req(
                        "Quote the candidate's complete locator and docket number from the source.",
                        validation_fn=lambda ctx: _validate_grounding(ctx, candidate),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return DocketReviewOutcome(None, run=run, failure_reason=run.failure_reason)
        # A successful IVR run has passed schema validation. The stage still
        # verifies grounding before admission as a final boundary check.
        return DocketReviewOutcome(DocketSiteDecision.model_validate_json(run.output), run=run)
