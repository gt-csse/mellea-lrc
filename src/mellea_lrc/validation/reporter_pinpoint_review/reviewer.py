"""Run IVR opinion review and retain the validated response and repair trace."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import ValidationError

from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.llm.reviewer import IvrReviewer
from mellea_lrc.model.citations.reporter_pinpoint import ReporterOpinionEvidence, ReporterSupportDecision
from mellea_lrc.model.ivr import IvrRun

from .context import _INSTRUCTION, ReporterPinpointReviewContext


@dataclass(frozen=True, slots=True)
class ReporterPinpointReviewOutcome:
    decision: ReporterSupportDecision | None
    run: IvrRun | None = None
    failure_reason: str | None = None
    passages: tuple[ReporterOpinionEvidence, ...] | None = None


class ReporterPinpointReviewer(Protocol):
    def __call__(
        self, context: ReporterPinpointReviewContext
    ) -> Awaitable[ReporterSupportDecision | ReporterPinpointReviewOutcome]: ...


def _validate_evidence(ctx: object, context: ReporterPinpointReviewContext) -> ValidationResult:
    try:
        context.response_model.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    try:
        context.parse_response(str(ctx.last_output().value), "validation")
    except ValueError as error:
        return ValidationResult(result=False, reason=str(error))
    return ValidationResult(result=True)


@dataclass(frozen=True, slots=True)
class IvrReporterPinpointReviewer(IvrReviewer):
    async def __call__(self, context: ReporterPinpointReviewContext) -> ReporterPinpointReviewOutcome:
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=context.prefix,
                user_variables={
                    "scope": context.scope.value,
                    "source_complete": str(context.source_complete),
                    "citation_quote": context.citation_quote,
                    "source_offset": str(context.source_offset),
                    "citing_context": context.citing_context,
                    "proposition": context.proposition,
                    "target": context.target,
                },
                output_format=context.response_model,
                requirements=(
                    req(
                        "Ground each evidence quote inside its named source pages, or unpaginated excerpt.",
                        validation_fn=lambda ctx: _validate_evidence(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return ReporterPinpointReviewOutcome(
                None, run=run, failure_reason=run.failure_reason or "IVR pinpoint review failed"
            )
        try:
            decision, passages = context.parse_response(run.output, "validation")
        except ValueError as error:
            return ReporterPinpointReviewOutcome(None, run=run, failure_reason=str(error))
        return ReporterPinpointReviewOutcome(decision, run=run, passages=passages)
