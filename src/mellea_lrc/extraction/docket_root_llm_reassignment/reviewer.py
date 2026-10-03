"""Stage-local IVR partition review for docket-root reassignment."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy

from mellea_lrc.extraction.docket_root_llm_reassignment.context import DocketRootReviewContext
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.llm.reviewer import IvrReviewer
from mellea_lrc.model.citations.docket_root_llm_reassignment import DocketRootPartition
from mellea_lrc.model.ivr import IvrRun


@dataclass(frozen=True, slots=True)
class DocketRootReviewOutcome:
    """A reviewed partition or failure with the complete model attempt history."""

    decision: DocketRootPartition | None
    run: IvrRun | None = None
    failure_reason: str | None = None


class DocketRootReviewer(Protocol):
    def __call__(
        self, context: DocketRootReviewContext
    ) -> Awaitable[DocketRootPartition | DocketRootReviewOutcome]: ...


_PREFIX = """Determine which docket citations in one filing refer to the same cited case or proceeding. Docket-number similarity only selected candidates for review; it is not proof of identity. Compare the numbers and the nearby case names, courts, dates, and filing context. Different cases can have similar docket numbers. Different written docket forms can refer to one case. Do not merge merely related cases or proceedings.

Return a partition of the candidate indices. Put indices in one group only when the citations refer to the same case. Include every index exactly once; use a one-index group when that citation is distinct or uncertain. Give a concise reason. Do not choose or write root IDs; the program handles assignments."""

_INSTRUCTION = """Partition these docket-root candidates by cited case identity. Candidate indices run from 0 through {{last_index}}. Return only groups and reason in the required structured JSON schema.

Candidates:
{{candidates}}"""


def _validate_partition(ctx: object, review: DocketRootReviewContext) -> ValidationResult:
    # The shared IVR wrapper handles schema errors first, including bad JSON.
    try:
        decision = DocketRootPartition.model_validate_json(str(ctx.last_output().value))
    except ValueError:
        return ValidationResult(result=True)
    error = review.partition_error(decision)
    return ValidationResult(result=error is None, reason=error)


@dataclass(frozen=True, slots=True)
class IvrDocketRootReviewer(IvrReviewer):
    """One structured partition with bounded IVR repair and a cached prompt prefix."""

    async def __call__(self, context: DocketRootReviewContext) -> DocketRootReviewOutcome:
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=_PREFIX,
                user_variables={
                    "last_index": str(len(context.candidates) - 1),
                    "candidates": context.prompt_candidates(),
                },
                output_format=DocketRootPartition,
                requirements=(
                    req(
                        "Partition every candidate index exactly once, including singletons.",
                        validation_fn=lambda ctx: _validate_partition(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return DocketRootReviewOutcome(None, run=run, failure_reason=run.failure_reason)
        return DocketRootReviewOutcome(DocketRootPartition.model_validate_json(run.output), run=run)
