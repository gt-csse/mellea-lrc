"""Ask once per fuzzy docket-root neighborhood which cited cases are identical."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dotenv import load_dotenv
from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy

from mellea_lrc.llm.config import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.llm.fuzziness import FuzzinessOption
from mellea_lrc.llm.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.model.citations import FullDocketCitation, latest
from mellea_lrc.model.citations.docket_root_equivalence import DocketRootPartition, DocketRootReview
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun

if TYPE_CHECKING:
    from mellea import MelleaSession


STAGE = "11_docket_root_equivalence_review"
MINIMUM_SIMILARITY_PERCENT = 40.0
MAX_TOKENS = 2500
MAX_MODEL_ATTEMPTS = 3
SESSION_ID = "mellea-lrc-docket-root-equivalence-v1"
_FUZZINESS = FuzzinessOption.edit_distance(
    similarity_percent=MINIMUM_SIMILARITY_PERCENT, whitespace_relaxation=True
)


@dataclass(frozen=True, slots=True)
class DocketRootCandidate:
    """One root and the filing evidence relevant to same-case judgment."""

    citation_id: str
    docket_number: str
    locator: str
    case_name: str | None
    court: str | None
    date: str | None
    context: str


@dataclass(frozen=True, slots=True)
class DocketRootReviewContext:
    """A connected candidate neighborhood, indexed in source order."""

    candidates: tuple[DocketRootCandidate, ...]

    @classmethod
    def from_document(
        cls, document: Document, roots: tuple[FullDocketCitation, ...]
    ) -> DocketRootReviewContext:
        candidates: list[DocketRootCandidate] = []
        for root in roots:
            locator = root.locator[-1]
            court = root.court[-1] if root.court else None
            court_text = None
            if court is not None:
                court_text = court.quote
                if court.normalizable:
                    court_text = f"{court_text or 'inferred'} [court ID: {court.get_normalized().id}]"
            span = root.locator_span
            candidates.append(
                DocketRootCandidate(
                    citation_id=root.id,
                    docket_number=document.text[locator.number_span.start : locator.number_span.end],
                    locator=locator.quote,
                    case_name=root.case_name[-1].quote if root.case_name else None,
                    court=court_text,
                    date=root.date[-1].quote if root.date else None,
                    context=document.text[max(0, span.start - 180) : min(len(document.text), span.end + 140)],
                )
            )
        return cls(candidates=tuple(candidates))

    def partition_error(self, decision: DocketRootPartition) -> str | None:
        indices = {index for group in decision.groups for index in group}
        expected = set(range(len(self.candidates)))
        if indices != expected:
            return f"Groups must contain each candidate index from 0 through {len(self.candidates) - 1} exactly once."
        return None

    def prompt_candidates(self) -> str:
        return json.dumps(
            [
                {
                    "index": index,
                    "docket_number": candidate.docket_number,
                    "locator": candidate.locator,
                    "case_name": candidate.case_name,
                    "court": candidate.court,
                    "date": candidate.date,
                    "filing_context": candidate.context,
                }
                for index, candidate in enumerate(self.candidates)
            ],
            ensure_ascii=False,
        )


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
class IvrDocketRootReviewer:
    """One structured partition with bounded IVR repair and a cached prompt prefix."""

    session: MelleaSession
    model_options: dict[str, object]
    max_attempts: int = MAX_MODEL_ATTEMPTS

    @classmethod
    def from_env(cls) -> IvrDocketRootReviewer:
        load_dotenv(override=False)
        config = llm_api_config_from_env(os.environ)
        return cls(
            session=start_mellea_session_from_env(),
            model_options={
                **config.mellea_call_options(max_tokens=MAX_TOKENS),
                "extra_body": {"session_id": SESSION_ID},
            },
        )

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


def _similar_numbers(left: str, right: str) -> bool:
    """Use the project's whitespace relaxed edit-distance policy on numbers only."""
    match = GroundingEvidence((EvidenceCandidate(right.casefold(), None),)).resolve(
        left.casefold(), _FUZZINESS
    )
    return match is not None and match.similarity_percent >= MINIMUM_SIMILARITY_PERCENT


def _candidate_components(
    document: Document,
) -> tuple[tuple[FullDocketCitation, ...], ...]:
    roots = tuple(root for root in document.roots if isinstance(root, FullDocketCitation))
    parent = list(range(len(roots)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(roots):
        left_span = left.locator[-1].number_span
        left_number = document.text[left_span.start : left_span.end]
        for right_index in range(left_index + 1, len(roots)):
            right_span = roots[right_index].locator[-1].number_span
            right_number = document.text[right_span.start : right_span.end]
            if _similar_numbers(left_number, right_number):
                parent[find(right_index)] = find(left_index)

    grouped: dict[int, list[FullDocketCitation]] = {}
    for index, root in enumerate(roots):
        grouped.setdefault(find(index), []).append(root)
    return tuple(tuple(group) for group in grouped.values() if len(group) >= 2)


async def review_docket_root_equivalence(
    document: Document, *, reviewer: DocketRootReviewer | None = None
) -> Document:
    """Review fuzzy docket-root neighborhoods and append confirmed root links.

    The threshold proposes review batches only. The model partitions each batch;
    program code chooses the earliest root of each same-case group and reassigns
    every occurrence attached to any losing root. No root assignment is erased.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "10_roots" not in document.stage_runs:
        raise ValueError("Form roots before reviewing docket-root equivalence")

    service = reviewer
    for roots in _candidate_components(document):
        context = DocketRootReviewContext.from_document(document, roots)
        if service is None:
            service = IvrDocketRootReviewer.from_env()
        returned = await service(context)
        outcome = (
            returned
            if isinstance(returned, DocketRootReviewOutcome)
            else DocketRootReviewOutcome(decision=returned)
        )
        decision = outcome.decision
        error = context.partition_error(decision) if decision is not None else None
        if error is not None and outcome.run is None:
            raise ValueError(error)
        if error is not None:
            decision = None
        anchor = roots[0].record(STAGE)
        anchor = anchor.with_docket_root_review(
            DocketRootReview(
                node_id=anchor.nodes[-1].id,
                candidate_ids=tuple(root.id for root in roots),
                decision=decision,
                ivr=outcome.run,
                failure_reason=error
                or outcome.failure_reason
                or ("Model review produced no partition" if decision is None else None),
            )
        )
        document = document.replace_citation(anchor)
        if decision is None:
            continue

        for group in decision.groups:
            if len(group) < 2:
                continue
            winner = roots[min(group)].id
            losers = {roots[index].id for index in group if roots[index].id != winner}
            # A previously deduplicated occurrence points to its former root.
            # Move the entire attachment set so no leaf is stranded on a root
            # that this review has turned into another root's leaf.
            for citation in tuple(document.full_locators):
                if latest(citation.root_id) in losers:
                    document = document.replace_citation(citation.record(STAGE).with_root(winner))
    return document.complete(STAGE)
