"""One grounded model choice and field review over bounded reporter candidates."""

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
from pydantic import ValidationError

from mellea_lrc.courtlistener import CourtListenerCluster, CourtListenerDocket
from mellea_lrc.extraction._support.context_windows import after, before
from mellea_lrc.llm.config import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.reporter_lookup import ReporterAmbiguousReviewDecision
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.validation._support.reporter_review_grounding import ReporterReviewGrounding
from mellea_lrc.validation.stage_names import REPORTER_ROOT_LOOKUP_AMBIGUOUS

if TYPE_CHECKING:
    from mellea import MelleaSession


@dataclass(frozen=True, slots=True)
class ReporterAmbiguousReviewContext(ReporterReviewGrounding):
    """Bounded filing text, every saved candidate, and prior rule assessments."""

    source: str
    locator: str
    case_name_window: str
    case_name_offset: int
    following_window: str
    following_offset: int
    current_case_name: str | None
    current_court: str | None
    current_date: str | None
    has_case_name: bool
    has_court: bool
    has_date: bool
    candidates: tuple[CourtListenerCluster, ...]
    dockets: tuple[CourtListenerDocket | None, ...]
    rule_results: tuple[dict[str, str | None], ...]
    passing_candidate_indices: tuple[int, ...]

    @classmethod
    def from_document(cls, document: Document, root: FullReporterCitation) -> ReporterAmbiguousReviewContext:
        lookup = root.reporter_exact_lookup
        resolution = root.reporter_exact_ambiguity_resolution
        if lookup is None or lookup.response is None or resolution is None:
            raise ValueError("Ambiguous review requires saved lookup and rule assessment")
        candidates = tuple(lookup.response.clusters)
        case_name_window, case_name_offset = before(document, root, 300)
        following_window, following_offset = after(document, root, 180)
        current_court = None
        if root.court:
            reading = root.court[-1]
            current_court = reading.quote
            if reading.quote is None and reading.normalizable:
                current_court = f"inferred court {reading.get_normalized().id}"
        docket_by_index = {
            item.candidate_index: item.response for item in root.reporter_exact_candidate_dockets
        }
        rule_results: list[dict[str, str | None]] = []
        for index in range(len(candidates)):
            results: dict[str, str | None] = {}
            for field in ("case_name", "court", "date"):
                matches = [
                    judgment
                    for judgment in getattr(root, f"{field}_judgments")
                    if judgment.node_id == resolution.node_id and judgment.candidate_index == index
                ]
                results[field] = matches[0].result.value if matches else None
            rule_results.append(results)
        return cls(
            source=document.text,
            locator=document.text[root.locator_span.start : root.locator_span.end],
            case_name_window=case_name_window,
            case_name_offset=case_name_offset,
            following_window=following_window,
            following_offset=following_offset,
            current_case_name=root.case_name[-1].quote if root.case_name else None,
            current_court=current_court,
            current_date=root.date[-1].quote if root.date else None,
            has_case_name=bool(root.case_name),
            has_court=bool(root.court),
            has_date=bool(root.date),
            candidates=candidates,
            dockets=tuple(docket_by_index.get(index) for index in range(len(candidates))),
            rule_results=tuple(rule_results),
            passing_candidate_indices=resolution.passing_candidate_indices,
        )

    def selected_docket(self, index: int) -> CourtListenerDocket | None:
        return self.dockets[index]

    def choice_error(self, decision: ReporterAmbiguousReviewDecision) -> str | None:
        index = decision.selected_candidate_index
        if index is not None and index >= len(self.candidates):
            return f"selected_candidate_index must be one of 0 through {len(self.candidates) - 1}, or null"
        return self.assessment_error(decision)


@dataclass(frozen=True, slots=True)
class ReporterAmbiguousReviewOutcome:
    """A model choice or failure with its inspectable IVR history."""

    decision: ReporterAmbiguousReviewDecision | None
    run: IvrRun | None = None
    failure_reason: str | None = None


class ReporterAmbiguousReviewer(Protocol):
    def __call__(
        self, context: ReporterAmbiguousReviewContext
    ) -> Awaitable[ReporterAmbiguousReviewDecision | ReporterAmbiguousReviewOutcome]: ...


MAX_TOKENS = 5000
MAX_MODEL_ATTEMPTS = 3
SESSION_ID = "mellea-lrc-reporter-ambiguous-review-v1"

_PREFIX = """Review one reporter citation against the complete bounded list of retrieved opinion records. In one answer, choose the single best representative record (by its candidate_index) or select null when none is supportable; reread the filing's case name, court, and date; and compare each field with your chosen record.

Every candidate remains available, including candidates that failed a preliminary rule comparison. Those comparisons are hints, not a filter or a verdict. If several records plausibly represent the same case, choose the best representative with a reason. A wrong case name, court, or date in the filing does not by itself remove the real record from consideration: select it when the locator and context support it, then mark that field mismatch. Do not invent another candidate, alter the reporter locator, or use outside knowledge.

For each filing field, set propose_replacement to true only to change or supply its reading, and quote replacement text exactly from the filing. Case name must come from before the locator; court and date must come from after it. Otherwise set propose_replacement to false and quote to null. Do not quote a retrieved record as a filing correction. Compare the corrected or existing filing reading with the selected candidate and return match, mismatch, or undetermined with a specific reason for each field. Conventional abbreviations and equivalent party forms can match; a misspelling is a mismatch, not an abbreviation. An absent value gives no opinion for that field. If you select null, all three comparisons must be undetermined, though you may still correct filing readings for later search.

The cluster's date_filed is an opinion-record date, not a linked docket's case-filing date; it may differ from a reporter publication year. Compare at the precision the filing states, but use undetermined if the supplied evidence does not establish the claimed decision date. A linked docket here supplies court evidence only. A reporter can itself identify a court when none is written.

Use only the supplied filing windows and candidates. Do not issue an overall identity verdict. Return selected_candidate_index, case_name, court, date, and reason in the required structured output."""

_INSTRUCTION = """Reporter locator: {{locator}}

Filing text before locator (case name may occur here):
{{before}}

Filing text after locator (court and date may occur here):
{{after}}

Current extracted readings (null means absent):
{{readings}}

All retrieved candidates, linked court evidence, and preliminary rule results:
{{candidates}}"""


def _validate_review(ctx: object, context: ReporterAmbiguousReviewContext) -> ValidationResult:
    try:
        answer = ReporterAmbiguousReviewDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    if (error := context.choice_error(answer)) is not None:
        return ValidationResult(result=False, reason=error)
    if context.grounded_corrections(answer) is None:
        return ValidationResult(
            result=False,
            reason=(
                "A replacement must quote text inside its allowed filing window; "
                "when propose_replacement is false, quote must be null."
            ),
        )
    return ValidationResult(result=True)


@dataclass(frozen=True, slots=True)
class IvrReporterAmbiguousReviewer:
    """One combined, cacheable model review of all saved candidates."""

    session: MelleaSession
    model_options: dict[str, object]
    max_attempts: int = MAX_MODEL_ATTEMPTS

    @classmethod
    def from_env(cls) -> IvrReporterAmbiguousReviewer:
        load_dotenv(override=False)
        config = llm_api_config_from_env(os.environ)
        return cls(
            session=start_mellea_session_from_env(),
            model_options={
                **config.mellea_call_options(max_tokens=MAX_TOKENS),
                "extra_body": {"session_id": SESSION_ID},
            },
        )

    async def __call__(self, context: ReporterAmbiguousReviewContext) -> ReporterAmbiguousReviewOutcome:
        candidates = [
            {
                "candidate_index": index,
                "cluster": cluster.model_dump(mode="json", exclude={"raw_json"}),
                "linked_docket_court": (
                    docket.model_dump(mode="json", exclude={"raw_json"}) if docket else None
                ),
                "preliminary_rule_results": context.rule_results[index],
                "preliminary_full_match": index in context.passing_candidate_indices,
            }
            for index, (cluster, docket) in enumerate(zip(context.candidates, context.dockets))
        ]
        readings = {
            "case_name": context.current_case_name,
            "court": context.current_court,
            "date": context.current_date,
        }
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=_PREFIX,
                user_variables={
                    "locator": context.locator,
                    "before": context.case_name_window,
                    "after": context.following_window,
                    "readings": json.dumps(readings, ensure_ascii=False),
                    "candidates": json.dumps(candidates, ensure_ascii=False, separators=(",", ":")),
                },
                output_format=ReporterAmbiguousReviewDecision,
                requirements=(
                    req(
                        "Choose a saved candidate and ground every proposed filing correction.",
                        validation_fn=lambda ctx: _validate_review(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return ReporterAmbiguousReviewOutcome(
                None, run=run, failure_reason=run.failure_reason or "IVR review failed"
            )
        return ReporterAmbiguousReviewOutcome(
            ReporterAmbiguousReviewDecision.model_validate_json(run.output), run=run
        )
