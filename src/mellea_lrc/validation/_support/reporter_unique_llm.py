"""One source-grounded model review of a unique reporter lookup."""

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
from mellea_lrc.llm.fuzziness import FuzzinessOption
from mellea_lrc.llm.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterUniqueReviewDecision
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span

if TYPE_CHECKING:
    from mellea import MelleaSession


@dataclass(frozen=True, slots=True)
class ReporterUniqueReviewContext:
    """Bounded source text and the sole saved opinion record for one review."""

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
    candidate: CourtListenerCluster
    docket: CourtListenerDocket | None

    @classmethod
    def from_document(cls, document: Document, root: FullReporterCitation) -> ReporterUniqueReviewContext:
        lookup = root.reporter_exact_lookup
        if lookup is None or lookup.response is None or len(lookup.response.clusters) != 1:
            raise ValueError("Unique review requires one saved reporter candidate")
        case_name_window, case_name_offset = before(document, root, 300)
        following_window, following_offset = after(document, root, 180)
        current_court = None
        if root.court:
            reading = root.court[-1]
            current_court = reading.quote
            if reading.quote is None and reading.normalizable:
                current_court = f"inferred court {reading.get_normalized().id}"
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
            candidate=lookup.response.clusters[0],
            docket=root.reporter_exact_docket.response if root.reporter_exact_docket else None,
        )

    def ground(self, field: str, quote: str) -> Span | None:
        """Resolve a proposed reading to exact source bytes in its allowed window."""
        if field == "case_name":
            window, offset = self.case_name_window, self.case_name_offset
        elif field in {"court", "date"}:
            window, offset = self.following_window, self.following_offset
        else:
            raise ValueError(f"Unknown reporter review field: {field}")
        found = GroundingEvidence((EvidenceCandidate(window, offset),)).find_fragment(
            quote,
            FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True),
        )
        if found is None:
            return None
        return Span(offset + found.start, offset + found.end)

    def grounded_corrections(self, decision: ReporterUniqueReviewDecision) -> dict[str, Span] | None:
        """Validate stated intent and ground every quote before a correction commits."""
        corrections: dict[str, Span] = {}
        for field in ("case_name", "court", "date"):
            assessment = getattr(decision, field)
            quote = assessment.quote
            if not assessment.propose_replacement:
                if quote is not None:
                    return None
                continue
            if quote is None or not quote.strip():
                return None
            span = self.ground(field, quote)
            if span is None:
                return None
            corrections[field] = span
        return corrections

    def assessment_error(self, decision: ReporterUniqueReviewDecision) -> str | None:
        """A match or mismatch needs a source reading, existing or proposed."""
        for field in ("case_name", "court", "date"):
            assessment = getattr(decision, field)
            if not getattr(self, f"has_{field}") and not assessment.propose_replacement:
                if assessment.result is not MatchResult.UNDETERMINED:
                    return f"{field} has no filing reading; use undetermined or quote one from the filing"
        return None


@dataclass(frozen=True, slots=True)
class ReporterUniqueReviewOutcome:
    """A model decision or failure, retaining the complete IVR attempt log."""

    decision: ReporterUniqueReviewDecision | None
    run: IvrRun | None = None
    failure_reason: str | None = None


class ReporterUniqueReviewer(Protocol):
    def __call__(
        self, context: ReporterUniqueReviewContext
    ) -> Awaitable[ReporterUniqueReviewDecision | ReporterUniqueReviewOutcome]: ...


MAX_TOKENS = 3000
MAX_MODEL_ATTEMPTS = 3
SESSION_ID = "mellea-lrc-reporter-unique-review-v2"

_PREFIX = """Review one reporter citation against one retrieved opinion record. Do all rereading, correction proposals, and field comparisons in this one answer.

The reporter locator is fixed. For case name, court, and date, first reread the filing text. Set propose_replacement to true only when you intend to change or supply that field; then quote the replacement exactly from the filing. If the current reading is fine, set propose_replacement to false and quote to null. Do not quote a value merely to restate a reading you are keeping. A proposal must be within the text before the locator for case name, or after it for court and date. Do not quote values from the retrieved record as corrections to the filing.

Compare the corrected or existing filing reading with the retrieved record. For each field return match, mismatch, or undetermined and a specific reason. Conventional abbreviations and equivalent party forms can match; a misspelling is a mismatch, not an abbreviation. Compare the full date when both sides provide it, otherwise compare the available precision. An absent value gives no opinion for that field. A reporter may itself identify a court even if none is written. Do not force agreement between an opinion date and a docket filing date.

Use only the supplied filing window and record. Do not select another candidate or decide a separate overall identity verdict. Return the required structured fields: case_name, court, date, and reason."""

_INSTRUCTION = """Reporter locator: {{locator}}

Filing text before locator (case name may occur here):
{{before}}

Filing text after locator (court and date may occur here):
{{after}}

Current extracted readings (null means absent):
{{readings}}

Sole retrieved opinion record:
{{candidate}}

Linked docket court evidence, if retrieved:
{{docket}}"""


def _validate_grounding(ctx: object, context: ReporterUniqueReviewContext) -> ValidationResult:
    try:
        answer = ReporterUniqueReviewDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    assessment_error = context.assessment_error(answer)
    if assessment_error is not None:
        return ValidationResult(result=False, reason=assessment_error)
    if context.grounded_corrections(answer) is not None:
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason=(
            "A replacement must have a nonempty quote inside its allowed filing window; "
            "when propose_replacement is false, quote must be null."
        ),
    )


@dataclass(frozen=True, slots=True)
class IvrReporterUniqueReviewer:
    """One combined IVR review with a shared, cacheable instruction prefix."""

    session: MelleaSession
    model_options: dict[str, object]
    max_attempts: int = MAX_MODEL_ATTEMPTS

    @classmethod
    def from_env(cls) -> IvrReporterUniqueReviewer:
        load_dotenv(override=False)
        config = llm_api_config_from_env(os.environ)
        return cls(
            session=start_mellea_session_from_env(),
            model_options={
                **config.mellea_call_options(max_tokens=MAX_TOKENS),
                "extra_body": {"session_id": SESSION_ID},
            },
        )

    async def __call__(self, context: ReporterUniqueReviewContext) -> ReporterUniqueReviewOutcome:
        candidate = context.candidate.model_dump_json(exclude={"raw_json"})
        docket = context.docket.model_dump_json(exclude={"raw_json"}) if context.docket else "null"
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
                    "candidate": candidate,
                    "docket": docket,
                },
                output_format=ReporterUniqueReviewDecision,
                requirements=(
                    req(
                        "Ground each proposed correction in its filing window.",
                        validation_fn=lambda ctx: _validate_grounding(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return ReporterUniqueReviewOutcome(
                None, run=run, failure_reason=run.failure_reason or "IVR review failed"
            )
        return ReporterUniqueReviewOutcome(
            ReporterUniqueReviewDecision.model_validate_json(run.output), run=run
        )
