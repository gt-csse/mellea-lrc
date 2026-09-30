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
from mellea_lrc.llm.config import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.model.citation_windows import after, before
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.reporter_lookup import ReporterUniqueReviewDecision
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.validation.reporter_review.court_context import (
    inferred_reporter_court_note,
    reporter_court_context,
)
from mellea_lrc.validation.reporter_review.grounding import ReporterReviewGrounding

if TYPE_CHECKING:
    from mellea import MelleaSession


@dataclass(frozen=True, slots=True)
class ReporterUniqueReviewContext(ReporterReviewGrounding):
    """Bounded source text and the sole saved opinion record for one review."""

    source: str
    locator: str
    case_name_window: str
    case_name_offset: int
    following_window: str
    following_offset: int
    current_case_name: str | None
    current_case_name_normalized: str | None
    current_court: str | None
    current_date: str | None
    has_case_name: bool
    has_court: bool
    has_date: bool
    candidate: CourtListenerCluster
    docket: CourtListenerDocket | None
    inferred_court_note: str | None = None

    def candidate_available(self) -> dict[str, bool]:
        return {
            "case_name": bool(self.candidate.case_name_full or self.candidate.case_name),
            "court": bool(
                self.candidate.court_id
                or self.candidate.court
                or (self.docket and (self.docket.court_id or self.docket.court))
            ),
            "date": bool(self.candidate.date_filed),
        }

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
                inferred = reading.get_normalized()
                current_court = f"inferred court {inferred.id} ({inferred.name})"
        return cls(
            source=document.text,
            locator=document.text[root.locator_span.start : root.locator_span.end],
            case_name_window=case_name_window,
            case_name_offset=case_name_offset,
            following_window=following_window,
            following_offset=following_offset,
            current_case_name=root.case_name[-1].quote if root.case_name else None,
            current_case_name_normalized=(
                root.case_name[-1].get_normalized().as_citation()
                if root.case_name and root.case_name[-1].normalizable
                else None
            ),
            current_court=current_court,
            current_date=root.date[-1].quote if root.date else None,
            has_case_name=bool(root.case_name),
            has_court=bool(root.court),
            has_date=bool(root.date),
            candidate=lookup.response.clusters[0],
            docket=root.reporter_exact_docket.response if root.reporter_exact_docket else None,
            inferred_court_note=inferred_reporter_court_note(root),
        )


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
SESSION_ID = "mellea-lrc-reporter-unique-review-v9"

_PREFIX = """Review one reporter citation against one retrieved opinion record. Do all rereading, correction proposals, and field comparisons in this one answer.

The reporter locator is fixed. For case name, court, and date, first reread the filing text. Set propose_replacement to true only when you intend to change or supply that field; then quote the replacement exactly from the filing. If the current reading is fine, set propose_replacement to false and quote to null. Do not quote a value merely to restate a reading you are keeping. A proposal must be within the text before the locator for case name, or after it for court and date. Do not quote values from the retrieved record as corrections to the filing.

For case_name, also supply normalized as the structured name read from the filing. Use kind=adversarial with both printed parties, kind=in_re or kind=ex_parte with a printed subject, or kind=partial with only its partial text when the filing supplies a credible name fragment but not a complete case name. Leave plaintiff, defendant, and subject null for kind=partial; never complete a missing party from the retrieved record. Supply normalized even when keeping an existing grounded quote; use null only when no case name or fragment is grounded. The quote may contain page headers or other layout noise between name parts. Include that noise in the quoted span, but omit it from normalized. Do not take the normalized name from the retrieved record.

For court and date, judge the filing reading against the retrieved evidence directly. If you propose a replacement quote, the program will normalize that quote afterward; you do not need to supply a normalized court or date.

Compare the corrected or existing filing reading with the retrieved record. For each field return match or mismatch with a specific reason when both sides have evidence. Use unavailable if either side lacks usable evidence for that field. Make a best-effort judgment when both sides have evidence; do not use unavailable merely because equivalence is difficult to decide. Conventional abbreviations and equivalent party forms can match; a misspelling is a mismatch, not an abbreviation. Compare the full date when both sides provide it, otherwise compare the available precision. A reporter may itself identify a court even if none is written. Do not force agreement between an opinion date and a docket filing date.

Court codes, citation abbreviations, and full court names can differ while identifying the same tribunal. The supplied court-name expansions explain recognized record codes; compare the actual courts, not their spelling. A different district or department is not equivalent merely because it is nearby or shares a broader court name. The opinion record and its linked docket are separate sources of court evidence; if they conflict, weigh their provenance and explain your court assessment. You may use ordinary court-naming conventions to interpret supplied labels, but do not invent case-specific facts.

Use only the supplied filing window and record for case-specific facts. Do not select another candidate or decide a separate overall identity verdict. Return the required structured fields: case_name, court, date, and reason."""

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
{{docket}}

Known full names for retrieved court codes, kept separate by source:
{{court_name_context}}"""


def _validate_grounding(ctx: object, context: ReporterUniqueReviewContext) -> ValidationResult:
    try:
        answer = ReporterUniqueReviewDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    assessment_error = context.assessment_error(answer, candidate_available=context.candidate_available())
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
            "case_name": {
                "quote": context.current_case_name,
                "normalized": context.current_case_name_normalized,
            },
            "court": context.current_court,
            "date": context.current_date,
        }
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION
                + (
                    "\n\nReporter-based court inference:\n{{inferred_court_note}}"
                    if context.inferred_court_note
                    else ""
                ),
                prefix=_PREFIX,
                user_variables={
                    "locator": context.locator,
                    "before": context.case_name_window,
                    "after": context.following_window,
                    "readings": json.dumps(readings, ensure_ascii=False),
                    "candidate": candidate,
                    "docket": docket,
                    "court_name_context": json.dumps(
                        reporter_court_context(context.candidate, context.docket),
                        ensure_ascii=False,
                    ),
                    **(
                        {"inferred_court_note": context.inferred_court_note}
                        if context.inferred_court_note
                        else {}
                    ),
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
