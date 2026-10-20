"""One source-grounded model review of a unique reporter lookup."""

from __future__ import annotations

import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import ValidationError

from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.llm.reviewer import IvrReviewer
from mellea_lrc.model.citation_windows import after, before
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.fields.base import current_reading
from mellea_lrc.model.citations.reporter_lookup import ReporterUniqueReviewDecision
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.providers.courtlistener import CourtListenerCluster, CourtListenerDocket
from mellea_lrc.validation.reporter_review.court_context import (
    inferred_reporter_court_note,
    reporter_court_context,
)
from mellea_lrc.validation.reporter_review.grounding import ReporterReviewGrounding


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
        current_name = current_reading(root.case_name)
        current_date = current_reading(root.date)
        current_court = None
        reading = current_reading(root.court)
        if reading is not None:
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
            current_case_name=current_name.quote if current_name is not None else None,
            current_case_name_normalized=(
                current_name.get_normalized().as_citation()
                if current_name is not None and current_name.normalizable
                else None
            ),
            current_court=current_court,
            current_date=current_date.quote if current_date is not None else None,
            has_case_name=current_name is not None,
            has_court=reading is not None,
            has_date=current_date is not None,
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


_PREFIX = """Review one reporter citation against one retrieved opinion record. Do all rereading, correction proposals, and field comparisons in this one answer.

The reporter locator is fixed. For case name, court, and date, first reread the filing text. use action=replace to change or supply a reading and quote the replacement exactly from the filing. Use action=keep with quote=null to preserve the reading. Use action=not_stated with quote=null, result=unavailable, and a reason when the filing supplies no applicable field, including when the current reading was borrowed from another citation or nearby prose. For a not_stated case name, normalized must be null. Do not clear a field merely because it disagrees with a record; that is a mismatch. Case name must come from before the locator; court and date must come from after it. Do not quote a value merely to restate a reading you are keeping. A proposal must be within the text before the locator for case name, or after it for court and date. Do not quote values from the retrieved record as corrections to the filing.

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
            "keep and not_stated require quote=null; not_stated also requires result=unavailable."
        ),
    )


@dataclass(frozen=True, slots=True)
class IvrReporterUniqueReviewer(IvrReviewer):
    """One combined IVR review with a shared, cacheable instruction prefix."""

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
