"""Review one saved CourtListener shortlist for each docket root."""

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

from mellea_lrc.courtlistener.models import CourtListenerSearchResult
from mellea_lrc.extraction._support.context_windows import after, before
from mellea_lrc.llm.config import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookupFailure,
    DocketLookupFieldAssessment,
    DocketLookupReview,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.fields.court import Court
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun

if TYPE_CHECKING:
    from mellea import MelleaSession


STAGE = "docket_root_lookup_review"
MAX_TOKENS = 4000
MAX_MODEL_ATTEMPTS = 3
SESSION_ID = "mellea-lrc-docket-review-v1"

_PREFIX = """Review one docket citation against every shortlisted CourtListener search record in one answer. Select the best candidate by its candidate_index, or select null if the supplied evidence does not support any candidate. The shortlisted records are the complete choice set. A similarity score helps find plausible docket numbers; it is not a case-identity verdict.

The cited case name, court, date, or even docket number may be wrong. Candidate choice asks which record the locator identifies, not whether every field in the citation is correct. When an equivalent docket number and compatible court identify a record, select it even if its case name disagrees; report that disagreement as a case-name mismatch. Do not reject the record merely because a field you are checking is wrong. If the locator and independent context do not identify any record, select null. Compare docket number, case name, court, and date independently with the selected record; return match, mismatch, or undetermined and a specific reason for each. If you select null, all four field results must be undetermined. Do not issue an overall identity verdict or change the citation's extracted readings.

For docket numbers, consider meaningful formatting differences such as punctuation, leading zeroes, and omitted administrative prefixes for a division or case type, but do not assume distinct numbers are equivalent. For case names, consider conventional abbreviations and equivalent party forms, but treat a misspelling as a mismatch. For courts, compare the actual tribunal, including district or department, rather than relying on similar labels. A federal district court and a bankruptcy court in that district are different courts. The citation's written court label determines what it states; do not silently add a bankruptcy designation from the docket number, nearby context, or retrieved record. Court-name context expands known court codes on each side while preserving the original labels. Use undetermined when the evidence cannot establish a comparison.

The cited date is an opinion or decision date. In a type=d docket search record, dateFiled is the date the case docket was initiated; it does not establish the cited opinion date, so the date assessment must be undetermined for a selected type=d record. In a type=o opinion search record, dateFiled is the opinion-record filing date and can be compared with the cited opinion date at the precision stated in the citation. Do not treat a docket initiation date as an opinion date merely because the record also contains a docket_id.

Different search hits can describe the same case but different opinions, orders, or docket cards. When more than one hit identifies the case, prefer a record that also supports the cited decision date, if one is supplied. If no opinion record supports that date, a matching docket card can still identify the case, but its initiation date leaves the cited decision date undetermined. Do not prefer a differently dated opinion merely because it appears first.

The search-status summary shows whether pagination stopped with another page available or a provider failure occurred. Such a partial candidate set can still be reviewed; make a best-effort choice from the supplied records and explain the limitation in your reason. Do not imply that unseen search results were ruled out.

Use only the supplied filing context and search records for case-specific facts. Explain why the selected record represents the same case, or why none can be selected. Return the required structured fields: selected_candidate_index, docket_number, case_name, court, date, and reason."""

_INSTRUCTION = """Cited docket locator: {{locator}}

Filing text before the locator:
{{before}}

Filing text after the locator:
{{after}}

Current extracted readings:
{{readings}}

Search completeness and failures:
{{search_status}}

Shortlisted CourtListener records (candidate_index is the saved lookup index):
{{candidates}}"""


def _short_string(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    return value if len(value) <= limit else value[:limit] + "…"


def _short_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        name = item.get("name") or item.get("partyName") if isinstance(item, dict) else item
        if short := _short_string(name, 120):
            names.append(short)
        if len(names) == 4:
            break
    return names


def _record_summary(
    record: dict[str, object], parsed: CourtListenerSearchResult, source_type: str
) -> dict[str, object]:
    """Bound model context while leaving the complete result in the saved lookup."""
    fields: dict[str, object] = {
        "docketNumber": _short_string(parsed.docket_number, 180),
        "caseName": _short_string(parsed.case_name, 180),
        "caseNameFull": _short_string(parsed.case_name_full, 220),
        "court": _short_string(parsed.court, 160),
        "court_id": _short_string(parsed.court_id, 80),
        "dateFiled": _short_string(parsed.date_filed, 40),
        "dateFiled_meaning": ("case_docket_initiation" if source_type == "d" else "opinion_record_filing"),
        "absolute_url": _short_string(parsed.absolute_url, 200),
        "docket_id": parsed.docket_id,
        "cluster_id": parsed.cluster_id,
    }
    summary = {key: value for key, value in fields.items() if value is not None}
    names = _short_names(record.get("parties") or record.get("party"))
    if names:
        summary["party_names"] = names
    citations = _short_names(record.get("citation"))
    if citations:
        summary["citations"] = citations[:3]
    snippets: list[str] = []
    if snippet := _short_string(record.get("snippet"), 240):
        snippets.append(snippet)
    opinions = record.get("opinions")
    if isinstance(opinions, list):
        for opinion in opinions:
            if isinstance(opinion, dict) and (snippet := _short_string(opinion.get("snippet"), 240)):
                snippets.append(snippet)
                break
    if snippets:
        summary["snippets"] = snippets
    return summary


def _failure_summary(failure: DocketLookupFailure | None) -> dict[str, object] | None:
    if failure is None:
        return None
    return {
        "failure_type": failure.failure_type,
        "message": _short_string(failure.message, 200),
        "upstream_status_code": failure.upstream_status_code,
        "url": _short_string(failure.url, 240),
    }


@dataclass(frozen=True, slots=True)
class DocketLookupReviewContext:
    """Bounded source context and all shortlisted records for one docket root."""

    locator: str
    before_window: str
    after_window: str
    readings: dict[str, str | None]
    search_status: dict[str, object]
    candidates: tuple[dict[str, object], ...]
    shortlisted_candidate_indices: tuple[int, ...]

    @staticmethod
    def _court_name_context(record: CourtListenerSearchResult) -> dict[str, str | None]:
        raw_id = record.court_id
        if raw_id is None:
            return {"raw_court_id": None, "full_name": None}
        try:
            full_name = Court.from_id(raw_id).name
        except ValueError:
            full_name = None
        return {"raw_court_id": raw_id, "full_name": full_name}

    @classmethod
    def from_document(cls, document: Document, root: FullDocketCitation) -> DocketLookupReviewContext:
        lookup = root.docket_lookup
        if lookup is None:
            raise ValueError("Docket review requires a saved lookup")
        before_window, _ = before(document, root, 300)
        after_window, _ = after(document, root, 240)
        locator = root.locator[-1]
        court_reading = root.court[-1] if root.court else None
        court = court_reading.quote if court_reading else None
        if court_reading is not None and court_reading.normalizable:
            normalized = court_reading.get_normalized()
            court = f"{court or 'inferred'} [court ID: {normalized.id}; {normalized.name}]"
        attempts = [
            {
                "source_type": attempt.source_type,
                "query": _short_string(attempt.query, 220),
                "pages_seen": len(attempt.pages),
                "retries": len(attempt.retry_failures),
                "next_page_available": bool(attempt.pages and attempt.pages[-1].get("next")),
                "failure": _failure_summary(attempt.failure),
            }
            for attempt in lookup.attempts
        ]
        search_status = {
            "attempts": attempts,
            "lookup_failure": _failure_summary(lookup.failure),
            "incomplete": bool(
                lookup.failure or any(item["next_page_available"] or item["failure"] for item in attempts)
            ),
        }
        candidates: list[dict[str, object]] = []
        for index in lookup.shortlisted_candidate_indices:
            pointer = lookup.candidates[index]
            attempt = lookup.attempts[pointer.attempt_index]
            page = attempt.pages[pointer.page_index]
            results = page.get("results")
            if not isinstance(results, list) or pointer.result_index >= len(results):
                raise ValueError("Shortlisted docket candidate points to a missing search result")
            result = results[pointer.result_index]
            if not isinstance(result, dict):
                raise ValueError("Shortlisted docket candidate must point to a search object")
            parsed = CourtListenerSearchResult.model_validate(result)
            candidates.append(
                {
                    "candidate_index": index,
                    "source_type": pointer.source_type,
                    "record_id": pointer.record_id,
                    "docket_number": pointer.docket_number,
                    "docket_similarity": pointer.docket_similarity,
                    "court_name_context": cls._court_name_context(parsed),
                    "record_summary": _record_summary(result, parsed, pointer.source_type),
                }
            )
        return cls(
            locator=locator.quote,
            before_window=before_window,
            after_window=after_window,
            readings={
                "docket_number": document.text[locator.number_span.start : locator.number_span.end],
                "case_name": root.case_name[-1].quote if root.case_name else None,
                "court": court,
                "date": root.date[-1].quote if root.date else None,
            },
            search_status=search_status,
            candidates=tuple(candidates),
            shortlisted_candidate_indices=lookup.shortlisted_candidate_indices,
        )

    def choice_error(self, decision: DocketLookupReviewDecision) -> str | None:
        index = decision.selected_candidate_index
        if index is None:
            return None
        if index not in self.shortlisted_candidate_indices:
            return f"selected_candidate_index must be one of {self.shortlisted_candidate_indices}, or null"
        candidate = next(item for item in self.candidates if item["candidate_index"] == index)
        if candidate["source_type"] == "d" and decision.date.result is not MatchResult.UNDETERMINED:
            return "A type=d docket dateFiled cannot establish the cited opinion date"
        return None


@dataclass(frozen=True, slots=True)
class DocketLookupReviewOutcome:
    """A model decision or failure, with its complete IVR attempt trace."""

    decision: DocketLookupReviewDecision | None
    run: IvrRun | None = None
    failure_reason: str | None = None


class DocketLookupReviewer(Protocol):
    def __call__(
        self, context: DocketLookupReviewContext
    ) -> Awaitable[DocketLookupReviewDecision | DocketLookupReviewOutcome]: ...


def _validate_review(ctx: object, context: DocketLookupReviewContext) -> ValidationResult:
    try:
        answer = DocketLookupReviewDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    error = context.choice_error(answer)
    return ValidationResult(result=error is None, reason=error)


@dataclass(frozen=True, slots=True)
class IvrDocketLookupReviewer:
    """One structured IVR call over a docket root's saved shortlist."""

    session: MelleaSession
    model_options: dict[str, object]
    max_attempts: int = MAX_MODEL_ATTEMPTS

    @classmethod
    def from_env(cls) -> IvrDocketLookupReviewer:
        load_dotenv(override=False)
        config = llm_api_config_from_env(os.environ)
        return cls(
            session=start_mellea_session_from_env(),
            model_options={
                **config.mellea_call_options(max_tokens=MAX_TOKENS),
                "extra_body": {"session_id": SESSION_ID},
            },
        )

    async def __call__(self, context: DocketLookupReviewContext) -> DocketLookupReviewOutcome:
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=_PREFIX,
                user_variables={
                    "locator": context.locator,
                    "before": context.before_window,
                    "after": context.after_window,
                    "readings": json.dumps(context.readings, ensure_ascii=False),
                    "search_status": json.dumps(
                        context.search_status, ensure_ascii=False, separators=(",", ":")
                    ),
                    "candidates": json.dumps(context.candidates, ensure_ascii=False, separators=(",", ":")),
                },
                output_format=DocketLookupReviewDecision,
                requirements=(
                    req(
                        "Choose only a shortlisted record and assess its supported date evidence.",
                        validation_fn=lambda ctx: _validate_review(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return DocketLookupReviewOutcome(
                decision=None, run=run, failure_reason=run.failure_reason or "IVR review failed"
            )
        try:
            decision = DocketLookupReviewDecision.model_validate_json(run.output)
        except ValidationError as exc:
            return DocketLookupReviewOutcome(
                decision=None, run=run, failure_reason=f"IVR output did not match the review schema: {exc}"
            )
        return DocketLookupReviewOutcome(decision=decision, run=run)


def _no_candidate_decision(context: DocketLookupReviewContext) -> DocketLookupReviewDecision:
    reason = "No shortlisted CourtListener record is available for comparison."
    undetermined = DocketLookupFieldAssessment(result=MatchResult.UNDETERMINED, reason=reason)
    selection_reason = (
        "The search stopped before all results were available, and no candidate was shortlisted "
        "from the saved pages."
        if context.search_status["incomplete"]
        else "The docket search produced no shortlisted candidate to select."
    )
    return DocketLookupReviewDecision(
        selected_candidate_index=None,
        docket_number=undetermined,
        case_name=undetermined,
        court=undetermined,
        date=undetermined,
        reason=selection_reason,
    )


async def docket_root_lookup_review(
    document: Document, *, reviewer: DocketLookupReviewer | None = None
) -> Document:
    """Review each docket root once, retaining the choice or complete failure."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "docket_root_lookup" not in document.stage_runs:
        raise ValueError("Complete docket root lookup before its model review")
    service = reviewer
    for root in tuple(item for item in document.roots if isinstance(item, FullDocketCitation)):
        if root.docket_lookup is None:
            raise ValueError("Docket root review requires a saved lookup on every docket root")
        context = DocketLookupReviewContext.from_document(document, root)
        recorded = root.record(STAGE)
        if not context.candidates:
            review = DocketLookupReview(
                node_id=recorded.nodes[-1].id, decision=_no_candidate_decision(context)
            )
        else:
            if service is None:
                service = IvrDocketLookupReviewer.from_env()
            result = await service(context)
            outcome = (
                result
                if isinstance(result, DocketLookupReviewOutcome)
                else DocketLookupReviewOutcome(decision=result)
            )
            failure = outcome.failure_reason
            if outcome.run is not None and not outcome.run.success:
                failure = failure or outcome.run.failure_reason or "IVR review failed"
            if outcome.decision is not None and (error := context.choice_error(outcome.decision)):
                failure = error
            review = (
                DocketLookupReview(
                    node_id=recorded.nodes[-1].id,
                    ivr=outcome.run,
                    failure_reason=failure or "Model review produced no decision",
                )
                if failure is not None or outcome.decision is None
                else DocketLookupReview(
                    node_id=recorded.nodes[-1].id,
                    decision=outcome.decision,
                    ivr=outcome.run,
                )
            )
        document = document.replace_citation(recorded.with_docket_lookup_review(review))
    return document.complete(STAGE)
