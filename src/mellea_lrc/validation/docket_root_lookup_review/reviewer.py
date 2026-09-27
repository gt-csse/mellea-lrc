"""Review one saved CourtListener shortlist for each docket root."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Protocol

from dotenv import load_dotenv
from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import ValidationError

from mellea_lrc.courtlistener.models import CourtListenerSearchResult
from mellea_lrc.llm.config import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citation_windows import after, before
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookupFailure,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.fields.court import Court
from mellea_lrc.model.citations.fields.date import normalize_date
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span

if TYPE_CHECKING:
    from mellea import MelleaSession


MAX_TOKENS = 6000
MAX_MODEL_ATTEMPTS = 3
SESSION_ID = "mellea-lrc-docket-review-v4"

_PREFIX = """Review one docket citation against every shortlisted CourtListener search record in one answer. Reread the filing fields, propose any grounded corrections, and compare the resulting readings with one selected candidate. Select the best candidate by its candidate_index, or select null if the supplied evidence does not support any candidate. The shortlisted records are the complete choice set. A similarity score helps find plausible docket numbers; it is not a case-identity verdict.

For each field, set propose_replacement to true only if the current filing reading is missing or incorrect, and quote the replacement exactly from the filing. Otherwise set propose_replacement to false and quote to null. The docket number must come from the number text after its label, the case name from the filing text before the locator, and court and date from the text after its colocation group. Never copy a retrieved record as a filing correction. For case_name also supply the structured normalized name read from the filing, even when keeping the existing quote; use null only when no case name is grounded. A quoted span may contain layout noise; keep that noise in the quote but omit it from the normalized name. The program grounds proposals to source spans and normalizes court and date quotes afterward.

The cited case name, court, date, or even docket number may be wrong. Candidate choice asks which record the locator identifies, not whether every field in the citation is correct. When an equivalent docket number and compatible court identify a record, select it even if its case name disagrees; report that disagreement as a case-name mismatch. Do not reject the record merely because a field you are checking is wrong. If the locator and independent context do not identify any record, select null. Compare the corrected or existing docket number, case name, court, and date independently with the selected record. For each field, give match or mismatch and a specific reason when both sides have evidence. Use unavailable if either side lacks usable evidence for that field. Make a best-effort comparison when both sides have evidence. If you select null, use unavailable for every field; you may still correct filing readings for later stages. Do not issue an overall identity verdict.

For docket numbers, consider meaningful formatting differences such as punctuation, leading zeroes, and omitted administrative prefixes for a division or case type, but do not assume distinct numbers are equivalent. For case names, consider conventional abbreviations and equivalent party forms, but treat a misspelling as a mismatch. A docket record's case title (caption) may shorten a party list or change over time; inspect the supplied party names and do not call a field mismatched solely because a matching party is absent from the lead caption. If a historical name relationship is plausible but unsupported, make a best-effort match or mismatch based on the supplied evidence and explain the uncertainty. For courts, compare the actual tribunal, including district or department, rather than relying on similar labels. A federal district court and a bankruptcy court in that district are different courts. The citation's written court label determines what it states; do not silently add a bankruptcy designation from the docket number, nearby context, or retrieved record. Court-name context expands known court codes on each side while preserving the original labels.

The cited date is an opinion or decision date. In a type=d docket search record, dateFiled is when the case docket was initiated. Judge only chronological compatibility: a cited decision date on or after the docket filing date is compatible (match); an earlier cited decision date is incompatible (mismatch). A cited year alone is compatible when it is the filing year or later, because the day is unstated. If either date is missing or unreadable, use unavailable. This compatibility check does not establish the exact opinion date. In a type=o opinion search record, dateFiled is the opinion-record filing date and can be compared with the cited opinion date at the precision stated in the citation.

Different search hits can describe the same case but different opinions, orders, or docket cards. When more than one hit identifies the case, prefer a record that also supports the cited decision date, if one is supplied. A matching docket card can establish chronological compatibility without proving the exact decision date. Do not prefer a differently dated opinion merely because it appears first.

The search-status summary shows whether pagination stopped with another page available or a provider failure occurred. Such a partial candidate set can still be reviewed; make a best-effort choice from the supplied records and explain the limitation in your reason. Do not imply that unseen search results were ruled out.

Use only the supplied filing context and search records for case-specific facts. Explain why the selected record represents the same case, or why none can be selected. Return the required structured fields: selected_candidate_index, docket_number, case_name, court, date, and reason, including each field's correction intent and quote."""

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
        if len(names) == 12:
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


def _docket_date_compatibility(cited_quote: str | None, filed_value: object) -> MatchResult:
    """Compare a cited decision date with case initiation at the cited precision."""
    if cited_quote is None:
        return MatchResult.UNAVAILABLE
    if not isinstance(filed_value, str):
        return MatchResult.UNAVAILABLE
    try:
        cited = normalize_date(cited_quote)
        filed = date.fromisoformat(filed_value[:10])
    except ValueError:
        return MatchResult.UNAVAILABLE
    if cited.month is None:
        compatible = cited.year >= filed.year
    elif cited.day is None:
        compatible = (cited.year, cited.month) >= (filed.year, filed.month)
    else:
        compatible = date(cited.year, cited.month, cited.day) >= filed
    return MatchResult.MATCH if compatible else MatchResult.MISMATCH


@dataclass(frozen=True, slots=True)
class DocketLookupReviewContext:
    """Bounded source context and all shortlisted records for one docket root."""

    source: str
    locator: str
    number_window: str
    number_offset: int
    before_window: str
    before_offset: int
    after_window: str
    after_offset: int
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
        before_window, before_offset = before(document, root, 300)
        after_window, after_offset = after(document, root, 240)
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
            source=document.text,
            locator=locator.quote,
            number_window=document.text[locator.number_span.start : locator.span.end],
            number_offset=locator.number_span.start,
            before_window=before_window,
            before_offset=before_offset,
            after_window=after_window,
            after_offset=after_offset,
            readings={
                "docket_number": document.text[locator.number_span.start : locator.number_span.end],
                "case_name": root.case_name[-1].quote if root.case_name else None,
                "case_name_normalized": (
                    root.case_name[-1].get_normalized().as_citation()
                    if root.case_name and root.case_name[-1].normalizable
                    else None
                ),
                "court": court,
                "date": root.date[-1].quote if root.date else None,
            },
            search_status=search_status,
            candidates=tuple(candidates),
            shortlisted_candidate_indices=lookup.shortlisted_candidate_indices,
        )

    def grounded_corrections(self, decision: DocketLookupReviewDecision) -> dict[str, Span] | None:
        """Locate every proposed reading in its allowed filing window."""
        corrections: dict[str, Span] = {}
        windows = {
            "docket_number": (self.number_window, self.number_offset),
            "case_name": (self.before_window, self.before_offset),
            "court": (self.after_window, self.after_offset),
            "date": (self.after_window, self.after_offset),
        }
        for field, (window, offset) in windows.items():
            assessment = getattr(decision, field)
            if not assessment.propose_replacement:
                continue
            if assessment.quote is None:
                return None
            found = GroundingEvidence((EvidenceCandidate(window, offset),)).find_fragment(
                assessment.quote,
                FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True),
            )
            if found is None:
                return None
            corrections[field] = Span(offset + found.start, offset + found.end)
        return corrections

    def choice_error(self, decision: DocketLookupReviewDecision) -> str | None:
        corrections = self.grounded_corrections(decision)
        if corrections is None:
            return "A proposed replacement must quote text inside its allowed filing window"
        has_name = self.readings["case_name"] is not None or "case_name" in corrections
        if has_name and decision.case_name.normalized is None:
            return "A grounded case name needs its structured normalization"
        if not has_name and decision.case_name.normalized is not None:
            return "Case-name normalization needs an existing or proposed filing quote"
        index = decision.selected_candidate_index
        if index is not None and index not in self.shortlisted_candidate_indices:
            return f"selected_candidate_index must be one of {self.shortlisted_candidate_indices}, or null"
        candidate = next((item for item in self.candidates if item["candidate_index"] == index), None)
        summary = candidate["record_summary"] if candidate is not None else None
        if summary is not None and not isinstance(summary, dict):
            raise ValueError("Saved docket candidate has no record summary")
        availability = {
            "docket_number": bool(summary and summary.get("docketNumber")),
            "case_name": bool(
                summary
                and (summary.get("caseNameFull") or summary.get("caseName") or summary.get("party_names"))
            ),
            "court": bool(summary and (summary.get("court_id") or summary.get("court"))),
            "date": bool(summary and summary.get("dateFiled")),
        }
        for field in ("docket_number", "case_name", "court", "date"):
            has_reading = field == "docket_number" or self.readings[field] is not None or field in corrections
            result = getattr(decision, field).result
            if not has_reading and result is not MatchResult.UNAVAILABLE:
                return f"{field} has no filing reading; use unavailable or quote one from the filing"
            if has_reading and not availability[field] and result is not MatchResult.UNAVAILABLE:
                return f"{field} has no usable selected-record evidence; use unavailable"
            if has_reading and availability[field] and result is MatchResult.UNAVAILABLE:
                return f"{field} and the selected record both have evidence; judge match or mismatch"
        if candidate is None:
            return None
        if candidate["source_type"] == "d":
            assert summary is not None
            cited_date = (
                self.source[corrections["date"].start : corrections["date"].end]
                if "date" in corrections
                else self.readings["date"]
            )
            expected = _docket_date_compatibility(cited_date, summary.get("dateFiled"))
            if decision.date.result is not expected:
                return (
                    "For a type=d record, compare the cited decision date only for chronological "
                    f"compatibility with dateFiled; expected date result: {expected.value}"
                )
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
                        "Choose only a shortlisted record and assess its date at the supported precision.",
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
