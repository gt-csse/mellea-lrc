"""Optional, iterative review of docket sites outside the narrow rule reader.

This stage creates only source-grounded full docket locators. It neither reads
context fields nor forms groups or roots; those stages run once hunting ends.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from mellea_lrc.extraction.locators import DOCKET_PREFIX_PATTERN
from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.fields.docket import DOCKET_ENTRY_PATTERN
from mellea_lrc.model.document import Document
from mellea_lrc.model.site_review import ReviewAttempt, SiteReview
from mellea_lrc.model.span import Span
from mellea_lrc.preprocessing.document_index import is_within
from mellea_lrc.text_match import fuzzy_literal

STAGE = "docket_locator_site_hunting"
_CONTEXT = 170

# These patterns propose text for review; they do not decide docket validity.
# A labelled identifier is an opaque run rather than a jurisdiction-specific grammar.
_OPAQUE_TOKEN = r"[A-Za-z0-9](?:[A-Za-z0-9:/\\-]|\.(?=[A-Za-z0-9]|[^\S\r\n]+[A-Za-z0-9]))*"
_OPAQUE_IDENTIFIER = rf"{_OPAQUE_TOKEN}(?:[^\S\r\n]+{_OPAQUE_TOKEN}){{0,4}}"
_LABELLED = re.compile(rf"{DOCKET_PREFIX_PATTERN}(?P<number>{_OPAQUE_IDENTIFIER})", re.I)

# Unlabelled identifiers need more citation context before incurring a model call.
_BARE = re.compile(r"(?<![A-Za-z0-9:/\\-])[A-Za-z0-9][A-Za-z0-9:/\\-]{1,45}(?![A-Za-z0-9:/\\-])")
_COMPOUND = re.compile(
    r"(?<![A-Za-z0-9:/\\-])[A-Za-z0-9]{1,16}(?:[:/\\-][^\S\r\n]*[A-Za-z0-9]{1,16}){2,5}(?![A-Za-z0-9:/\\-])"
)
_CASE_CUE = re.compile(r"\bv\.\s|\bin\s+re\b", re.I)
_BARE_JOIN = re.compile(r"[^A-Za-z0-9]{0,20}\Z")


@dataclass(frozen=True, slots=True)
class DocketSiteCandidate:
    """A source span worth reviewing, not a claimed docket citation."""

    locator_span: Span
    number_span: Span
    locator_text: str
    docket_number: str
    context: str


class DocketSiteDecision(BaseModel):
    """The model's four-field answer; source grounding is a separate check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_docket_citation: bool
    locator: str | None
    docket_number: str | None
    reason: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class DocketReviewOutcome:
    """A validated answer plus the full model-attempt history, if any."""

    decision: DocketSiteDecision | None
    attempts: tuple[ReviewAttempt, ...] = ()
    failure_reason: str | None = None


class DocketSiteReviewer(Protocol):
    def __call__(
        self, candidate: DocketSiteCandidate
    ) -> Awaitable[DocketSiteDecision | DocketReviewOutcome]: ...


def _masked_text(document: Document) -> str:
    masked = list(document.text)
    for citation in document.full_locators:
        span = citation.locator_span
        masked[span.start : span.end] = " " * (span.end - span.start)
    for span in document.index_spans:
        masked[span.start : span.end] = " " * (span.end - span.start)
    # An ECF/Doc./Dkt./D.I. entry number identifies a filing within a docket,
    # not the case docket itself. Reuse the entry reader's syntax so an inner
    # "No." cannot be proposed as a full docket locator.
    for entry in DOCKET_ENTRY_PATTERN.finditer(document.text):
        masked[entry.start() : entry.end()] = " " * (entry.end() - entry.start())
    return "".join(masked)


def _candidate(document: Document, start: int, end: int, number_start: int) -> DocketSiteCandidate:
    left = max(0, start - _CONTEXT)
    right = min(len(document.text), end + _CONTEXT)
    return DocketSiteCandidate(
        locator_span=Span(start=start, end=end),
        number_span=Span(start=number_start, end=end),
        locator_text=document.text[start:end],
        docket_number=document.text[number_start:end],
        # The scan is masked, but the review sees the original citation context.
        context=document.text[left:right],
    )


def suspected_dockets(document: Document) -> tuple[DocketSiteCandidate, ...]:
    """Propose unread labelled, parallel, and citation-shaped docket sites."""
    masked = _masked_text(document)
    sites = [
        _candidate(document, *match.span(), match.start("number")) for match in _LABELLED.finditer(masked)
    ]
    reporter_spans = tuple(
        citation.locator_span
        for citation in document.full_locators
        if isinstance(citation, FullReporterCitation)
    )
    held = [site.locator_span for site in sites]
    for reporter in reporter_spans:
        for match in _BARE.finditer(masked, max(0, reporter.start - 80), reporter.start):
            start, end = match.span()
            token = masked[start:end]
            before = masked[:start].rstrip()
            if (
                end > reporter.start
                or not _BARE_JOIN.fullmatch(masked[end : reporter.start])
                or not any(char.isdigit() for char in token)
                or not any(char in ":/\\-" for char in token)
                or (before and before[-1].isalnum())
                or any(start < span.end and end > span.start for span in held)
                or any(
                    prior.end <= start and _BARE_JOIN.fullmatch(masked[prior.end : start])
                    for prior in reporter_spans
                )
            ):
                continue
            sites.append(_candidate(document, start, end, start))
            held.append(Span(start=start, end=end))

    for match in _COMPOUND.finditer(masked):
        start, end = match.span()
        if (
            not any(char.isdigit() for char in masked[start:end])
            or any(start < span.end and end > span.start for span in held)
            or not _CASE_CUE.search(masked[max(0, start - 130) : start])
        ):
            continue
        following = masked[end : end + 70]
        opening = following.find("(")
        if opening < 0 or opening > 35 or ")" not in following[opening:]:
            continue
        sites.append(_candidate(document, start, end, start))
        held.append(Span(start=start, end=end))
    return tuple(sorted(sites, key=lambda site: (site.locator_span.start, site.locator_span.end)))


def _grounded(candidate: DocketSiteCandidate, decision: DocketSiteDecision) -> bool:
    """Permit spacing noise, but never a changed docket character."""
    if not decision.locator or not decision.docket_number:
        return False
    return bool(
        re.fullmatch(fuzzy_literal(decision.locator, whitespace=True), candidate.locator_text)
        and re.fullmatch(fuzzy_literal(decision.docket_number, whitespace=True), candidate.docket_number)
    )


async def hunt_docket_locators(
    document: Document,
    *,
    reviewer: DocketSiteReviewer | None = None,
) -> Document:
    """Review one site at a time; accepted sites affect the next proposal mask."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "full_reporter_locators" not in document.stage_runs or "docket_locators" not in document.stage_runs:
        raise ValueError("Run both rule locator stages before docket site hunting")
    if "colocations" in document.stage_runs:
        raise ValueError("Docket site hunting must precede colocation")
    inspected: set[tuple[int, int]] = set()
    current = document
    while True:
        candidate = next(
            (
                site
                for site in suspected_dockets(current)
                if (site.locator_span.start, site.locator_span.end) not in inspected
            ),
            None,
        )
        if candidate is None:
            return current.complete(STAGE)
        inspected.add((candidate.locator_span.start, candidate.locator_span.end))
        if reviewer is None:
            from mellea_lrc.llm.docket_review import OpenRouterDocketReviewer

            reviewer = OpenRouterDocketReviewer.from_env()
        review = await reviewer(candidate)
        outcome = review if isinstance(review, DocketReviewOutcome) else DocketReviewOutcome(review)
        decision = outcome.decision
        citation_id: str | None = None
        if decision is None:
            result = "failed"
            reason = outcome.failure_reason or "Model review did not produce a valid decision"
        elif not decision.is_docket_citation:
            result = "declined"
            reason = decision.reason
        elif not _grounded(candidate, decision):
            result = "failed"
            reason = "Proposed locator or docket number did not ground to the candidate source span"
        else:
            result = "accepted"
            reason = decision.reason
            citation_id = f"docket:{candidate.locator_span.start}:{candidate.locator_span.end}"
            current = current.add_citation(
                FullDocketCitation.from_locator(
                    citation_id=citation_id,
                    stage=STAGE,
                    source=current.text,
                    span=candidate.locator_span,
                    number_span=candidate.number_span,
                )
            )
        current = current.add_site_review(
            SiteReview(
                stage=STAGE,
                candidate_span=candidate.locator_span,
                candidate_text=candidate.locator_text,
                outcome=result,
                reason=reason,
                proposed_locator=decision.locator if decision is not None else None,
                proposed_identifier=decision.docket_number if decision is not None else None,
                citation_id=citation_id,
                attempts=outcome.attempts,
            )
        )
