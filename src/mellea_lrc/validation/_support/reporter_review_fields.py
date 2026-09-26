"""Materialize one model review as citation readings and field judgments."""

from __future__ import annotations

from mellea_lrc.courtlistener import CourtListenerDocket
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.span import Span
from mellea_lrc.validation._support.reporter_review_grounding import ReporterReviewDecision


def append_corrections(
    root: FullReporterCitation,
    source: str,
    corrections: dict[str, Span],
    decision: ReporterReviewDecision,
) -> FullReporterCitation:
    """Write a model name normalization and any changed source readings."""
    name_span = corrections.get("case_name")
    if name_span is None and root.case_name:
        name_span = root.case_name[-1].span
    normalized_name = decision.case_name.normalized
    if name_span is not None and normalized_name is not None:
        prior = root.case_name[-1] if root.case_name else None
        if (
            prior is None
            or prior.span != name_span
            or not prior.normalizable
            or prior.get_normalized() != normalized_name
        ):
            root = root.with_case_name(source, name_span, normalized=normalized_name)
    for field, span in corrections.items():
        if field == "case_name":
            continue
        prior = getattr(root, field)
        if prior and prior[-1].span == span:
            continue
        if field == "court":
            root = root.with_court(source, span)
        else:
            root = root.with_date(source, span)
    return root


def append_field_judgments(
    root: FullReporterCitation,
    decision: ReporterReviewDecision,
    candidate_index: int,
) -> FullReporterCitation:
    """Point each field judgment to its newest reading and chosen cluster."""
    root = root.with_case_name_judgment(
        len(root.case_name) - 1 if root.case_name else None,
        candidate_index,
        decision.case_name.result,
    )
    root = root.with_court_judgment(
        len(root.court) - 1 if root.court else None,
        candidate_index,
        decision.court.result,
    )
    return root.with_date_judgment(
        len(root.date) - 1 if root.date else None,
        candidate_index,
        decision.date.result,
    )


def identity_verdict(
    root: FullReporterCitation,
    decision: ReporterReviewDecision,
    candidate_index: int,
    docket: CourtListenerDocket | None,
) -> IdentityVerdict:
    """Derive identity from independent field decisions for one saved cluster."""
    if any(
        assessment.result is MatchResult.MISMATCH
        for assessment in (decision.case_name, decision.court, decision.date)
    ):
        return IdentityVerdict.WRONG_IDENTITY
    lookup = root.reporter_exact_lookup
    if lookup is None or lookup.response is None:
        raise ValueError("Reporter review requires saved lookup candidates")
    candidate = lookup.response.clusters[candidate_index]
    if (
        not root.case_name
        or not (candidate.case_name_full or candidate.case_name)
        or decision.case_name.result is not MatchResult.MATCH
    ):
        return IdentityVerdict.DEFERRED
    has_candidate_court = bool(
        candidate.court_id or candidate.court or (docket and (docket.court_id or docket.court))
    )
    if root.court and has_candidate_court and decision.court.result is MatchResult.UNDETERMINED:
        return IdentityVerdict.DEFERRED
    if root.date and candidate.date_filed and decision.date.result is MatchResult.UNDETERMINED:
        return IdentityVerdict.DEFERRED
    return IdentityVerdict.CORRECT_IDENTITY
