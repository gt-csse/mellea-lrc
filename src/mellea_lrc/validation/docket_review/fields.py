"""Append grounded docket field corrections without making an identity decision."""

from __future__ import annotations

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import DocketLookupReviewDecision
from mellea_lrc.model.span import Span


def append_corrections(
    root: FullDocketCitation,
    source: str,
    corrections: dict[str, Span],
    decision: DocketLookupReviewDecision,
) -> FullDocketCitation:
    """Append source-grounded readings under the review's decision node."""
    number_span = corrections.get("docket_number")
    if number_span is not None and number_span != root.locator[-1].number_span:
        root = root.with_docket_number(source, number_span)
    name_span = corrections.get("case_name")
    if name_span is None and root.case_name:
        name_span = root.case_name[-1].span
    normalized_name = decision.case_name.normalized
    if name_span is not None and normalized_name is not None:
        prior_name = root.case_name[-1] if root.case_name else None
        if (
            prior_name is None
            or prior_name.span != name_span
            or not prior_name.normalizable
            or prior_name.get_normalized() != normalized_name
        ):
            root = root.with_case_name(source, name_span, normalized=normalized_name)
    for field in ("court", "date"):
        span = corrections.get(field)
        if span is None:
            continue
        prior = getattr(root, field)
        if prior and prior[-1].span == span:
            continue
        root = root.with_court(source, span) if field == "court" else root.with_date(source, span)
    return root
