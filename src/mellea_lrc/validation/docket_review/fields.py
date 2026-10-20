"""Append grounded docket field corrections without making an identity decision."""

from __future__ import annotations

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import DocketLookupReviewDecision
from mellea_lrc.model.citations.fields.base import current_reading
from mellea_lrc.model.citations.judgments import FieldCorrectionAction
from mellea_lrc.parsing.span import Span


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
    for field in ("case_name", "court", "date"):
        if getattr(decision, field).action is FieldCorrectionAction.NOT_STATED:
            root = getattr(root, f"with_{field}_not_stated")()
    name_span = corrections.get("case_name")
    prior_name = current_reading(root.case_name)
    if name_span is None and prior_name is not None:
        name_span = prior_name.span
    normalized_name = decision.case_name.normalized
    if name_span is not None and normalized_name is not None:
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
        prior = current_reading(getattr(root, field))
        if prior is not None and prior.span == span:
            continue
        root = root.with_court(source, span) if field == "court" else root.with_date(source, span)
    return root
