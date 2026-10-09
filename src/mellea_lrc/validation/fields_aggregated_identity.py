"""Materialize overall identity from a selected docket record's field judgments."""

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.document import Document

SUBSTAGE = "validate_roots.docket_lookup.identity_aggregation"
NEXT_SUBSTAGE = "validate_roots.locator_body_corroboration.courtlistener_opinion_retrieval"


def fields_aggregated_identity(document: Document) -> Document:
    """Append a verdict for each docket root routed here; no retrieval or model call.

    A mismatch in any compared field establishes wrong identity. Docket number
    and case name must be available to establish correct identity; unavailable
    court or date does not contradict it. Reviews without a selected record
    remain eligible for body search and do not issue an identity here.
    """
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if "validate_roots.docket_lookup.govinfo_review" not in document.substage_runs:
        raise ValueError("Complete docket record reviews before aggregating identity")
    for root in document.roots:
        if root.next_substage != SUBSTAGE:
            continue
        if not isinstance(root, FullDocketCitation):
            raise ValueError("Field aggregation currently supports docket roots only")
        if root.identity_judgments or root.body_reviews:
            raise ValueError("A docket root awaiting field aggregation already has an identity review")
        decision = None
        for review in (root.govinfo_docket_review, root.docket_lookup_review):
            if (
                review is not None
                and review.decision is not None
                and review.decision.selected_candidate_index is not None
            ):
                decision = review.decision
                break
        if decision is None:
            raise ValueError("Field aggregation requires an accepted selected docket record")
        comparisons = (decision.docket_number, decision.case_name, decision.court, decision.date)
        if any(field.result is MatchResult.MISMATCH for field in comparisons):
            verdict = IdentityVerdict.WRONG_IDENTITY
        elif any(
            field.result is MatchResult.UNAVAILABLE for field in (decision.docket_number, decision.case_name)
        ):
            verdict = IdentityVerdict.UNDETERMINED
        else:
            verdict = IdentityVerdict.CORRECT_IDENTITY
        recorded = root.record(SUBSTAGE).with_identity_judgment(verdict)
        recorded = recorded.with_route(NEXT_SUBSTAGE if verdict is IdentityVerdict.UNDETERMINED else None)
        document = document.replace_citation(recorded)
    return document.complete_substage(SUBSTAGE)
