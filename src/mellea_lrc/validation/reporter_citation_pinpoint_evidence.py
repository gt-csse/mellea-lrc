"""Prepare occurrence-level proposition and selected-page evidence."""

from __future__ import annotations

from mellea_lrc.model.citations.reporter_page_resolution import ReporterPageResolutionOutcome
from mellea_lrc.model.citations.reporter_pinpoint import (
    PinpointEvidenceOutcome,
    ReporterCitationPinpointEvidence,
)
from mellea_lrc.model.document import Document

STAGE = "44_reporter_citation_pinpoint_evidence"
SOURCE_STAGES = ("42_reporter_citation_opinion_review", "43_reporter_citation_propositions")


def prepare_reporter_citation_pinpoint_evidence(document: Document) -> Document:
    """Preserve explicit unavailable states before reviewing opinion support."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if any(stage not in document.stage_runs for stage in SOURCE_STAGES):
        raise ValueError(
            "Complete opinion selection and proposition reading before preparing pinpoint evidence"
        )

    for citation in document.citations:
        if not citation.reporter_page_resolutions:
            continue
        resolution_index = len(citation.reporter_page_resolutions) - 1
        resolution = citation.reporter_page_resolutions[resolution_index]
        proposition_index = next(
            (
                index
                for index in reversed(range(len(citation.reporter_propositions)))
                if citation.reporter_propositions[index].resolution_index == resolution_index
            ),
            None,
        )
        proposition = (
            citation.reporter_propositions[proposition_index] if proposition_index is not None else None
        )
        selections = citation.get_reporter_page_selection()
        pages = tuple(page for page in selections if page is not None)
        if resolution.outcome is ReporterPageResolutionOutcome.NO_PIN:
            outcome = PinpointEvidenceOutcome.NO_PINCITE
            reason = "This occurrence has no written or immediately inherited pinpoint."
        elif proposition is None or proposition.decision is None:
            outcome = PinpointEvidenceOutcome.READING_FAILED
            reason = (
                f"Proposition reading failed: {proposition.failure_reason}"
                if proposition is not None
                else "No proposition reading is available for this page resolution."
            )
        elif not proposition.passages:
            outcome = PinpointEvidenceOutcome.NO_PROPOSITION
            reason = "The filing occurrence supplies no attributed proposition."
        elif resolution.outcome is ReporterPageResolutionOutcome.UNNORMALIZABLE:
            outcome = PinpointEvidenceOutcome.UNNORMALIZABLE
            reason = "The source locator or pinpoint requires normalization before its target can be checked."
        elif not selections or any(page is None for page in selections):
            outcome = PinpointEvidenceOutcome.MISSING_PAGES
            reason = (
                "At least one requested page has no selected opinion source; full-opinion review is needed."
            )
        else:
            outcome = PinpointEvidenceOutcome.READY
            reason = "Grounded filing propositions and sources for every requested page are available."
        recorded = citation.record(STAGE)
        evidence = ReporterCitationPinpointEvidence(
            node_id=recorded.nodes[-1].id,
            root_id=resolution.root_id,
            resolution_index=resolution_index,
            proposition_index=proposition_index,
            pages=pages,
            outcome=outcome,
            reason=reason,
        )
        document = document.replace_citation(recorded.with_reporter_pinpoint_evidence(evidence))
    return document.complete(STAGE)
