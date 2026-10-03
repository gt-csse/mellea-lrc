"""Resolve each reporter-family occurrence against the root's shared page index."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mellea_lrc.matching.literal import fuzzy_literal
from mellea_lrc.model.citation_chronology import citation_chronology
from mellea_lrc.model.citations import (
    Citation,
    FullReporterCitation,
    IdCitation,
    ShortReporterCitation,
    latest,
)
from mellea_lrc.model.citations.reporter_page_resolution import (
    OpinionPageReference,
    ReporterCitationPageResolution,
    ReporterPageCandidates,
    ReporterPageResolutionOutcome,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_root_opinion_page_index import STAGE as INDEX_STAGE

STAGE = "41_reporter_citation_page_resolution"


@dataclass(frozen=True)
class _ReadingSources:
    locator: Citation
    pin: Citation | None


def _same_literal(left: str, right: str) -> bool:
    return re.fullmatch(fuzzy_literal(left, whitespace=True, newline=True), right, re.I) is not None


def _resolve(
    citation: Citation, root: FullReporterCitation, sources: _ReadingSources
) -> ReporterCitationPageResolution:
    locator_log = (
        sources.locator.short_locator
        if isinstance(sources.locator, ShortReporterCitation)
        else sources.locator.locator
    )
    pin_log = sources.pin.pin_cite if sources.pin is not None else None
    pointers = {
        "node_id": citation.nodes[-1].id,
        "root_id": root.id,
        "locator_citation_id": sources.locator.id,
        "locator_reading_index": len(locator_log) - 1,
        "pin_citation_id": sources.pin.id if pin_log else None,
        "pin_reading_index": len(pin_log) - 1 if pin_log else None,
    }
    if not pin_log:
        return ReporterCitationPageResolution(
            **pointers,
            outcome=ReporterPageResolutionOutcome.NO_PIN,
            reason="No written pinpoint or immediately inherited Id. pinpoint",
        )
    if not locator_log[-1].normalizable or not pin_log[-1].normalizable:
        return ReporterCitationPageResolution(
            **pointers,
            outcome=ReporterPageResolutionOutcome.UNNORMALIZABLE,
            reason="The source locator or pinpoint requires normalization review",
        )
    locator = locator_log[-1].get_normalized()
    pages: list[ReporterPageCandidates] = []
    for target_index, target in enumerate(pin_log[-1].get_normalized()):
        for label in range(target.first, target.last + 1):
            candidates: list[OpinionPageReference] = []
            for opinion in root.reporter_root_opinion_page_index.opinions:
                for page_index, page in enumerate(opinion.pages):
                    if (page.kind is not None and page.kind != target.kind) or not _same_literal(
                        str(label), page.label.lstrip("*¶").strip()
                    ):
                        continue
                    # A printed page number is meaningful only within its
                    # reporter edition. Unknown namespaces remain candidates
                    # for review; a known different edition is excluded.
                    if page.volume is not None and page.volume != locator.volume:
                        continue
                    if page.edition is not None and not _same_literal(locator.edition, page.edition):
                        continue
                    candidates.append(
                        OpinionPageReference(
                            opinion_id=opinion.opinion_id,
                            page_index=page_index,
                            pagination_confirmed=(
                                page.kind == target.kind
                                and page.volume is not None
                                and page.edition is not None
                            ),
                        )
                    )
            pages.append(
                ReporterPageCandidates(
                    target_index=target_index, label=label, kind=target.kind, candidates=tuple(candidates)
                )
            )
    if any(not page.candidates for page in pages):
        outcome = ReporterPageResolutionOutcome.UNLOCATED
        reason = "At least one requested page has no matching source marker; full-document review is needed"
    elif all(len(page.candidates) == 1 and page.candidates[0].pagination_confirmed for page in pages):
        outcome = ReporterPageResolutionOutcome.RESOLVED
        reason = "Each requested page has one source in the cited reporter edition"
    else:
        outcome = ReporterPageResolutionOutcome.AMBIGUOUS
        reason = "Page labels have multiple writings or an unconfirmed reporter namespace"
    # Page selection does not establish proposition support, and does not yet
    # resolve a requested footnote. A range may legitimately cross writings.
    return ReporterCitationPageResolution(**pointers, outcome=outcome, pages=tuple(pages), reason=reason)


def resolve_reporter_citation_pages(document: Document) -> Document:
    """Use each occurrence's locator/pinpoint without changing identity or attachment.

    Root-level retrieval is shared, but the root's canonical pinpoint is never
    imposed on its leaves. An unqualified Id. may inherit the immediately
    preceding citation's pinpoint; intervening noncase authorities break that
    rule. Lead, concurrence, dissent and combined records have no automatic
    priority: pagination selects a source, not an opinion-type preference.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if INDEX_STAGE not in document.stage_runs:
        raise ValueError("Index reporter-root opinion pages before resolving citations")
    roots = {
        root.id: root
        for root in document.roots
        if isinstance(root, FullReporterCitation) and root.reporter_root_opinion_page_index is not None
    }
    # Keep the same noncase interruption semantics as Id. attribution. This is
    # an occurrence chronology, not a search backwards for a convenient root.
    prior_root: str | None = None
    prior_sources: _ReadingSources | None = None
    for _, _, citation in citation_chronology(document):
        root_id = latest(citation.root_id) if citation is not None else None
        if citation is None or root_id not in roots:
            prior_root = prior_sources = None
            continue
        root = roots[root_id]
        locator_source = (
            citation if isinstance(citation, (FullReporterCitation, ShortReporterCitation)) else root
        )
        pin_source = citation if citation.pin_cite else None
        if isinstance(citation, IdCitation) and prior_root == root_id and prior_sources is not None:
            locator_source = prior_sources.locator
            if pin_source is None:
                pin_source = prior_sources.pin
        sources = _ReadingSources(locator_source, pin_source)
        recorded = citation.record(STAGE)
        document = document.replace_citation(
            recorded.with_reporter_page_resolution(_resolve(recorded, root, sources))
        )
        prior_root, prior_sources = root_id, sources
    return document.complete(STAGE)
