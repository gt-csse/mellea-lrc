"""The extraction arms this project can be scored as.

Each arm is a small function from ``(document, text)`` to public ``Occurrence``
objects, spelled out rather than composed by a framework so that reading one
tells you exactly what ran, in order.

The experimental arms have no domain-object form -- an ``AdjudicatedLocator``
is not an ``ExtractedCitation`` -- which is why they emit public occurrences
directly rather than a serialized ``ExtractedDocument``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass

from eyecite import get_citations
from eyecite.models import CitationBase

from evaluations.extraction.occurrences import Occurrence, deduplicate
from mellea_lrc.experimental.grounded_adjudication import (
    adjudicate_docket,
    adjudicate_site,
    suspected_dockets,
    suspected_sites,
)
from mellea_lrc.experimental.relaxed_eyecite_extractor import extract_relaxed_citations
from mellea_lrc.extraction import extract_citations
from mellea_lrc.extraction.types import ExtractedDocument

Arm = Callable[[str, str], list[Occurrence]]

_FULL_CASE = "FullCaseCitation"


def _adjudicated(call: Coroutine):
    """Run one model call to completion.

    The adjudication layer is async because Mellea's session API is. Nothing
    here benefits from that -- every site is judged in turn -- so the
    concurrency stops at this line and the arms stay ordinary functions.
    """
    return asyncio.run(call)


def _from_extracted_document(document: str, extracted: ExtractedDocument) -> list[Occurrence]:
    """Take the full case citations out of an ExtractedDocument.

    The span used is ``locator_span``, the minimum sufficient case identifier,
    not the citation's full extent. Short forms, ``id.``, ``supra`` and statutes
    are out of the benchmark's scope and would score as false positives.
    """
    return [
        Occurrence(
            document=document,
            start=citation.locator_span.start,
            end=citation.locator_span.end,
            matched_text=extracted.text[citation.locator_span.start : citation.locator_span.end],
            detail={
                "volume": citation.citation.volume,
                "reporter": citation.citation.reporter,
                "page": citation.citation.page,
            },
        )
        for citation in extracted.citations
        if type(citation.citation).__name__ == _FULL_CASE
    ]


def _from_eyecite(document: str, text: str, citations: list[CitationBase]) -> list[Occurrence]:
    """Take the full case citations straight out of an eyecite result."""
    occurrences = []
    for citation in citations:
        if type(citation).__name__ != _FULL_CASE:
            continue
        start, end = citation.span()
        groups = citation.groups
        occurrences.append(
            Occurrence(
                document=document,
                start=start,
                end=end,
                matched_text=text[start:end],
                detail={
                    "volume": groups.get("volume"),
                    "reporter": groups.get("reporter"),
                    "page": groups.get("page"),
                },
            )
        )
    return occurrences


def eyecite_as_published(document: str, text: str) -> list[Occurrence]:
    """Eyecite alone, with nothing done to the text first."""
    return _from_eyecite(document, text, get_citations(text))


def production(document: str, text: str) -> list[Occurrence]:
    """Eyecite with whitespace repair. What Mellea-LRC ships.

    Eyecite's generated patterns join volume, reporter and page with a literal
    single space, so one doubled space -- which PDF extraction leaves behind
    routinely -- makes a citation vanish outright rather than parse imperfectly.
    Collapsing those runs before parsing and mapping the spans back costs
    nothing and moves no offsets.
    """
    return _from_extracted_document(document, extract_citations(text))


def _recover_locators(document: str, extracted: ExtractedDocument) -> list[Occurrence]:
    """Hunt for missed locators and let a model adjudicate each site.

    Sites are found against a copy of the text with every extracted citation
    blanked out, so this only looks where the extractor came up empty. A site is
    a candidate, not a finding: most are rejected, and anything reported is
    grounded back to the source before it is believed.
    """
    recovered = []
    for site in suspected_sites(extracted):
        for locator in _adjudicated(adjudicate_site(extracted.text, site)):
            recovered.append(
                Occurrence(
                    document=document,
                    start=locator.span.start,
                    end=locator.span.end,
                    matched_text=locator.text,
                    detail={
                        "volume": locator.volume,
                        "reporter": locator.reporter,
                        "page": locator.page,
                        "match_method": locator.match_method,
                        "stage": "llm_site_recovery",
                    },
                )
            )
    return recovered


def _recover_dockets(document: str, extracted: ExtractedDocument) -> list[Occurrence]:
    """Find docket-shaped strings and let a model confirm each one.

    A docket number names a case only with its court, so the courts written near
    the site are resolved against courts-db and offered as a closed set. The
    model picks one or declines; it cannot invent a court.
    """
    recovered = []
    for site in suspected_dockets(extracted):
        docket = _adjudicated(adjudicate_docket(extracted.text, site))
        if docket is None:
            continue
        recovered.append(
            Occurrence(
                document=document,
                start=docket.docket_span.start,
                end=docket.docket_span.end,
                matched_text=docket.docket_text,
                detail={
                    "court": docket.court_text,
                    "court_id": docket.court_id,
                    "court_span": (
                        {"start": docket.court_span.start, "end": docket.court_span.end}
                        if docket.court_span is not None
                        else None
                    ),
                    "stage": "llm_docket_recovery",
                },
            )
        )
    return recovered


def production_with_recovery(document: str, text: str) -> list[Occurrence]:
    """Production, then model recovery of the locators and dockets it missed."""
    extracted = extract_citations(text)
    return deduplicate(
        [
            *_from_extracted_document(document, extracted),
            *_recover_locators(document, extracted),
            *_recover_dockets(document, extracted),
        ]
    )


def layout_tolerant_with_recovery(document: str, text: str) -> list[Occurrence]:
    """The layout-tolerant tokenizer, then model recovery of what it still missed."""
    extracted = extract_relaxed_citations(text)
    return deduplicate(
        [
            *_from_extracted_document(document, extracted),
            *_recover_locators(document, extracted),
            *_recover_dockets(document, extracted),
        ]
    )


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One arm and the components it runs."""

    run: Arm
    components: str
    needs_model: bool = False


ARMS: dict[str, ArmSpec] = {
    "eyecite": ArmSpec(run=eyecite_as_published, components="eyecite as published"),
    "production": ArmSpec(run=production, components="eyecite + whitespace repair"),
    "production+recovery": ArmSpec(
        run=production_with_recovery,
        components="production + site hunting + model adjudication",
        needs_model=True,
    ),
    "layout-tolerant+recovery": ArmSpec(
        run=layout_tolerant_with_recovery,
        components="layout-tolerant tokenizer + site hunting + model adjudication",
        needs_model=True,
    ),
}
