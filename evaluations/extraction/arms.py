"""The extraction arms this project can be scored as."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from eyecite import get_citations
from eyecite.models import CitationBase

from evaluations.extraction.occurrences import Occurrence
from mellea_lrc.extraction import extract_citations
from mellea_lrc.extraction.types import ExtractedDocument

Arm = Callable[[str, str], list[Occurrence]]

_FULL_CASE = "FullCaseCitation"


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


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One arm and the components it runs."""

    run: Arm
    components: str


ARMS: dict[str, ArmSpec] = {
    "eyecite": ArmSpec(run=eyecite_as_published, components="eyecite as published"),
    "production": ArmSpec(run=production, components="eyecite + whitespace repair"),
}
