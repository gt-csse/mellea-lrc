"""Eyecite-backed citation extraction into canonical core representations."""

from __future__ import annotations

import uuid
from typing import cast

from eyecite import get_citations, resolve_citations
from eyecite.models import (
    CitationBase,
    FullCaseCitation as EyeciteFullCaseCitation,
    FullJournalCitation as EyeciteFullJournalCitation,
    FullLawCitation as EyeciteFullLawCitation,
    IdCitation as EyeciteIdCitation,
    ReferenceCitation as EyeciteReferenceCitation,
    Resource,
    ShortCaseCitation as EyeciteShortCaseCitation,
    SupraCitation as EyeciteSupraCitation,
    UnknownCitation as EyeciteUnknownCitation,
)

from mellea_lrc.core.citations import (
    CanonicalCitation,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
)
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing.plain_text import preprocess_plain_text_from_string
from mellea_lrc.preprocessing.types import PreprocessedDocument  # noqa: TC001

EYECITE_CITATION_TYPES = frozenset(
    {
        EyeciteFullCaseCitation,
        EyeciteFullLawCitation,
        EyeciteFullJournalCitation,
        EyeciteShortCaseCitation,
        EyeciteSupraCitation,
        EyeciteIdCitation,
        EyeciteReferenceCitation,
        EyeciteUnknownCitation,
    }
)


def _to_full_case(citation: EyeciteFullCaseCitation) -> FullCaseCitation:
    return FullCaseCitation(
        plaintiff=citation.metadata.plaintiff,
        defendant=citation.metadata.defendant,
        volume=citation.groups.get("volume"),
        reporter=citation.groups.get("reporter"),
        page=citation.groups.get("page"),
        pin_cite=citation.metadata.pin_cite,
        extra=citation.metadata.extra,
        year=citation.metadata.year,
        court=citation.metadata.court,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_full_law(citation: EyeciteFullLawCitation) -> FullLawCitation:
    return FullLawCitation(
        volume=citation.groups.get("title"),
        reporter=citation.groups.get("reporter"),
        page=citation.groups.get("section"),
        pin_cite=citation.metadata.pin_cite,
        year=citation.metadata.year,
        publisher=citation.metadata.publisher,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_full_journal(citation: EyeciteFullJournalCitation) -> FullJournalCitation:
    return FullJournalCitation(
        volume=citation.groups.get("volume"),
        reporter=citation.groups.get("reporter"),
        page=citation.groups.get("page"),
        pin_cite=citation.metadata.pin_cite,
        year=citation.metadata.year,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_short_case(citation: EyeciteShortCaseCitation) -> ShortCaseCitation:
    return ShortCaseCitation(
        volume=citation.groups.get("volume"),
        reporter=citation.groups.get("reporter"),
        page=citation.groups.get("page"),
        pin_cite=citation.metadata.pin_cite,
        court=citation.metadata.court,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_supra(citation: EyeciteSupraCitation) -> SupraCitation:
    return SupraCitation(
        pin_cite=citation.metadata.pin_cite,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_id(citation: EyeciteIdCitation) -> IdCitation:
    return IdCitation(
        pin_cite=citation.metadata.pin_cite,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_reference(citation: EyeciteReferenceCitation) -> ReferenceCitation:
    return ReferenceCitation(
        plaintiff=citation.metadata.plaintiff,
        defendant=citation.metadata.defendant,
    )


def _to_unknown(_citation: EyeciteUnknownCitation) -> UnknownCitation:
    return UnknownCitation()


def _to_canonical(citation: CitationBase) -> CanonicalCitation:
    if isinstance(citation, EyeciteFullCaseCitation):
        return _to_full_case(citation)
    if isinstance(citation, EyeciteFullLawCitation):
        return _to_full_law(citation)
    if isinstance(citation, EyeciteFullJournalCitation):
        return _to_full_journal(citation)
    if isinstance(citation, EyeciteShortCaseCitation):
        return _to_short_case(citation)
    if isinstance(citation, EyeciteSupraCitation):
        return _to_supra(citation)
    if isinstance(citation, EyeciteIdCitation):
        return _to_id(citation)
    if isinstance(citation, EyeciteReferenceCitation):
        return _to_reference(citation)
    if isinstance(citation, EyeciteUnknownCitation):
        return _to_unknown(citation)
    msg = f"Unknown citation type: {type(citation).__name__}"
    raise TypeError(msg)


def _assign_citation_ids(
    citations: list[CitationBase],
) -> list[tuple[CitationBase, str]]:
    citation_ids: list[tuple[CitationBase, str]] = []
    for citation in citations:
        if type(citation) not in EYECITE_CITATION_TYPES:
            msg = (
                f"Unknown citation type: {type(citation).__name__}. "
                "All citation types must be handled explicitly."
            )
            raise ValueError(msg)
        citation_ids.append((citation, str(uuid.uuid4())[:8]))
    return citation_ids


def _build_antecedent_map(
    resolutions: dict[Resource, list[CitationBase]],
    citation_ids: list[tuple[CitationBase, str]],
) -> dict[str, str]:
    """Map reference citation ids to their resolved full citation id."""
    citation_to_id = {id(citation): citation_id for citation, citation_id in citation_ids}
    antecedent_map: dict[str, str] = {}
    for grouped in resolutions.values():
        full_citation_id = citation_to_id[id(grouped[0])]
        for reference in grouped[1:]:
            reference_id = citation_to_id[id(reference)]
            antecedent_map[reference_id] = full_citation_id
    return antecedent_map


def _extract_from_text(
    preprocessed: PreprocessedDocument,
) -> ExtractedDocument:
    """Extract canonical citations from a preprocessed document."""
    text = preprocessed.text
    eyecite_citations = get_citations(text)
    resolutions = cast(
        dict[Resource, list[CitationBase]],
        resolve_citations(eyecite_citations),
    )
    citation_ids = _assign_citation_ids(eyecite_citations)
    antecedent_map = _build_antecedent_map(resolutions, citation_ids)

    extracted: list[ExtractedCitation] = []
    for eyecite_citation, citation_id in citation_ids:
        span_start, span_end = eyecite_citation.full_span()
        locator_start, locator_end = eyecite_citation.span()
        extracted.append(
            ExtractedCitation(
                citation_id=citation_id,
                span=Span(start=span_start, end=span_end),
                locator_span=Span(start=locator_start, end=locator_end),
                matched_text=eyecite_citation.matched_text(),
                citation=_to_canonical(eyecite_citation),
                resolves_to=antecedent_map.get(citation_id),
            )
        )

    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=tuple(extracted),
        extraction_metadata=ExtractionMetadata(),
    )


def extract_baseline(preprocessed: PreprocessedDocument) -> ExtractedDocument:
    """Extract canonical citations using eyecite as the baseline engine."""
    return _extract_from_text(preprocessed)


def extract(preprocessed: PreprocessedDocument) -> ExtractedDocument:
    """Extract canonical citations from a preprocessed document.

    Alias for :func:`extract_baseline`. Prefer :func:`run_extraction` for the
    layer-level pipeline entrypoint.
    """
    return extract_baseline(preprocessed)


def extract_citations(text: str, *, source_path: str | None = None) -> ExtractedDocument:
    """Extract citations from raw Layer 2 text."""
    preprocessed = preprocess_plain_text_from_string(text, source_path=source_path)
    return extract_baseline(preprocessed)
