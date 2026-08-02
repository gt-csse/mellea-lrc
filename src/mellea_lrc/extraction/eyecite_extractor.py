"""Eyecite-backed citation extraction into canonical core representations."""

from __future__ import annotations

import re
import uuid
from bisect import bisect_left, bisect_right
from typing import cast

from eyecite import get_citations, resolve_citations
from eyecite.annotate import SpanUpdater
from eyecite.models import (
    CitationBase,
    Resource,
)
from eyecite.models import (
    FullCaseCitation as EyeciteFullCaseCitation,
)
from eyecite.models import (
    FullJournalCitation as EyeciteFullJournalCitation,
)
from eyecite.models import (
    FullLawCitation as EyeciteFullLawCitation,
)
from eyecite.models import (
    IdCitation as EyeciteIdCitation,
)
from eyecite.models import (
    ReferenceCitation as EyeciteReferenceCitation,
)
from eyecite.models import (
    ShortCaseCitation as EyeciteShortCaseCitation,
)
from eyecite.models import (
    SupraCitation as EyeciteSupraCitation,
)
from eyecite.models import (
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
from mellea_lrc.preprocessing.types import PreprocessedDocument

_REPEATED_INLINE_WHITESPACE = re.compile(r"[ \t]{2,}")


def _get_citations_with_recovered_spans(text: str) -> list[CitationBase]:
    """Extract citations from whitespace-collapsed text, then remap spans back to `text`.

    Docling PDF extraction leaves runs of repeated spaces/tabs (e.g. from
    justified-text columns) that break eyecite's tokenizer and silently drop
    otherwise well-formed citations. Collapsing those runs before extraction
    recovers them; `SpanUpdater` then maps each citation's offsets from the
    collapsed text back to `text` so downstream span-based text slicing is
    unaffected.
    """
    cleaned = _REPEATED_INLINE_WHITESPACE.sub(" ", text)
    citations = get_citations(cleaned)
    if cleaned == text:
        return citations

    # Collapsing repeated whitespace is a net insertion going cleaned -> text
    # (each run gains its extra characters back), the opposite of eyecite's
    # own plain/markup mapping (a net deletion). bisect_right on the start and
    # bisect_left on the end is what lands exactly on citation boundaries for
    # an insertion-direction diff; the reverse pairing off-by-ones by one
    # collapsed space at segment boundaries.
    updater = SpanUpdater(cleaned, text)
    for citation in citations:
        start, end = citation.span()
        citation.span_start = updater.update(start, bisect_right)
        citation.span_end = updater.update(end, bisect_left)
        if citation.full_span_start is not None:
            citation.full_span_start = updater.update(citation.full_span_start, bisect_right)
        if citation.full_span_end is not None:
            citation.full_span_end = updater.update(citation.full_span_end, bisect_left)
    return citations


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
    eyecite_citations = _get_citations_with_recovered_spans(text)
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
