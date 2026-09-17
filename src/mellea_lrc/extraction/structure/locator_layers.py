"""Project extraction records into the locator and colocation layers.

The layers answer separate questions. A locator is one occurrence of a complete
reporter or docket identifier in the document. Repeated identifiers retain
their own spans regardless of which root they later share. A colocation is a
group of citation-record ids whose locators occupy the same citation site.
Singletons remain locators and do not become one-member colocations.

These are projections over the citation records, not a replacement for them:
the locator layer preserves the exact text span, while the colocation layer
preserves which records were grouped. Keeping both explicit makes it possible
to evaluate locator reading and grouping independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import CitationKind, DocketCitation, FullCaseCitation, citation_kind
from mellea_lrc.core.spans import Span

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.extraction.rules import ExtractionRules
    from mellea_lrc.extraction.types import Document
    from mellea_lrc.preprocessing.types import PreprocessedDocument

_LOCATOR_TYPES = (FullCaseCitation, DocketCitation)


@dataclass(frozen=True, slots=True)
class Locator:
    """One complete case locator occurrence and the span that states it."""

    citation_id: str
    kind: CitationKind
    span: Span
    text: str


@dataclass(frozen=True, slots=True)
class LocatorLayers:
    """Separate outputs from locator reading: occurrence spans and groups of ids."""

    locators: tuple[Locator, ...]
    colocations: tuple[tuple[str, ...], ...]


def find_locators(
    document: PreprocessedDocument | Document,
    rules: ExtractionRules | None = None,
) -> LocatorLayers:
    """Run root extraction and return its two explicit locator-layer outputs.

    Every complete locator occurrence appears in locators, including repeated
    identifiers. A singleton has no colocation group. Supplying
    no rules selects eyecite defaults; pass stable(rules) for the project's
    docket and grouping rules.
    """
    from mellea_lrc.extraction.eyecite_extractor import grow_roots
    from mellea_lrc.preprocessing.types import PreprocessedDocument

    if isinstance(document, PreprocessedDocument):
        preprocessed = document
    else:
        preprocessed = PreprocessedDocument(
            source_metadata=document.source_metadata,
            text=document.text,
            preprocessing_metadata=document.preprocessing_metadata,
        )
    extracted = grow_roots(preprocessed, rules=rules)
    return LocatorLayers(
        locators=extracted.locators,
        colocations=extracted.colocations,
    )


def locator_layer(text: str, citations: Sequence[CitationRecord]) -> tuple[Locator, ...]:
    """Return every complete reporter and docket locator occurrence.

    Root assignment and deduplication do not affect this layer. An identifier
    written twice contributes two locators with distinct occurrence spans.
    """
    locators: list[Locator] = []
    for item in citations:
        citation = item.source
        span = citation.locator_span
        if not isinstance(citation, _LOCATOR_TYPES) or span is None:
            continue
        locators.append(
            Locator(
                citation_id=item.citation_id,
                kind=citation_kind(citation),
                span=span,
                text=text[span.start : span.end],
            )
        )
    return tuple(sorted(locators, key=lambda locator: (locator.span.start, locator.span.end)))


def colocation_layer(citations: Sequence[CitationRecord]) -> tuple[tuple[str, ...], ...]:
    """Return co-located case locator groups as citation-record ids.

    Group members are occurrences, not only the first occurrence of each
    citation-tree root. That keeps repeated parallel citations visible. A
    singleton has no group and remains distinguishable from a colocation.
    """
    groups: dict[str, list[CitationRecord]] = {}
    for item in citations:
        if item.colocation_id and isinstance(item.source, _LOCATOR_TYPES):
            groups.setdefault(item.colocation_id, []).append(item)

    ordered: list[tuple[str, ...]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(
            key=lambda item: (
                item.source.locator_span.start if item.source.locator_span is not None else -1,
                item.source.locator_span.end if item.source.locator_span is not None else -1,
            )
        )
        ordered.append(tuple(item.citation_id for item in members))
    return tuple(ordered)
