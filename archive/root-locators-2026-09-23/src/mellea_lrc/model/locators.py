"""Pure locator occurrence and colocation projections over citation records."""

from collections.abc import Sequence
from dataclasses import dataclass

from mellea_lrc.model.citations import CitationKind, DocketCitation, FullCaseCitation, citation_kind
from mellea_lrc.model.record import CitationRecord
from mellea_lrc.model.spans import Span

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


def locator_layer(text: str, citations: Sequence[CitationRecord]) -> tuple[Locator, ...]:
    """Return every complete reporter and docket locator occurrence."""
    locators: list[Locator] = []
    for item in citations:
        citation = item.fields
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
    """Return co-located case locator groups as citation-record ids."""
    groups: dict[str, list[CitationRecord]] = {}
    for item in citations:
        if item.colocation_id and isinstance(item.fields, _LOCATOR_TYPES):
            groups.setdefault(item.colocation_id, []).append(item)

    ordered: list[tuple[str, ...]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(
            key=lambda item: (
                item.fields.locator_span.start if item.fields.locator_span is not None else -1,
                item.fields.locator_span.end if item.fields.locator_span is not None else -1,
            )
        )
        ordered.append(tuple(item.citation_id for item in members))
    return tuple(ordered)
