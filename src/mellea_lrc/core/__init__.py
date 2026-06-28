"""Canonical domain models shared across mellea-lrc."""

from mellea_lrc.core.citations import (
    FULL_CITATION_KINDS,
    CanonicalCitation,
    CitationKind,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
    citation_kind,
    is_full_citation,
)
from mellea_lrc.core.documents import DocumentBase, SourceFormat, SourceMetadata
from mellea_lrc.core.spans import Span

__all__ = [
    "FULL_CITATION_KINDS",
    "CanonicalCitation",
    "CitationKind",
    "DocumentBase",
    "FullCaseCitation",
    "FullJournalCitation",
    "FullLawCitation",
    "IdCitation",
    "ReferenceCitation",
    "ShortCaseCitation",
    "SourceFormat",
    "SourceMetadata",
    "Span",
    "SupraCitation",
    "UnknownCitation",
    "citation_kind",
    "is_full_citation",
]
