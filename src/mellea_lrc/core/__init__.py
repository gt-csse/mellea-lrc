"""Canonical domain models shared across mellea-lrc."""

from mellea_lrc.core.citations import (
    CanonicalCitation,
    CitationKind,
    DocketCitation,
    DocketEntry,
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
    "CanonicalCitation",
    "CitationKind",
    "DocketCitation",
    "DocketEntry",
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
