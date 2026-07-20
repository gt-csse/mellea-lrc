"""Document-level validation orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.validation.execution import run_citation_loop
from mellea_lrc.validation.types import CitationValidation, ValidatedDocument

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import ExtractedDocument


def initialize_validation(document: ExtractedDocument) -> ValidatedDocument:
    """Create one empty validation progression per extracted citation."""
    return ValidatedDocument(
        source=document,
        citations=tuple(CitationValidation(citation=item) for item in document.citations),
    )


def validate_document(
    document: ExtractedDocument,
    *,
    client: CourtListenerServiceClient | None = None,
) -> ValidatedDocument:
    """Run each extracted citation through the configured validation loop."""
    service = client if client is not None else CourtListenerClient()
    initialized = initialize_validation(document)
    citations = tuple(run_citation_loop(item, client=service) for item in initialized.citations)
    return ValidatedDocument(source=document, citations=citations)
