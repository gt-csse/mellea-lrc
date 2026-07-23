"""Document-level validation orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.validation.execution import CitationValidationRunner
from mellea_lrc.validation.types import CitationValidation, ValidatedDocument

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import ExtractedDocument


def initialize_validation(document: ExtractedDocument) -> ValidatedDocument:
    """Create one empty validation progression per extracted citation."""
    return ValidatedDocument(
        source=document,
        citations=tuple(CitationValidation(citation=item) for item in document.citations),
    )


async def validate_document(
    document: ExtractedDocument,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> ValidatedDocument:
    """Run each extracted citation through the common validation progression."""
    service = client if client is not None else CourtListenerClient()
    initialized = initialize_validation(document)
    runner = CitationValidationRunner(client=service)
    citations = [
        await runner.run_validation(
            citation,
            document_text=document.text,
            session=session,
        )
        for citation in initialized.citations
    ]
    return ValidatedDocument(source=document, citations=tuple(citations))
