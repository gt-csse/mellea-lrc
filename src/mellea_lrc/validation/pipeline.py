"""Document-level validation orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.validation.execution import CitationValidationRunner
from mellea_lrc.validation.root_context import masked_root_context
from mellea_lrc.validation.types import CitationValidation, ValidatedDocument

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import Document


def initialize_validation(document: Document) -> ValidatedDocument:
    """Create one validation progression per active citation, preserving the source."""
    return ValidatedDocument(
        source=document,
        citations=tuple(CitationValidation(citation=item) for item in document.active_citations),
    )


async def validate_document(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> ValidatedDocument:
    """Run the currently admitted reporter-locator identity checkpoint.

    This is intentionally the same safe scope as ``validate_document_identity``
    until docket lookup, lookup misses, and the downstream opinion stages have
    their own settled contracts.
    """
    return await validate_document_identity(document, client=client, session=session)


async def validate_document_identity(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> ValidatedDocument:
    """Resume a locator checkpoint through identity, before pinpoint checking.

    Locator checkpoints currently contain only complete reporter and docket
    occurrences, so every active record is intentionally included.  This is a
    bounded evaluation entrypoint, not the final document-native handback API.
    """
    service = client if client is not None else CourtListenerClient()
    initialized = initialize_validation(document)
    runner = CitationValidationRunner(client=service)
    citations = [
        await runner.run_identity_validation(
            citation,
            document_text=masked_root_context(document, citation.citation).as_document_text(
                document_length=len(document.text)
            ),
            session=session,
        )
        for citation in initialized.citations
    ]
    return ValidatedDocument(source=document, citations=tuple(citations))
