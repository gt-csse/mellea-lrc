"""Document-level validation orchestration."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup
from mellea_lrc.validation.execution import ValidationOperation, run_citation_loop
from mellea_lrc.validation.model import (
    CitationValidation,
    ValidationDocument,
    ValidationNode,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import ExtractedDocument


def initialize_validation(document: ExtractedDocument) -> ValidationDocument:
    """Create one empty validation progression per extracted citation."""
    return ValidationDocument(
        source=document,
        citations=tuple(CitationValidation(citation=item) for item in document.citations),
    )


def validate_document(
    document: ExtractedDocument,
    *,
    client: CourtListenerServiceClient | None = None,
) -> ValidationDocument:
    """Run each extracted citation through the configured validation loop."""
    service = client if client is not None else CourtListenerClient()
    initialized = initialize_validation(document)
    first_operation = partial(run_exact_locator_lookup, client=service)
    citations = tuple(
        run_citation_loop(
            item,
            initial_operations=(first_operation,),
            route=_next_operations,
        )
        for item in initialized.citations
    )
    return ValidationDocument(source=document, citations=citations)


def _next_operations(_node: ValidationNode) -> tuple[ValidationOperation, ...]:
    """Return no follow-up operations until another branch is implemented."""
    return ()
