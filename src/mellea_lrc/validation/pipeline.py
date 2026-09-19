"""The serializable full-reporter-locator identity checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.validation.execution import CitationValidationRunner
from mellea_lrc.validation.root_context import masked_root_context
from mellea_lrc.validation.types import CitationValidation, ValidatedDocument

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import Document


def initialize_full_reporter_locator_identity(document: Document) -> ValidatedDocument:
    """Create the checkpoint that later validation stages resume.

    Every active citation remains addressable in source order. This checkpoint
    creates no validation node for a docket or any other locator kind; a later
    docket or search stage therefore starts from the same serialized evidence
    without interpreting an earlier generic deferral.
    """
    return ValidatedDocument(
        source=document,
        citations=tuple(CitationValidation(citation=item) for item in document.active_citations),
    )


async def run_full_reporter_locator_identity(
    checkpoint: ValidatedDocument,
    *,
    client: CourtListenerServiceClient | None = None,
    session: MelleaSession | None = None,
) -> ValidatedDocument:
    """Resolve only full reporter locators, returning the same checkpoint type.

    Docket locators, leaves, statutes, and journals are left entirely untouched.
    A resolved or explicitly deferred reporter progression is also left as-is,
    making a completed serialized checkpoint safe to pass directly into its
    next stage. A partial progression is rejected rather than replayed, because
    replaying a lookup would create a second evidence node for one operation.

    The stage intentionally does not consume co-location. A future
    cross-locator reconciliation stage may use stored colocation and both
    independent identity results after docket lookup has its own contract.
    """
    service = client if client is not None else CourtListenerClient()
    runner = CitationValidationRunner(client=service)
    progressions: list[CitationValidation] = []
    for progression in checkpoint.citations:
        if not isinstance(progression.citation.stated, FullCaseCitation):
            progressions.append(progression)
            continue
        if progression.identity_resolution is not None:
            progressions.append(progression)
            continue
        if progression.nodes:
            msg = (
                f"Cannot resume partial full reporter-locator identity progression for "
                f"{progression.citation_id!r}; restart from its pre-identity checkpoint."
            )
            raise ValueError(msg)
        progressions.append(
            await runner.run_full_reporter_locator_identity(
                progression,
                document_text=masked_root_context(checkpoint.source, progression.citation).as_document_text(
                    document_length=len(checkpoint.source.text)
                ),
                session=session,
            )
        )
    return ValidatedDocument(source=checkpoint.source, citations=tuple(progressions))
