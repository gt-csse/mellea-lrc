"""Validation pipeline for extracted citations."""

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.courtlistener.client import CourtListenerClient
from mellea_lrc.courtlistener.remote import CourtListenerAccessClient
from mellea_lrc.courtlistener.types import CitationLookupClient, CourtListenerCitationLookup
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument
from mellea_lrc.validation.types import (
    CitationValidation,
    ValidatedDocument,
    ValidationClientMode,
    ValidationMetadata,
    ValidationStatus,
)

SOURCE = "cl-access"
DEFAULT_CLIENT_MODE: ValidationClientMode = "deployed"
HTTP_FOUND = 200
HTTP_MULTIPLE_CHOICES = 300
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429


def run_validation(
    extraction: ExtractedDocument,
    *,
    client_mode: ValidationClientMode = DEFAULT_CLIENT_MODE,
    client: CitationLookupClient | None = None,
) -> ValidatedDocument:
    """Run first-layer existence validation for extractable full case citations."""
    lookup_client = _lookup_client(client_mode, client)
    return ValidatedDocument(
        source_metadata=extraction.source_metadata,
        text=extraction.text,
        preprocessing_metadata=extraction.preprocessing_metadata,
        citations=extraction.citations,
        extraction_metadata=extraction.extraction_metadata,
        validations=tuple(_validate_citation(item, lookup_client) for item in extraction.citations),
        validation_metadata=ValidationMetadata(client_mode=client_mode, source=SOURCE),
    )


def _lookup_client(
    client_mode: str,
    client: CitationLookupClient | None,
) -> CitationLookupClient:
    if client_mode == "deployed":
        _ensure_no_client_override(client_mode, client)
        return CourtListenerAccessClient()
    if client_mode == "sdk":
        _ensure_no_client_override(client_mode, client)
        return CourtListenerClient()
    if client_mode == "custom":
        if client is None:
            msg = "client is required when client_mode='custom'"
            raise ValueError(msg)
        return client

    msg = "client_mode must be one of: 'deployed', 'sdk', or 'custom'"
    raise ValueError(msg)


def _ensure_no_client_override(
    client_mode: str,
    client: CitationLookupClient | None,
) -> None:
    if client is not None:
        msg = f"client must be None when client_mode='{client_mode}'; use client_mode='custom'"
        raise ValueError(msg)


def _validate_citation(
    item: ExtractedCitation,
    client: CitationLookupClient,
) -> CitationValidation:
    citation = item.citation
    if not isinstance(citation, FullCaseCitation):
        return CitationValidation(
            citation_id=item.citation_id,
            locator=None,
            status=ValidationStatus.SKIPPED,
            source=SOURCE,
            message="Only FullCaseCitation is validated in the first-layer existence check.",
        )

    if not citation.volume or not citation.reporter or not citation.page:
        return CitationValidation(
            citation_id=item.citation_id,
            locator=None,
            status=ValidationStatus.INVALID,
            source=SOURCE,
            message="Missing volume, reporter, or page.",
        )

    lookup = client.lookup_citation(citation.volume, citation.reporter, citation.page)
    return _validation_from_lookup(item.citation_id, lookup)


def _validation_from_lookup(
    citation_id: str,
    lookup: CourtListenerCitationLookup,
) -> CitationValidation:
    case_names = tuple(item.case_name for item in lookup.matches if item.case_name)
    return CitationValidation(
        citation_id=citation_id,
        locator=lookup.citation,
        status=_status_from_lookup(lookup.status),
        source=SOURCE,
        message=_message_from_lookup(lookup, case_names),
        lookup_status=lookup.status,
        lookup_cache=lookup.cache,
        lookup_key=lookup.key,
        error_message=lookup.error_message,
        failure_detail=lookup.failure_detail,
        matches=lookup.matches,
        extra_data=lookup.extra_data,
    )


def _status_from_lookup(status: int) -> ValidationStatus:
    if status == HTTP_FOUND:
        return ValidationStatus.FOUND
    if status == HTTP_MULTIPLE_CHOICES:
        return ValidationStatus.AMBIGUOUS
    if status == HTTP_NOT_FOUND:
        return ValidationStatus.NOT_FOUND
    if status == HTTP_BAD_REQUEST:
        return ValidationStatus.INVALID
    if status == HTTP_TOO_MANY_REQUESTS:
        return ValidationStatus.THROTTLED
    return ValidationStatus.LOOKUP_FAILED


def _message_from_lookup(
    lookup: CourtListenerCitationLookup,
    case_names: tuple[str, ...],
) -> str:
    suffix = f" - {'; '.join(case_names[:3])}" if case_names else ""
    status = _status_from_lookup(lookup.status)
    if status == ValidationStatus.FOUND:
        return f"CourtListener: found {lookup.citation}{suffix}"
    if status == ValidationStatus.AMBIGUOUS:
        return f"CourtListener: ambiguous ({len(lookup.matches)} matches) {lookup.citation}{suffix}"
    if status == ValidationStatus.NOT_FOUND:
        return f"CourtListener: not found {lookup.citation}"
    if status == ValidationStatus.INVALID:
        return f"CourtListener: invalid citation {lookup.citation}"
    if status == ValidationStatus.THROTTLED:
        return f"CourtListener: lookup throttled {lookup.citation}"
    if lookup.error_message:
        return f"CourtListener: lookup failed {lookup.citation}: {lookup.error_message}"
    return f"CourtListener: lookup status {lookup.status} {lookup.citation}"
