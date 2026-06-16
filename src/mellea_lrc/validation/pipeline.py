"""Validation pipeline for extracted citations."""

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction.types import DocumentExtraction, ExtractedCitation
from mellea_lrc.validation.cl_access import (
    CourtListenerAccessClient,
    CourtListenerCitationLookup,
)
from mellea_lrc.validation.types import (
    CitationValidation,
    DocumentValidation,
    ValidationStatus,
)

SOURCE = "cl-access"
HTTP_FOUND = 200
HTTP_MULTIPLE_CHOICES = 300
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429


def validate_extraction(
    extraction: DocumentExtraction,
    *,
    client: CourtListenerAccessClient | None = None,
) -> DocumentValidation:
    """Run first-layer existence validation for extractable full case citations."""
    lookup_client = client or CourtListenerAccessClient()
    return DocumentValidation(
        validations=tuple(
            _validate_citation(item, lookup_client) for item in extraction.citations
        )
    )


def _validate_citation(
    item: ExtractedCitation,
    client: CourtListenerAccessClient,
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
    case_names = tuple(
        case_name
        for cluster in lookup.clusters
        if isinstance((case_name := cluster.get("case_name")), str)
    )
    return CitationValidation(
        citation_id=citation_id,
        locator=lookup.citation,
        status=_status_from_lookup(lookup.status),
        source=SOURCE,
        message=_message_from_lookup(lookup, case_names),
        case_names=case_names,
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
        return f"CourtListener: ambiguous ({len(lookup.clusters)} matches) {lookup.citation}{suffix}"
    if status == ValidationStatus.NOT_FOUND:
        return f"CourtListener: not found {lookup.citation}"
    if status == ValidationStatus.INVALID:
        return f"CourtListener: invalid citation {lookup.citation}"
    if status == ValidationStatus.THROTTLED:
        return f"CourtListener: lookup throttled {lookup.citation}"
    if lookup.error_message:
        return f"CourtListener: lookup failed {lookup.citation}: {lookup.error_message}"
    return f"CourtListener: lookup status {lookup.status} {lookup.citation}"
