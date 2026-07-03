"""Validation pipeline for extracted citations.

Validation is a deterministic existence check against CourtListener: each
extracted citation resolves to one variant of the ``CitationValidation``
discriminated union, and the ``Found`` variant additionally carries a
``CourtResolutionTrace`` describing how the CourtListener-side court was
obtained. Validation only retrieves data; it never compares the resolved
court against the citation court from extraction.

The court resolution work itself lives in
:mod:`mellea_lrc.validation.court_resolution`; this orchestrator only owns the
lookup call and the per-run docket cache that deduplicates GETs across
citations in one document.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.courtlistener.client import CourtListenerClient
from mellea_lrc.courtlistener.remote import CourtListenerAccessClient
from mellea_lrc.courtlistener.types import (
    CitationLookupClient,
    CitationValidationClient,
    CourtListenerCitationLookup,
)
from mellea_lrc.validation.court_resolution import resolve_court
from mellea_lrc.validation.not_found_search import search_case_name_candidates
from mellea_lrc.validation.types import (
    AmbiguousCitationValidation,
    CitationValidation,
    FoundCitationValidation,
    InvalidCitationValidation,
    LookupFailedCitationValidation,
    NotFoundCitationValidation,
    RetrievedCandidate,
    SkippedCitationValidation,
    ThrottledCitationValidation,
    ValidatedDocument,
    ValidationClientMode,
    ValidationMetadata,
    ValidationStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.types import CourtListenerCitationRecord
    from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument

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
    client: CitationValidationClient | None = None,
) -> ValidatedDocument:
    """Run first-layer existence validation plus court resolution for each full case citation.

    The stage is fully deterministic (no LLM). Each citation yields one variant of
    the ``CitationValidation`` discriminated union; ``Found`` citations additionally
    carry a ``CourtResolutionTrace``. The docket-id cache deduplicates docket GETs
    across citations sharing the same docket within one document.
    """
    lookup_client = _lookup_client(client_mode, client)
    docket_court_cache: dict[str, str | None] = {}
    started = time.perf_counter()
    validations = tuple(
        _validate_citation(item, lookup_client, docket_court_cache) for item in extraction.citations
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    return ValidatedDocument(
        source_metadata=extraction.source_metadata,
        text=extraction.text,
        preprocessing_metadata=extraction.preprocessing_metadata,
        citations=extraction.citations,
        extraction_metadata=extraction.extraction_metadata,
        validations=validations,
        validation_metadata=ValidationMetadata(
            client_mode=client_mode,
            source=SOURCE,
            duration_ms=duration_ms,
        ),
    )


def _lookup_client(
    client_mode: str,
    client: CitationValidationClient | None,
) -> CitationValidationClient:
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
    client: CitationValidationClient | None,
) -> None:
    if client is not None:
        msg = f"client must be None when client_mode='{client_mode}'; use client_mode='custom'"
        raise ValueError(msg)


def _validate_citation(
    item: ExtractedCitation,
    client: CitationValidationClient,
    docket_court_cache: dict[str, str | None],
) -> CitationValidation:
    citation = item.citation
    if not isinstance(citation, FullCaseCitation):
        return SkippedCitationValidation(
            citation_id=item.citation_id,
            source=SOURCE,
        )

    if not citation.volume or not citation.reporter or not citation.page:
        return InvalidCitationValidation(
            citation_id=item.citation_id,
            source=SOURCE,
        )

    lookup = client.lookup_citation(citation.volume, citation.reporter, citation.page)
    return _validation_from_lookup(item.citation_id, lookup, client, docket_court_cache, citation)


def _validation_from_lookup(
    citation_id: str,
    lookup: CourtListenerCitationLookup,
    client: CitationValidationClient,
    docket_court_cache: dict[str, str | None],
    citation: FullCaseCitation,
) -> CitationValidation:
    status = _status_from_lookup(lookup.status)
    common = {
        "citation_id": citation_id,
        "locator": lookup.citation,
        "source": SOURCE,
        "lookup_status": lookup.status,
        "lookup_cache": lookup.cache,
        "lookup_key": lookup.key,
        "extra_data": lookup.extra_data,
    }
    if status is ValidationStatus.FOUND:
        record = lookup.records[0] if lookup.records else None
        if record is None:
            # Defensive: a 200 with no record is a malformed lookup; surface as failure.
            return LookupFailedCitationValidation(
                **common,
                error_message="CourtListener returned HTTP 200 but no records.",
                failure_detail=lookup.failure_detail,
            )
        return FoundCitationValidation(
            **common,
            candidate=_retrieved_candidate(
                citation_id,
                0,
                record,
                client=client,
                cache=docket_court_cache,
            ),
        )
    if status is ValidationStatus.AMBIGUOUS:
        return AmbiguousCitationValidation(
            **common,
            candidates=tuple(
                _retrieved_candidate(
                    citation_id,
                    index,
                    record,
                    client=client,
                    cache=docket_court_cache,
                )
                for index, record in enumerate(lookup.records)
            ),
        )
    if status is ValidationStatus.NOT_FOUND:
        return NotFoundCitationValidation(
            **common,
            candidate_search=search_case_name_candidates(citation, client=client),
        )
    if status is ValidationStatus.THROTTLED:
        return ThrottledCitationValidation(
            **common,
            error_message=lookup.error_message,
            failure_detail=lookup.failure_detail,
        )
    return LookupFailedCitationValidation(
        **common,
        error_message=lookup.error_message,
        failure_detail=lookup.failure_detail,
    )


def _retrieved_candidate(
    citation_id: str,
    index: int,
    record: CourtListenerCitationRecord,
    *,
    client: CitationValidationClient,
    cache: dict[str, str | None],
) -> RetrievedCandidate:
    """Attach stable validation identity and provenance to one retrieved record."""
    return RetrievedCandidate(
        candidate_id=f"{citation_id}:candidate:{index}",
        record=record,
        court_resolution=resolve_court(record, client=client, cache=cache),
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


# Re-export the simpler protocol so legacy downstream imports keep working.
__all__ = [
    "CitationLookupClient",
    "run_validation",
]
