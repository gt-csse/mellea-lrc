"""Retrieval pipeline for extracted citations.

Retrieval is a deterministic existence check against CourtListener: each
extracted citation resolves to one variant of the ``CitationRetrieval``
discriminated union, and the ``Found`` variant additionally carries a
``CourtResolutionTrace`` describing how the CourtListener-side court was
obtained. Retrieval only retrieves data; it never compares the resolved
court against the citation court from extraction.

The court resolution work itself lives in
:mod:`mellea_lrc.retrieval.court_resolution`; this orchestrator only owns the
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
    CitationRetrievalClient,
    CourtListenerCitationLookup,
)
from mellea_lrc.retrieval.court_resolution import resolve_court
from mellea_lrc.retrieval.not_found_search import search_case_name_candidates
from mellea_lrc.jurisdiction_inference.pipeline import infer_jurisdiction
from mellea_lrc.jurisdiction_inference.types import InferredDocument
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.retrieval.case_name_prepare import prepare_case_name_for_search
from mellea_lrc.retrieval.types import (
    AmbiguousCitationRetrieval,
    CaseNameSearchPreparation,
    CitationRetrieval,
    CourtListenerRequestTrace,
    FoundCitationRetrieval,
    InvalidCitationRetrieval,
    LookupFailedCitationRetrieval,
    NotFoundCitationRetrieval,
    RetrievedCandidate,
    SkippedCitationRetrieval,
    ThrottledCitationRetrieval,
    RetrievedDocument,
    RetrievalClientMode,
    RetrievalMetadata,
    RetrievalStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.types import CourtListenerCitationRecord
    from mellea_lrc.extraction.types import ExtractedCitation

SOURCE = "cl-access"
DEFAULT_CLIENT_MODE: RetrievalClientMode = "deployed"
HTTP_FOUND = 200
HTTP_MULTIPLE_CHOICES = 300
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429


def run_retrieval(
    document: InferredDocument,
    *,
    client_mode: RetrievalClientMode = DEFAULT_CLIENT_MODE,
    client: CitationRetrievalClient | None = None,
) -> RetrievedDocument:
    """Run first-layer existence retrieval plus court resolution for each full case citation.

    The stage is fully deterministic (no LLM). Each citation yields one variant of
    the ``CitationRetrieval`` discriminated union; ``Found`` citations additionally
    carry a ``CourtResolutionTrace``. The docket-id cache deduplicates docket GETs
    across citations sharing the same docket within one document.

    If the input document does not carry jurisdiction jurisdictions (i.e. it is an
    ``ExtractedDocument`` rather than an ``InferredDocument``), inference is run
    automatically as a fast local step before retrieval.
    """
    if not isinstance(document, InferredDocument):
        document = infer_jurisdiction(document)
    lookup_client = _lookup_client(client_mode, client)
    docket_court_cache: dict[str, str | None] = {}
    started = time.perf_counter()
    retrievals = tuple(
        _retrieve_citation(item, lookup_client, docket_court_cache) for item in document.citations
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    return RetrievedDocument(
        source_metadata=document.source_metadata,
        text=document.text,
        preprocessing_metadata=document.preprocessing_metadata,
        citations=document.citations,
        extraction_metadata=document.extraction_metadata,
        jurisdictions=document.jurisdictions,
        retrievals=retrievals,
        retrieval_metadata=RetrievalMetadata(
            client_mode=client_mode,
            source=SOURCE,
            duration_ms=duration_ms,
        ),
    )


async def run_retrieval_async(
    document: InferredDocument,
    *,
    client_mode: RetrievalClientMode = DEFAULT_CLIENT_MODE,
    client: CitationRetrievalClient | None = None,
    session: MelleaSession | None = None,
) -> RetrievedDocument:
    """Run retrieval with mandatory LLM case-name preparation for not-found search."""
    if not isinstance(document, InferredDocument):
        document = infer_jurisdiction(document)
    lookup_client = _lookup_client(client_mode, client)
    session = session or start_mellea_session_from_env()
    docket_court_cache: dict[str, str | None] = {}
    started = time.perf_counter()
    retrievals = [
        await _retrieve_citation_async(
            item,
            document.text,
            lookup_client,
            docket_court_cache,
            session,
        )
        for item in document.citations
    ]
    duration_ms = (time.perf_counter() - started) * 1000.0
    return RetrievedDocument(
        source_metadata=document.source_metadata,
        text=document.text,
        preprocessing_metadata=document.preprocessing_metadata,
        citations=document.citations,
        extraction_metadata=document.extraction_metadata,
        jurisdictions=document.jurisdictions,
        retrievals=tuple(retrievals),
        retrieval_metadata=RetrievalMetadata(
            client_mode=client_mode,
            source=SOURCE,
            duration_ms=duration_ms,
        ),
    )


def _lookup_client(
    client_mode: str,
    client: CitationRetrievalClient | None,
) -> CitationRetrievalClient:
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
    client: CitationRetrievalClient | None,
) -> None:
    if client is not None:
        msg = f"client must be None when client_mode='{client_mode}'; use client_mode='custom'"
        raise ValueError(msg)


def _retrieve_citation(
    item: ExtractedCitation,
    client: CitationRetrievalClient,
    docket_court_cache: dict[str, str | None],
) -> CitationRetrieval:
    citation = item.citation
    if not isinstance(citation, FullCaseCitation):
        return SkippedCitationRetrieval(
            citation_id=item.citation_id,
            source=SOURCE,
        )

    edition = citation.reporter.edition_short_name if citation.reporter else None
    if not citation.volume or not edition or not citation.page:
        return InvalidCitationRetrieval(
            citation_id=item.citation_id,
            source=SOURCE,
        )

    lookup = client.lookup_citation(citation.volume, edition, citation.page)
    return _retrieval_from_lookup(item.citation_id, lookup, client, docket_court_cache, citation)


async def _retrieve_citation_async(
    item: ExtractedCitation,
    document_text: str,
    client: CitationRetrievalClient,
    docket_court_cache: dict[str, str | None],
    session: MelleaSession,
) -> CitationRetrieval:
    citation = item.citation
    if not isinstance(citation, FullCaseCitation):
        return SkippedCitationRetrieval(
            citation_id=item.citation_id,
            source=SOURCE,
        )

    edition = citation.reporter.edition_short_name if citation.reporter else None
    if not citation.volume or not edition or not citation.page:
        return InvalidCitationRetrieval(
            citation_id=item.citation_id,
            source=SOURCE,
        )

    lookup = client.lookup_citation(citation.volume, edition, citation.page)
    status = _status_from_lookup(lookup.status)
    if status is not RetrievalStatus.NOT_FOUND:
        return _retrieval_from_lookup(item.citation_id, lookup, client, docket_court_cache, citation)
    preparation = await prepare_case_name_for_search(
        session,
        document_text=document_text,
        citation=item,
    )
    return _retrieval_from_lookup(
        item.citation_id,
        lookup,
        client,
        docket_court_cache,
        citation,
        preparation=preparation,
    )


def _retrieval_from_lookup(
    citation_id: str,
    lookup: CourtListenerCitationLookup,
    client: CitationRetrievalClient,
    docket_court_cache: dict[str, str | None],
    citation: FullCaseCitation,
    *,
    preparation: CaseNameSearchPreparation | None = None,
) -> CitationRetrieval:
    status = _status_from_lookup(lookup.status)
    common = {
        "citation_id": citation_id,
        "locator": lookup.citation,
        "source": SOURCE,
        "request_trace": CourtListenerRequestTrace(
            http_status=lookup.status,
            cache=lookup.cache,
            key=lookup.key,
            error_message=lookup.error_message,
        ),
        "extra_data": lookup.extra_data,
    }
    if status is RetrievalStatus.FOUND:
        record = lookup.records[0] if lookup.records else None
        if record is None:
            # Defensive: a 200 with no record is a malformed lookup; surface as failure.
            return LookupFailedCitationRetrieval(
                citation_id=citation_id,
                locator=lookup.citation,
                source=SOURCE,
                request_trace=CourtListenerRequestTrace(
                    http_status=lookup.status,
                    cache=lookup.cache,
                    key=lookup.key,
                    error_message="CourtListener returned HTTP 200 but no records.",
                ),
                extra_data=lookup.extra_data,
                failure_detail=lookup.failure_detail,
            )
        return FoundCitationRetrieval(
            **common,
            candidate=_retrieved_candidate(
                citation_id,
                0,
                record,
                client=client,
                cache=docket_court_cache,
            ),
        )
    if status is RetrievalStatus.AMBIGUOUS:
        return AmbiguousCitationRetrieval(
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
    if status is RetrievalStatus.NOT_FOUND:
        return NotFoundCitationRetrieval(
            **common,
            candidate_search=search_case_name_candidates(
                citation,
                client=client,
                preparation=preparation,
            ),
        )
    if status is RetrievalStatus.THROTTLED:
        return ThrottledCitationRetrieval(
            **common,
            failure_detail=lookup.failure_detail,
        )
    return LookupFailedCitationRetrieval(
        **common,
        failure_detail=lookup.failure_detail,
    )


def _retrieved_candidate(
    citation_id: str,
    index: int,
    record: CourtListenerCitationRecord,
    *,
    client: CitationRetrievalClient,
    cache: dict[str, str | None],
) -> RetrievedCandidate:
    """Attach stable retrieval identity and provenance to one retrieved record."""
    return RetrievedCandidate(
        candidate_id=f"{citation_id}:candidate:{index}",
        record=record,
        court_resolution=resolve_court(record, client=client, cache=cache),
    )


def _status_from_lookup(status: int) -> RetrievalStatus:
    if status == HTTP_FOUND:
        return RetrievalStatus.FOUND
    if status == HTTP_MULTIPLE_CHOICES:
        return RetrievalStatus.AMBIGUOUS
    if status == HTTP_NOT_FOUND:
        return RetrievalStatus.NOT_FOUND
    if status == HTTP_BAD_REQUEST:
        return RetrievalStatus.INVALID
    if status == HTTP_TOO_MANY_REQUESTS:
        return RetrievalStatus.THROTTLED
    return RetrievalStatus.LOOKUP_FAILED


# Re-export the simpler protocol so legacy downstream imports keep working.
__all__ = [
    "CitationLookupClient",
    "run_retrieval",
    "run_retrieval_async",
]
