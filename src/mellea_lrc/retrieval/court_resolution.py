"""Court resolution for retrieved citation candidates.

Court resolution is the deterministic (non-LLM) work of deciding which
CourtListener court a retrieved candidate belongs to. The resolution ``trace``
captures how the CourtListener court was obtained (cluster payload, docket
GET, or unavailable). Retrieval only retrieves this data; it never compares
the resolved court against the citation court from extraction — that
comparison is the assessment stage's job.

This module owns that work so ``retrieval/pipeline.py`` stays a thin existence
lookup; the court concern no longer leaks into the retrieval orchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.retrieval.types import (
    CourtListenerRequestTrace,
    CourtResolutionSource,
    CourtResolutionTrace,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationRecord
    from mellea_lrc.courtlistener.protocols import CitationRetrievalClient


def resolve_court(
    record: CourtListenerCitationRecord,
    *,
    client: CitationRetrievalClient,
    cache: dict[str, str | None],
) -> CourtResolutionTrace:
    """Resolve the CourtListener court for one retrieved match.

    Args:
        record: One CourtListener record retrieved as a candidate.
        client: The retrieval client, used only when a docket GET is required. Must
            expose ``get_docket`` per ``CitationRetrievalClient``; clients that only
            implement ``CitationLookupClient`` are detected via ``hasattr`` and treated
            as ``NO_DOCKET_ID`` rather than swallowing ``AttributeError``.
        cache: Per-run docket-id -> resolved-court cache, normalized to ``str`` keys so
            JSON-int (``42``) and JSON-string (``"42"``) collapse to one entry.

    """
    (
        courtlistener_court_id,
        resolved_via,
        docket_id,
        docket_url,
        _cached,
        _error_message,
        request_trace,
    ) = _resolve_courtlistener_court(record, client, cache)

    return CourtResolutionTrace(
        courtlistener_court_id=courtlistener_court_id,
        resolved_via=resolved_via,
        docket_id=docket_id,
        docket_url=docket_url,
        request_trace=request_trace,
    )


def _resolve_courtlistener_court(
    record: CourtListenerCitationRecord,
    client: CitationRetrievalClient,
    cache: dict[str, str | None],
) -> tuple[
    str | None,
    CourtResolutionSource,
    str | None,
    str | None,
    bool,
    str | None,
    CourtListenerRequestTrace | None,
]:
    """Return the resolved court fields and any docket-request trace."""
    if record.court_id:
        return record.court_id, CourtResolutionSource.CLUSTER_PROVIDED, None, None, False, None, None

    if not record.docket_id:
        return None, CourtResolutionSource.NO_DOCKET_ID, None, None, False, None, None

    docket_id = record.docket_id
    docket_url = f"/dockets/{docket_id}"

    if docket_id in cache:
        cached_id = cache[docket_id]
        return (
            cached_id,
            CourtResolutionSource.DOCKET_LOOKUP,
            docket_id,
            docket_url,
            True,
            None,
            CourtListenerRequestTrace(cache="memory"),
        )

    if not hasattr(client, "get_docket"):
        # A client implementing only CitationLookupClient cannot enrich; record the
        # inability explicitly rather than silently catching AttributeError from a
        # missing method (the pre-refactor behaviour swallowed implementation bugs).
        cache[docket_id] = None
        return None, CourtResolutionSource.NO_DOCKET_ID, docket_id, docket_url, False, None, None

    try:
        docket = client.get_docket(docket_id)
    except (CourtListenerError, OSError, TypeError, ValueError) as exc:
        cache[docket_id] = None
        return (
            None,
            CourtResolutionSource.DOCKET_LOOKUP_FAILED,
            docket_id,
            docket_url,
            False,
            (f"{type(exc).__name__}: {exc}"),
            CourtListenerRequestTrace(
                http_status=exc.upstream_status_code if isinstance(exc, CourtListenerError) else None,
                key=exc.cache_key if isinstance(exc, CourtListenerError) else None,
                error_message=f"{type(exc).__name__}: {exc}",
            ),
        )

    court_id = docket.get("court_id") if isinstance(docket, Mapping) else None
    court_id = court_id if isinstance(court_id, str) else None
    cache[docket_id] = court_id
    # ``cached`` reflects the persistent R2 cache, not the per-run in-memory
    # dedup above. The per-run cache only avoids redundant calls within one
    # ``run_retrieval`` execution; the trace answers "was this served from the
    # persistent cache on a later run?", which is what the caller cares about.
    r2_cache_hit = docket.get("cache") == "hit" if isinstance(docket, Mapping) else False
    request_trace = CourtListenerRequestTrace(
        http_status=_optional_int(docket.get("http_status")),
        cache=_optional_str(docket.get("cache")),
        key=_optional_str(docket.get("key")),
    )
    return (
        court_id,
        CourtResolutionSource.DOCKET_LOOKUP,
        docket_id,
        docket_url,
        r2_cache_hit,
        None,
        request_trace,
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
