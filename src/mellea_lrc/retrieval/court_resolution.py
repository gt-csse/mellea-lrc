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
    CourtResolutionSource,
    CourtResolutionTrace,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.types import CourtListenerCitationRecord, CitationRetrievalClient


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
    courtlistener_court_id, resolved_via, docket_id, docket_url, cached, error_message = (
        _resolve_courtlistener_court(record, client, cache)
    )

    return CourtResolutionTrace(
        courtlistener_court_id=courtlistener_court_id,
        resolved_via=resolved_via,
        docket_id=docket_id,
        docket_url=docket_url,
        cached=cached,
        error_message=error_message,
    )


def _resolve_courtlistener_court(
    record: CourtListenerCitationRecord,
    client: CitationRetrievalClient,
    cache: dict[str, str | None],
) -> tuple[str | None, CourtResolutionSource, str | None, str | None, bool, str | None]:
    """Return ``(court_id, source, docket_id, docket_url, cached, error)``."""
    if record.court_id:
        return record.court_id, CourtResolutionSource.CLUSTER_PROVIDED, None, None, False, None

    if not record.docket_id:
        return None, CourtResolutionSource.NO_DOCKET_ID, None, None, False, None

    docket_id = record.docket_id
    docket_url = f"/dockets/{docket_id}"

    if docket_id in cache:
        cached_id = cache[docket_id]
        return cached_id, CourtResolutionSource.DOCKET_LOOKUP, docket_id, docket_url, True, None

    if not hasattr(client, "get_docket"):
        # A client implementing only CitationLookupClient cannot enrich; record the
        # inability explicitly rather than silently catching AttributeError from a
        # missing method (the pre-refactor behaviour swallowed implementation bugs).
        cache[docket_id] = None
        return None, CourtResolutionSource.NO_DOCKET_ID, docket_id, docket_url, False, None

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
        )

    court_id = docket.get("court_id") if isinstance(docket, Mapping) else None
    court_id = court_id if isinstance(court_id, str) else None
    cache[docket_id] = court_id
    return court_id, CourtResolutionSource.DOCKET_LOOKUP, docket_id, docket_url, False, None
