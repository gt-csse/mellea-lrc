"""Court resolution for found citations.

Court resolution is the deterministic (non-LLM) work of deciding which
CourtListener court a found citation belongs to. The resolution ``trace``
captures how the CourtListener court was obtained (cluster payload, docket
GET, or unavailable). Validation only retrieves this data; it never compares
the resolved court against the citation court from extraction — that
comparison is the assessment stage's job.

This module owns that work so ``validation/pipeline.py`` stays a thin existence
lookup; the court concern no longer leaks into the validation orchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.validation.types import (
    CourtResolutionSource,
    CourtResolutionTrace,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.types import CitationMatch, CitationValidationClient


def resolve_court(
    match: CitationMatch,
    *,
    client: CitationValidationClient,
    cache: dict[str, str | None],
) -> CourtResolutionTrace:
    """Resolve the CourtListener court for one found match.

    Args:
        match: The single found ``CitationMatch`` (typically ``lookup.matches[0]``).
        client: The validation client, used only when a docket GET is required. Must
            expose ``get_docket`` per ``CitationValidationClient``; clients that only
            implement ``CitationLookupClient`` are detected via ``hasattr`` and treated
            as ``NO_DOCKET_ID`` rather than swallowing ``AttributeError``.
        cache: Per-run docket-id -> resolved-court cache, normalized to ``str`` keys so
            JSON-int (``42``) and JSON-string (``"42"``) collapse to one entry.

    """
    courtlistener_court_id, resolved_via, docket_id, docket_url, cached, error_message = (
        _resolve_courtlistener_court(match, client, cache)
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
    match: CitationMatch,
    client: CitationValidationClient,
    cache: dict[str, str | None],
) -> tuple[str | None, CourtResolutionSource, str | None, str | None, bool, str | None]:
    """Return ``(court_id, source, docket_id, docket_url, cached, error)``."""
    if match.court_id:
        return match.court_id, CourtResolutionSource.CLUSTER_PROVIDED, None, None, False, None

    if not match.docket_id:
        return None, CourtResolutionSource.NO_DOCKET_ID, None, None, False, None

    docket_id = match.docket_id
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
        return None, CourtResolutionSource.DOCKET_LOOKUP_FAILED, docket_id, docket_url, False, (
            f"{type(exc).__name__}: {exc}"
        )

    court_id = docket.get("court_id") if isinstance(docket, Mapping) else None
    court_id = court_id if isinstance(court_id, str) else None
    cache[docket_id] = court_id
    return court_id, CourtResolutionSource.DOCKET_LOOKUP, docket_id, docket_url, False, None
