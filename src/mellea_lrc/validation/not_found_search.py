"""Case-name search for not-found citations.

When a reporter lookup 404s, the case may still be real under a different
locator. This module runs a single CourtListener relevance search on the
extracted case name and records only *how many* opinions matched — a retrieval,
not a comparison. Deciding whether any candidate is actually the cited case is
the assessment stage's job (case names are non-unique and often only
semantically equivalent), so nothing here inspects the individual results.

Kept separate from ``validation/pipeline.py`` so the pipeline stays a thin
existence-lookup orchestrator, mirroring ``validation/court_resolution.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.validation.types import CaseNameSearchStatus, CaseNameSearchTrace

if TYPE_CHECKING:
    from mellea_lrc.core.citations import FullCaseCitation
    from mellea_lrc.courtlistener.types import CitationValidationClient


def search_case_name_candidates(
    citation: FullCaseCitation,
    *,
    client: CitationValidationClient,
) -> CaseNameSearchTrace:
    """Count CourtListener opinions matching a not-found citation's case name.

    Only runs when both parties were extracted (a real "A v. B"); a single party
    or no case name yields nothing but noise, so those are skipped. Clients that
    cannot search (``search_opinions`` absent) are recorded explicitly rather
    than treated as an error.
    """
    plaintiff = citation.plaintiff
    defendant = citation.defendant
    if not plaintiff and not defendant:
        return CaseNameSearchTrace(status=CaseNameSearchStatus.SKIPPED_NO_CASE_NAME)
    if not (plaintiff and defendant):
        return CaseNameSearchTrace(status=CaseNameSearchStatus.SKIPPED_PARTIAL_CASE_NAME)

    query = f'caseName:"{plaintiff} v. {defendant}"'
    if not hasattr(client, "search_opinions"):
        return CaseNameSearchTrace(status=CaseNameSearchStatus.SEARCH_UNAVAILABLE, query=query)

    try:
        payload = client.search_opinions(query)
    except (CourtListenerError, OSError, TypeError, ValueError) as exc:
        return CaseNameSearchTrace(
            status=CaseNameSearchStatus.SEARCH_FAILED,
            query=query,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return CaseNameSearchTrace(
        status=CaseNameSearchStatus.SEARCHED,
        query=query,
        case_count=_case_count(payload),
    )


def _case_count(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    count = payload.get("count")
    if isinstance(count, bool) or not isinstance(count, int):
        return None
    return count
