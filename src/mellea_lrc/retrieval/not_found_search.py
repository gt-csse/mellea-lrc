"""Case-name search for not-found citations.

When a reporter lookup 404s, the case may still be real under a different
locator. This module sends one engineered query to both CourtListener's opinion
and RECAP corpora and records each response independently. This is retrieval,
not comparison: retrieval never combines counts or interprets the results.

Kept separate from ``retrieval/pipeline.py`` so the pipeline stays a thin
existence-lookup orchestrator, mirroring ``retrieval/court_resolution.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.retrieval.types import (
    CaseNameSearchCorpus,
    CaseNameSearchProbe,
    CaseNameSearchStatus,
    CaseNameSearchTrace,
)

if TYPE_CHECKING:
    from mellea_lrc.core.citations import FullCaseCitation
    from mellea_lrc.courtlistener.types import CitationRetrievalClient

HTTP_OK = 200
_PARTY_TOKEN = re.compile(r"(?:[A-Za-z]\.){2,}|[A-Za-z0-9]+")
_COURT_ID = re.compile(r"[A-Za-z0-9_-]+")
_NON_DISTINCTIVE_PARTY_TOKENS = frozenset(
    {
        "and",
        "co",
        "company",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "llc",
        "llp",
        "lp",
        "ltd",
        "no",
        "of",
        "the",
    }
)


def search_case_name_candidates(
    citation: FullCaseCitation,
    *,
    client: CitationRetrievalClient,
) -> CaseNameSearchTrace:
    """Count CourtListener opinions matching a not-found citation's case name.

    Only runs when both parties were extracted (a real "A v. B"); a single party
    or no case name yields nothing but noise, so those are skipped. Each search
    path is represented independently, including unavailable client methods.
    """
    plaintiff = citation.plaintiff
    defendant = citation.defendant
    if not plaintiff and not defendant:
        return CaseNameSearchTrace(status=CaseNameSearchStatus.SKIPPED_NO_CASE_NAME)
    if not (plaintiff and defendant):
        return CaseNameSearchTrace(status=CaseNameSearchStatus.SKIPPED_PARTIAL_CASE_NAME)

    try:
        query = _case_name_query(plaintiff, defendant, court=citation.court)
    except ValueError:
        return CaseNameSearchTrace(status=CaseNameSearchStatus.SEARCH_FAILED)

    probes = (
        _search_corpus(client, query, CaseNameSearchCorpus.OPINIONS, "search_opinions"),
        _search_corpus(client, query, CaseNameSearchCorpus.RECAP, "search_recap"),
    )
    successes = sum(probe.status is CaseNameSearchStatus.SEARCHED for probe in probes)
    if successes == len(probes):
        status = CaseNameSearchStatus.SEARCHED
    elif successes:
        status = CaseNameSearchStatus.PARTIAL
    elif all(probe.status is CaseNameSearchStatus.SEARCH_UNAVAILABLE for probe in probes):
        status = CaseNameSearchStatus.SEARCH_UNAVAILABLE
    else:
        status = CaseNameSearchStatus.SEARCH_FAILED
    return CaseNameSearchTrace(status=status, query=query, probes=probes)


def _search_corpus(
    client: CitationRetrievalClient,
    query: str,
    corpus: CaseNameSearchCorpus,
    method_name: str,
) -> CaseNameSearchProbe:
    method = getattr(client, method_name, None)
    if not callable(method):
        return CaseNameSearchProbe(corpus, CaseNameSearchStatus.SEARCH_UNAVAILABLE)
    try:
        payload = method(query)
    except CourtListenerError as exc:
        return CaseNameSearchProbe(
            corpus,
            CaseNameSearchStatus.SEARCH_FAILED,
            http_status=exc.upstream_status_code,
            error_message=f"{type(exc).__name__}: {exc}",
        )
    except (OSError, TypeError, ValueError) as exc:
        return CaseNameSearchProbe(
            corpus,
            CaseNameSearchStatus.SEARCH_FAILED,
            error_message=f"{type(exc).__name__}: {exc}",
        )
    http_status = _http_status(payload)
    cache = _cache_status(payload)
    if http_status != HTTP_OK:
        return CaseNameSearchProbe(
            corpus,
            CaseNameSearchStatus.SEARCH_FAILED,
            http_status=http_status,
            cache=cache,
            error_message=_response_error(payload, http_status),
        )
    case_count = _case_count(payload)
    if case_count is None:
        msg = "HTTP 200 CourtListener search response omitted count"
        raise ValueError(msg)
    return CaseNameSearchProbe(
        corpus,
        CaseNameSearchStatus.SEARCHED,
        http_status=http_status,
        cache=cache,
        case_count=case_count,
    )


def _case_name_query(
    plaintiff: str,
    defendant: str,
    *,
    court: str | None = None,
) -> str:
    """Build a field query from party anchors and the extracted court, when present."""
    query = f"caseName:({_party_anchor(plaintiff)} AND {_party_anchor(defendant)})"
    if court:
        if _COURT_ID.fullmatch(court) is None:
            msg = "Court ID did not contain a searchable CourtListener slug"
            raise ValueError(msg)
        query = f"{query} AND court_id:{court}"
    return query


def _party_anchor(party: str) -> str:
    """Return the first meaningful alphanumeric token, preserving dotted acronyms."""
    tokens = [token.replace(".", "") for token in _PARTY_TOKEN.findall(party)]
    if not tokens:
        msg = "Case-name party did not contain a searchable token"
        raise ValueError(msg)
    meaningful = [token for token in tokens if token.lower() not in _NON_DISTINCTIVE_PARTY_TOKENS]
    return (meaningful or tokens)[0]


def _http_status(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    status = payload.get("http_status")
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status


def _cache_status(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    cache = payload.get("cache")
    return cache if isinstance(cache, str) and cache else None


def _response_error(payload: object, http_status: int | None) -> str:
    if isinstance(payload, Mapping):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, Mapping):
            message = detail.get("message")
            if isinstance(message, str) and message:
                return message
            return str(dict(detail))
    if http_status is None:
        return "CourtListener search ended without an HTTP response."
    return f"CourtListener search returned HTTP {http_status}."


def _case_count(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    count = payload.get("count")
    if count is None:
        raw = payload.get("raw")
        count = raw.get("count") if isinstance(raw, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count
