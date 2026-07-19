"""Case-name search for not-found citations.

When a reporter lookup 404s, the case may still be real under a different
locator. This module sends one engineered query to both CourtListener's opinion
and RECAP corpora and records each response independently. This is retrieval,
not comparison: retrieval never combines counts or interprets the results.

Kept separate from ``retrieval/pipeline.py`` so the pipeline stays a thin
existence-lookup orchestrator, mirroring ``retrieval/court_resolution.py``.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import TYPE_CHECKING

from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult
from mellea_lrc.retrieval.docket_evidence import expand_docket_evidence
from mellea_lrc.retrieval.types import (
    CaseNamePreparationStatus,
    CaseNameSearchCorpus,
    CaseNameSearchCandidate,
    CaseNameSearchPreparation,
    CaseNameSearchProbe,
    CaseNameSearchStatus,
    CaseNameSearchTrace,
    CourtListenerRequestTrace,
    DocketCandidateEvidence,
    DocketEvidenceStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.core.citations import FullCaseCitation
    from mellea_lrc.courtlistener.protocols import CitationRetrievalClient

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
        "county",
        "city",
        "cnty",
        "cty",
        "estate",
        "ex",
        "inc",
        "incorporated",
        "llc",
        "llp",
        "lp",
        "ltd",
        "no",
        "of",
        "rel",
        "the",
    }
)


def execute_search_query(
    citation: FullCaseCitation,
    *,
    client: CitationRetrievalClient,
    preparation: CaseNameSearchPreparation | None = None,
) -> CaseNameSearchTrace:
    """Count CourtListener opinions matching a not-found citation's case name.

    Only runs when both parties were extracted (a real "A v. B"); a single party
    or no case name yields nothing but noise, so those are skipped. Each search
    path is represented independently, including unavailable client methods.
    """
    if preparation is None:
        msg = "candidate search requires case-name re-extraction evidence"
        raise ValueError(msg)
    plaintiff = preparation.plaintiff
    defendant = preparation.defendant
    if preparation.status is CaseNamePreparationStatus.FAILED:
        return CaseNameSearchTrace(
            status=CaseNameSearchStatus.SEARCH_FAILED,
            preparation=preparation,
        )
    if not plaintiff and not defendant:
        return CaseNameSearchTrace(
            status=CaseNameSearchStatus.SKIPPED_NO_CASE_NAME,
            preparation=preparation,
        )
    if not (plaintiff and defendant):
        return CaseNameSearchTrace(
            status=CaseNameSearchStatus.SKIPPED_PARTIAL_CASE_NAME,
            preparation=preparation,
        )

    try:
        query = _case_name_query(
            preparation.query_plaintiff or plaintiff,
            preparation.query_defendant or defendant,
            court=preparation.court or citation.court,
        )
    except ValueError:
        return CaseNameSearchTrace(
            status=CaseNameSearchStatus.SEARCH_FAILED,
            preparation=preparation,
        )

    initial_probes = (
        _search_corpus(client, query, CaseNameSearchCorpus.OPINIONS, "search_opinions"),
        _search_corpus(client, query, CaseNameSearchCorpus.RECAP, "search_recap"),
    )
    probes = tuple(
        _expand_recap_probe(
            probe,
            client=client,
            cited_year=citation.year,
            cited_date=preparation.decision_date,
            plaintiff=plaintiff,
            defendant=defendant,
        )
        if probe.corpus is CaseNameSearchCorpus.RECAP
        else probe
        for probe in initial_probes
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
    return CaseNameSearchTrace(status=status, query=query, probes=probes, preparation=preparation)


def _search_corpus(
    client: CitationRetrievalClient,
    query: str,
    corpus: CaseNameSearchCorpus,
    method_name: str,
) -> CaseNameSearchProbe:
    method = getattr(client, method_name, None)
    if not callable(method):
        return CaseNameSearchProbe(
            corpus,
            CaseNameSearchStatus.SEARCH_UNAVAILABLE,
            CourtListenerRequestTrace(),
        )
    try:
        result = method(query)
    except CourtListenerError as exc:
        return CaseNameSearchProbe(
            corpus,
            CaseNameSearchStatus.SEARCH_FAILED,
            CourtListenerRequestTrace(
                http_status=exc.upstream_status_code,
                key=exc.cache_key,
                error_message=f"{type(exc).__name__}: {exc}",
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        return CaseNameSearchProbe(
            corpus,
            CaseNameSearchStatus.SEARCH_FAILED,
            CourtListenerRequestTrace(error_message=f"{type(exc).__name__}: {exc}"),
        )
    if not isinstance(result, CourtListenerSearchResult):
        return CaseNameSearchProbe(
            corpus,
            CaseNameSearchStatus.SEARCH_FAILED,
            CourtListenerRequestTrace(
                error_message="Search client returned an unvalidated result."
            ),
        )
    if result.http_status != HTTP_OK:
        return CaseNameSearchProbe(
            corpus,
            CaseNameSearchStatus.SEARCH_FAILED,
            CourtListenerRequestTrace(
                http_status=result.http_status,
                cache=result.cache,
                key=result.key,
                error_message=(
                    result.error_message
                    or (
                        "CourtListener search ended without an HTTP response."
                        if result.http_status is None
                        else f"CourtListener search returned HTTP {result.http_status}."
                    )
                ),
            ),
        )
    if result.count is None:
        msg = "HTTP 200 CourtListener search response omitted count"
        raise ValueError(msg)
    return CaseNameSearchProbe(
        corpus,
        CaseNameSearchStatus.SEARCHED,
        CourtListenerRequestTrace(
            http_status=result.http_status,
            cache=result.cache,
            key=result.key,
        ),
        case_count=result.count,
        candidates=tuple(
            CaseNameSearchCandidate(
                case_name=record.case_name,
                court_id=record.court_id,
                date_filed=record.date_filed,
                docket_number=record.docket_number,
                cluster_id=record.cluster_id,
                docket_id=record.docket_id,
                absolute_url=record.absolute_url,
            )
            for record in result.records[:5]
        ),
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


def _expand_recap_probe(
    probe: CaseNameSearchProbe,
    *,
    client: CitationRetrievalClient,
    cited_year: str | None,
    cited_date: str | None,
    plaintiff: str,
    defendant: str,
) -> CaseNameSearchProbe:
    """Expand only the already-bounded RECAP summaries with stable docket IDs."""
    if probe.status is not CaseNameSearchStatus.SEARCHED:
        return probe
    return replace(
        probe,
        candidates=tuple(
            replace(
                candidate,
                docket_evidence=(
                    expand_docket_evidence(
                        client=client,
                        docket_id=candidate.docket_id,
                        cited_year=cited_year,
                        cited_date=cited_date,
                    )
                    if _candidate_parties_match(candidate.case_name, plaintiff, defendant)
                    and _docket_can_contain_cited_year(candidate.date_filed, cited_year)
                    else DocketCandidateEvidence(
                        status=(
                            DocketEvidenceStatus.SKIPPED_PARTY_MISMATCH
                            if not _candidate_parties_match(
                                candidate.case_name,
                                plaintiff,
                                defendant,
                            )
                            else DocketEvidenceStatus.SKIPPED_AFTER_CITED_YEAR
                        ),
                        case_name=candidate.case_name,
                        court_id=candidate.court_id,
                        docket_number=candidate.docket_number,
                        date_filed=candidate.date_filed,
                    )
                ),
            )
            if candidate.docket_id
            else candidate
            for candidate in probe.candidates
        ),
    )


def _candidate_parties_match(
    case_name: str | None,
    plaintiff: str,
    defendant: str,
) -> bool:
    """Require both engineered party anchors before spending expansion requests."""
    if not case_name:
        return False
    tokens = {token.lower().replace(".", "") for token in _PARTY_TOKEN.findall(case_name)}
    return _party_anchor(plaintiff).lower() in tokens and _party_anchor(defendant).lower() in tokens


def _docket_can_contain_cited_year(date_filed: str | None, cited_year: str | None) -> bool:
    """Prune only dockets that begin well after the asserted decision year.

    A docket's ``date_filed`` is the proceeding's opening date, whereas the
    citation year describes the decision.  A one-year tolerance preserves
    candidates affected by source/citation metadata disagreement without
    spending two expansion requests on obviously later, same-name cases.
    """
    if not date_filed or not cited_year:
        return True
    filed_year = date_filed[:4]
    return not (
        filed_year.isdigit()
        and cited_year.isdigit()
        and int(filed_year) > int(cited_year) + 1
    )
