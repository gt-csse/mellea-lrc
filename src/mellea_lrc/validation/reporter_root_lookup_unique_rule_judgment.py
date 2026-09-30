"""Rule-judge saved unique reporter lookups without provider calls."""

from __future__ import annotations

from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterExactLookupOutcome
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_exact.fields import (
    case_name_result,
    court_result,
    date_result,
    locator_present,
)

STAGE = "13.1_reporter_root_lookup_unique_rule_judgment"
LOOKUP_STAGE = "12.2_reporter_root_lookup_docket_retrieval"


def reporter_root_lookup_unique_rule_judgment(document: Document) -> Document:
    """Compare each unique candidate's fields and route disagreements to review."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if LOOKUP_STAGE not in document.stage_runs:
        raise ValueError("Retrieve reporter lookups before reviewing them")
    for root in tuple(item for item in document.roots if isinstance(item, FullReporterCitation)):
        if root.next_stage != STAGE:
            continue
        lookup = root.reporter_exact_lookup
        if (
            lookup is None
            or lookup.outcome is not ReporterExactLookupOutcome.UNIQUE
            or lookup.query is None
            or lookup.response is None
            or len(lookup.response.clusters) != 1
        ):
            raise ValueError("Unique reporter review requires one saved candidate")
        citation = root.record(STAGE)
        candidate = lookup.response.clusters[0]
        docket = citation.reporter_exact_docket.response if citation.reporter_exact_docket else None
        results: list[MatchResult] = []
        if citation.case_name:
            result = case_name_result(citation, candidate)
            citation = citation.with_case_name_judgment(len(citation.case_name) - 1, 0, result)
            results.append(result)
        if citation.court:
            result = court_result(citation, candidate, docket)
            citation = citation.with_court_judgment(len(citation.court) - 1, 0, result)
            results.append(result)
        if citation.date and candidate.date_filed:
            result = date_result(citation, candidate)
            citation = citation.with_date_judgment(len(citation.date) - 1, 0, result)
            results.append(result)
        if (
            citation.case_name_judgments
            and all(result is MatchResult.MATCH for result in results)
            and locator_present(candidate, lookup.query) is not False
        ):
            citation = citation.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY).with_route(None)
        else:
            citation = citation.with_route("14_reporter_root_lookup_unique_llm_judgment")
        document = document.replace_citation(citation)
    return document.complete(STAGE)
