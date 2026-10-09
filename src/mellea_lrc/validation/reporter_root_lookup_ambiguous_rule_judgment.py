"""Rule-only assessment of saved ambiguous reporter evidence."""

from __future__ import annotations

from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    REPORTER_LOOKUP_CANDIDATE_LIMIT,
    ReporterExactAmbiguityOutcome,
    ReporterExactAmbiguityResolution,
    ReporterExactLookupOutcome,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_exact.fields import (
    case_name_result,
    court_result,
    date_result,
    locator_present,
)

SUBSTAGE = "validate_roots.reporter_lookup.ambiguous_rule_judgment"
LOOKUP_SUBSTAGE = "validate_roots.reporter_lookup.docket_retrieval"


def reporter_root_lookup_ambiguous_rule_judgment(
    document: Document,
) -> Document:
    """Judge every bounded candidate and admit only a unique full rule match.

    Zero or several passing candidates remain available for a later model
    review. A result with at least 20 candidates is preserved but not sent
    through this bounded rule review. This substage never repeats citation lookup.
    """
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if LOOKUP_SUBSTAGE not in document.substage_runs:
        raise ValueError("Retrieve ambiguous docket evidence before candidate review")
    roots = tuple(root for root in document.roots if isinstance(root, FullReporterCitation))
    for root in roots:
        if root.next_substage != SUBSTAGE:
            continue
        lookup = root.reporter_exact_lookup
        if (
            lookup is None
            or lookup.outcome is not ReporterExactLookupOutcome.AMBIGUOUS
            or lookup.query is None
            or lookup.response is None
        ):
            raise ValueError("Ambiguous route requires a saved multi-candidate reporter lookup")
        recorded = root.record(SUBSTAGE)
        candidates = lookup.response.clusters
        if len(candidates) >= REPORTER_LOOKUP_CANDIDATE_LIMIT:
            resolution = ReporterExactAmbiguityResolution(
                node_id=recorded.nodes[-1].id,
                outcome=ReporterExactAmbiguityOutcome.CANDIDATE_LIMIT_EXCEEDED,
            )
            recorded = recorded.with_reporter_exact_ambiguity_resolution(resolution)
            recorded = recorded.with_route("reporter_root_lookup_large_candidate_review")
        else:
            dockets = {
                item.candidate_index: item.response for item in recorded.reporter_exact_candidate_dockets
            }
            passing: list[int] = []
            for index, candidate in enumerate(candidates):
                results: list[MatchResult] = []
                if recorded.case_name:
                    result = case_name_result(recorded, candidate)
                    recorded = recorded.with_case_name_judgment(len(recorded.case_name) - 1, index, result)
                    results.append(result)
                if recorded.court:
                    result = court_result(recorded, candidate, dockets.get(index))
                    recorded = recorded.with_court_judgment(len(recorded.court) - 1, index, result)
                    results.append(result)
                if recorded.date and candidate.date_filed:
                    result = date_result(recorded, candidate)
                    recorded = recorded.with_date_judgment(len(recorded.date) - 1, index, result)
                    results.append(result)
                if (
                    recorded.case_name
                    and results
                    and all(result is MatchResult.MATCH for result in results)
                    and locator_present(candidate, lookup.query) is not False
                ):
                    passing.append(index)
            selected = passing[0] if len(passing) == 1 else None
            resolution = ReporterExactAmbiguityResolution(
                node_id=recorded.nodes[-1].id,
                outcome=(
                    ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH
                    if selected is not None
                    else ReporterExactAmbiguityOutcome.NO_UNIQUE_RULE_MATCH
                ),
                passing_candidate_indices=tuple(passing),
                selected_candidate_index=selected,
            )
            recorded = recorded.with_reporter_exact_ambiguity_resolution(resolution)
            recorded = (
                recorded.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY).with_route(None)
                if selected is not None
                else recorded.with_route("validate_roots.reporter_lookup.ambiguous_llm_judgment")
            )
        document = document.replace_citation(recorded)
    return document.complete_substage(SUBSTAGE)
