"""Rule-only assessment of saved reporter lookups with multiple candidates."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Protocol

from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerDocket
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactAmbiguityResolution,
    ReporterExactCandidateDocket,
    ReporterExactLookupOutcome,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation._support.reporter_exact_fields import (
    candidate_court_id,
    case_name_result,
    court_result,
    date_result,
    locator_present,
)
from mellea_lrc.validation.stage_names import (
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    REPORTER_ROOT_LOOKUP_LARGE_CANDIDATE_REVIEW,
)

STAGE = REPORTER_ROOT_LOOKUP_AMBIGUOUS
LOOKUP_STAGE = REPORTER_ROOT_LOOKUP
CANDIDATE_LIMIT = 20


class ReporterDocketClient(Protocol):
    """Only linked-docket retrieval is needed; exact results are already saved."""

    def get_docket(self, docket_id: str) -> CourtListenerDocket | None: ...


def reporter_root_lookup_ambiguous(
    document: Document,
    *,
    client: ReporterDocketClient | None = None,
) -> Document:
    """Judge every bounded candidate and admit only a unique full rule match.

    Zero or several passing candidates remain available for a later model
    review. A result with at least 20 candidates is preserved but not sent
    through this bounded rule review. This stage never repeats citation lookup.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if LOOKUP_STAGE not in document.stage_runs:
        raise ValueError("Complete reporter lookup before ambiguous-candidate review")
    roots = tuple(root for root in document.roots if isinstance(root, FullReporterCitation))
    docket_cache: dict[str, CourtListenerDocket | None] = {}
    with ExitStack() as stack:
        service = client
        for root in roots:
            if not root.identity_judgments or root.identity_judgments[-1].next_stage != STAGE:
                continue
            lookup = root.reporter_exact_lookup
            if (
                lookup is None
                or lookup.outcome is not ReporterExactLookupOutcome.AMBIGUOUS
                or lookup.query is None
                or lookup.response is None
            ):
                raise ValueError("Ambiguous route requires a saved multi-candidate reporter lookup")
            recorded = root.record(STAGE)
            candidates = lookup.response.clusters
            if len(candidates) >= CANDIDATE_LIMIT:
                resolution = ReporterExactAmbiguityResolution(
                    node_id=recorded.nodes[-1].id,
                    outcome=ReporterExactAmbiguityOutcome.CANDIDATE_LIMIT_EXCEEDED,
                )
                recorded = recorded.with_reporter_exact_ambiguity_resolution(resolution)
                recorded = recorded.with_identity_judgment(
                    IdentityVerdict.DEFERRED, REPORTER_ROOT_LOOKUP_LARGE_CANDIDATE_REVIEW
                )
            else:
                passing: list[int] = []
                for index, candidate in enumerate(candidates):
                    docket: CourtListenerDocket | None = None
                    if recorded.court and candidate.docket_id and candidate_court_id(candidate) is None:
                        docket_id = candidate.docket_id
                        if docket_id not in docket_cache:
                            if service is None:
                                service = stack.enter_context(CourtListenerClient())
                            docket_cache[docket_id] = service.get_docket(docket_id)
                        docket = docket_cache[docket_id]
                        recorded = recorded.with_reporter_exact_candidate_docket(
                            ReporterExactCandidateDocket(
                                node_id=recorded.nodes[-1].id,
                                candidate_index=index,
                                docket_id=docket_id,
                                response=docket,
                            )
                        )
                    results: list[MatchResult] = []
                    if recorded.case_name:
                        result = case_name_result(recorded, candidate)
                        recorded = recorded.with_case_name_judgment(
                            len(recorded.case_name) - 1, index, result
                        )
                        results.append(result)
                    if recorded.court:
                        result = court_result(recorded, candidate, docket)
                        recorded = recorded.with_court_judgment(len(recorded.court) - 1, index, result)
                        results.append(result)
                    # As in the unique route, an unavailable date expresses no date opinion.
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
                    recorded.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
                    if selected is not None
                    else recorded.with_identity_judgment(
                        IdentityVerdict.DEFERRED, REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM
                    )
                )
            document = document.replace_citation(recorded)
    return document.complete(STAGE)
