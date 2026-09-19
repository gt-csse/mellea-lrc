"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.validation.aggregation import (
    requires_mellea_locator_candidate_choice,
    run_future_implementation_deferred_locator_identity_resolution,
    run_locator_candidate_assessment,
    run_locator_citation_summary,
    run_locator_identity_resolution,
    run_mellea_locator_candidate_choice,
    run_opinion_search_candidate_assessment,
    run_recap_search_candidate_assessment,
    run_search_citation_summary,
    run_search_deferred_locator_identity_resolution,
)
from mellea_lrc.validation.candidate_evaluation import (
    run_locator_candidate_evaluation,
    run_opinion_search_candidate_evaluation,
    run_recap_search_candidate_evaluation,
)
from mellea_lrc.validation.candidate_selection import (
    run_locator_candidate_selection,
    run_opinion_search_candidate_selection,
    run_recap_search_candidate_selection,
)
from mellea_lrc.validation.candidate_state import CandidateValidationState
from mellea_lrc.validation.case_search import (
    run_mellea_case_name_query_preparation,
    run_opinion_search,
    run_recap_search,
)
from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup
from mellea_lrc.validation.court_retrieval import run_docket_court_retrieval
from mellea_lrc.validation.field_checks import (
    run_court_check,
    run_exact_case_name_check,
    run_mellea_case_name_check,
    run_mellea_case_name_reextraction,
    run_year_check,
)
from mellea_lrc.validation.pinpoint_retrieval import (
    run_mellea_citing_proposition_extraction,
    run_mellea_pinpoint_check,
    run_reporter_page_retrieval,
)
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameReextractionOutcome,
    MelleaLocatorCandidateChoiceOutcome,
    OpinionSearchOutcome,
    RecapSearchOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import (
        CitationValidation,
        LocatorCitationSummaryNode,
        MelleaCaseNameCheckNode,
    )


@dataclass(frozen=True, slots=True)
class CitationValidationRunner:
    """Run one citation with bound validation dependencies."""

    client: CourtListenerServiceClient

    def run_exact_full_reporter_locator_lookup(
        self,
        validation: CitationValidation,
    ) -> CitationValidation:
        """Run only CourtListener's exact full-reporter-locator lookup.

        This deliberately makes no field assessment or identity decision.
        A caller can persist its candidate set, then independently run the
        unique-candidate and bounded-ambiguity identity stages.
        """
        stated = validation.citation.stated
        if not isinstance(stated, FullCaseCitation):
            msg = "Exact full reporter-locator lookup accepts only FullCaseCitation records"
            raise ValueError(msg)
        return validation.append(run_exact_locator_lookup(validation, client=self.client))

    async def run_full_reporter_locator_identity(
        self,
        validation: CitationValidation,
        *,
        document_text: str,
        session: MelleaSession | None = None,
    ) -> CitationValidation:
        """Resolve one full reporter locator through exact lookup evidence.

        A full reporter locator may take the exact unique or bounded ambiguous
        route. A lookup miss is deferred to the later search checkpoint; an
        over-limit result set is deferred to future implementation. The caller
        must filter the document to full reporter locators before this method:
        docket and other locator kinds are intentionally outside this stage.
        """
        validation = self.run_exact_full_reporter_locator_lookup(validation)
        exact_locator_lookup_node = validation.nodes[-1]
        if not isinstance(exact_locator_lookup_node, ExactLocatorLookupNode):
            msg = "Exact full reporter-locator lookup did not append its lookup node"
            raise ValueError(msg)

        if exact_locator_lookup_node.outcome is LocatorLookupOutcome.FOUND:
            return await self.run_locator_found_identity(
                validation,
                lookup=exact_locator_lookup_node,
                document_text=document_text,
                session=session,
            )
        if exact_locator_lookup_node.outcome is LocatorLookupOutcome.AMBIGUOUS:
            return await self.run_locator_ambiguous(
                validation,
                lookup=exact_locator_lookup_node,
                document_text=document_text,
                session=session,
            )
        if exact_locator_lookup_node.outcome is LocatorLookupOutcome.NOT_FOUND:
            return validation.append(
                run_search_deferred_locator_identity_resolution(
                    validation,
                    depends_on=(exact_locator_lookup_node.node_id,),
                    reason="Exact reporter-locator lookup found no candidate; continue in the search stage.",
                )
            )
        return validation.append(
            run_future_implementation_deferred_locator_identity_resolution(
                validation,
                depends_on=(exact_locator_lookup_node.node_id,),
                reason=(
                    "Full reporter-locator identity could not produce an exact candidate set eligible "
                    "for current review."
                ),
            )
        )

    async def run_locator_found_identity(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
        document_text: str,
        session: MelleaSession | None,
    ) -> CitationValidation:
        """Check a unique locator candidate without entering pinpoint work."""
        if lookup.outcome is not LocatorLookupOutcome.FOUND:
            msg = "run_locator_found_identity requires a found locator"
            raise ValueError(msg)
        if lookup.cluster is None:
            msg = "Found locator requires one opinion cluster"
            raise ValueError(msg)
        candidate = run_locator_candidate_evaluation(
            validation,
            cluster=lookup.cluster,
            candidate_index=1,
            depends_on=(lookup.node_id,),
        )
        validation = validation.append(candidate)
        validation = await self.run_locator_candidate_validation(
            validation,
            lookup=lookup,
            candidate=candidate,
            document_text=document_text,
            session=session,
            state=CandidateValidationState(),
        )
        summary = run_locator_citation_summary(validation)
        validation = validation.append(summary)
        return await _resolve_locator_identity_from_summary(
            validation,
            summary=summary,
            document_text=document_text,
            session=session,
        )

    async def run_locator_found(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
        document_text: str,
        session: MelleaSession | None,
    ) -> CitationValidation:
        """Run the complete graph rooted in one uniquely resolved locator.

        Graph:
            found locator
            └── locator candidate evaluation
                ├── exact case-name check + year check + docket court retrieval
                │   └── exact case-name mismatch ->
                │       ``run_locator_candidate_case_name_recovery``
                ├── docket court retrieval -> court check
                ├── reporter-page retrieval -> citing-proposition extraction -> pinpoint check
                └── completed checks -> locator candidate assessment -> citation summary
        """
        if lookup.outcome is not LocatorLookupOutcome.FOUND:
            msg = "run_locator_found requires a found locator"
            raise ValueError(msg)
        if lookup.cluster is None:
            msg = "Found locator requires one opinion cluster"
            raise ValueError(msg)
        candidate = run_locator_candidate_evaluation(
            validation,
            cluster=lookup.cluster,
            candidate_index=1,
            depends_on=(lookup.node_id,),
        )
        validation = validation.append(candidate)
        validation = await self.run_locator_candidate_validation(
            validation,
            lookup=lookup,
            candidate=candidate,
            document_text=document_text,
            session=session,
            state=CandidateValidationState(),
        )
        # MVE scope: reporter-page retrieval belongs only to the unique
        # exact-locator FOUND route. Ambiguous and search-derived candidates
        # intentionally wait for broader candidate/opinion scope semantics.
        retrieval = run_reporter_page_retrieval(validation, evaluation=candidate, client=self.client)
        validation = validation.append(retrieval)
        proposition = await run_mellea_citing_proposition_extraction(
            validation,
            trigger=retrieval,
            document_text=document_text,
            session=session,
        )
        validation = validation.append(proposition)
        validation = validation.append(
            await run_mellea_pinpoint_check(
                validation,
                retrieval=retrieval,
                proposition=proposition,
                session=session,
            )
        )
        summary = run_locator_citation_summary(validation)
        validation = validation.append(summary)
        return await _resolve_locator_identity_from_summary(
            validation,
            summary=summary,
            document_text=document_text,
            session=session,
        )

    async def run_locator_candidate_validation(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
        candidate: CandidateEvaluationNode,
        document_text: str,
        session: MelleaSession | None,
        state: CandidateValidationState,
    ) -> CitationValidation:
        """Complete the reusable validation subtree for one locator candidate.

        Graph:
            locator candidate evaluation
            ├── exact case-name check
            │   └── non-match -> ``run_locator_candidate_case_name_recovery``
            ├── year check
            ├── docket court retrieval -> court check
            └── locator candidate assessment
        """
        exact_case_name_check_node = run_exact_case_name_check(validation, candidate=candidate)
        year_check_node = run_year_check(validation, candidate=candidate)
        docket_court_retrieval_node = run_docket_court_retrieval(
            validation,
            candidate=candidate,
            client=self.client,
        )
        court_check_node = run_court_check(validation, evidence=docket_court_retrieval_node)
        validation = (
            validation.append(exact_case_name_check_node)
            .append(year_check_node)
            .append(docket_court_retrieval_node)
            .append(court_check_node)
        )
        state = _with_exact_case_name_result(state, exact_case_name_check_node)
        if exact_case_name_check_node.outcome is not FieldCheckOutcome.MATCH:
            validation, state = await self.run_locator_candidate_case_name_recovery(
                validation,
                lookup=lookup,
                candidate=candidate,
                exact_case_name_check=exact_case_name_check_node,
                document_text=document_text,
                session=session,
                state=state,
            )
        return validation.append(
            run_locator_candidate_assessment(
                validation,
                candidate=candidate,
                state=state,
            )
        )

    async def run_locator_candidate_case_name_recovery(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
        candidate: CandidateEvaluationNode,
        exact_case_name_check: ExactCaseNameCheckNode,
        document_text: str,
        session: MelleaSession | None,
        state: CandidateValidationState,
    ) -> tuple[CitationValidation, CandidateValidationState]:
        """Run the complete case-name recovery graph after a mismatch or a missing extraction.

        Graph:
            exact case-name check
            ├── match -> end
            ├── unavailable, no retrieved case name either -> end
            ├── unavailable, retrieved case name present -> Mellea local party re-extraction
            │   ├── complete -> Mellea re-extracted case-name check -> end
            │   └── partial, not found, unavailable, or failed -> end
            └── mismatch -> Mellea semantic case-name check
                ├── match or failed -> end
                └── mismatch -> Mellea local party re-extraction
                    ├── complete -> Mellea re-extracted case-name check -> end
                    └── partial, not found, unavailable, or failed -> end
        """
        if exact_case_name_check.outcome is FieldCheckOutcome.MATCH:
            return validation, state
        if (
            exact_case_name_check.outcome is FieldCheckOutcome.UNAVAILABLE
            and exact_case_name_check.retrieved_case_name is None
        ):
            # Nothing to eventually compare a recovered name against.
            return validation, state
        reextraction_trigger: ExactCaseNameCheckNode | MelleaCaseNameCheckNode = exact_case_name_check
        if exact_case_name_check.outcome is FieldCheckOutcome.MISMATCH:
            semantic = await run_mellea_case_name_check(
                validation,
                case_name_evidence=exact_case_name_check,
                session=session,
            )
            validation = validation.append(semantic)
            state = state.with_case_name_result(
                outcome=_aggregated_mellea_outcome(semantic.outcome),
                evidence="mellea",
                dependency_id=semantic.node_id,
            )
            if semantic.outcome is not MelleaCaseNameCheckOutcome.MISMATCH:
                return validation, state
            reextraction_trigger = semantic
        # A genuinely unavailable extraction still deserves a recovery attempt:
        # local re-extraction reads document text directly and may succeed
        # where the deterministic extractor found nothing at all.
        reextraction = await run_mellea_case_name_reextraction(
            validation,
            trigger=reextraction_trigger,
            locator_lookup=lookup,
            document_text=document_text,
            session=session,
        )
        validation = validation.append(reextraction)
        if reextraction.status is ValidationNodeStatus.SUCCEEDED:
            validation.citation.mark_extraction_reviewed_by_llm()
        state = state.with_reextraction(reextraction)
        if reextraction.outcome is not MelleaCaseNameReextractionOutcome.COMPLETE:
            return validation, state
        reextracted_check = await run_mellea_case_name_check(
            validation,
            case_name_evidence=reextraction,
            candidate=candidate,
            session=session,
        )
        validation = validation.append(reextracted_check)
        state = state.with_case_name_result(
            outcome=_aggregated_mellea_outcome(reextracted_check.outcome),
            evidence="mellea_reextracted",
            dependency_id=reextracted_check.node_id,
        )
        return validation, state

    async def run_locator_not_found(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
        document_text: str,
        session: MelleaSession | None,
    ) -> CitationValidation:
        """Run the complete local re-extraction graph rooted in a locator miss.

        Graph:
            locator not found
            └── Mellea local party re-extraction
                └── Mellea case-name query preparation
                    ├── CourtListener opinion search -> candidate selection -> evaluation x selected candidate
                    └── CourtListener RECAP search -> candidate selection -> evaluation x selected candidate
        """
        if lookup.outcome is not LocatorLookupOutcome.NOT_FOUND:
            msg = "run_locator_not_found requires a not-found locator"
            raise ValueError(msg)
        reextraction = await run_mellea_case_name_reextraction(
            validation,
            trigger=lookup,
            locator_lookup=lookup,
            document_text=document_text,
            session=session,
        )
        validation = validation.append(reextraction)
        if reextraction.status is ValidationNodeStatus.SUCCEEDED:
            validation.citation.mark_extraction_reviewed_by_llm()
        search_state = CandidateValidationState().with_reextraction(reextraction)
        preparation = await run_mellea_case_name_query_preparation(
            validation,
            reextraction=reextraction,
            session=session,
        )
        validation = validation.append(preparation)
        opinion_search = run_opinion_search(validation, preparation=preparation, client=self.client)
        recap_search = run_recap_search(validation, preparation=preparation, client=self.client)
        validation = validation.append(opinion_search).append(recap_search)
        if opinion_search.outcome is OpinionSearchOutcome.SEARCHED:
            opinion_selection = run_opinion_search_candidate_selection(
                validation,
                search=opinion_search,
            )
            validation = validation.append(opinion_selection)
            if opinion_selection.selected_candidate_count:
                results = opinion_search.results[: opinion_selection.selected_candidate_count]
                if len(results) != opinion_selection.selected_candidate_count:
                    msg = "Opinion-search result payload is shorter than its selected candidate count"
                    raise ValueError(msg)
                for candidate_index, result in enumerate(results, start=1):
                    candidate = run_opinion_search_candidate_evaluation(
                        validation,
                        result=result,
                        candidate_index=candidate_index,
                        depends_on=(opinion_selection.node_id,),
                    )
                    validation = validation.append(candidate)
                    validation = await self.run_search_candidate_validation(
                        validation,
                        candidate=candidate,
                        session=session,
                        state=search_state,
                    )
        if recap_search.outcome is RecapSearchOutcome.SEARCHED:
            recap_selection = run_recap_search_candidate_selection(validation, search=recap_search)
            validation = validation.append(recap_selection)
            if recap_selection.selected_candidate_count:
                results = recap_search.results[: recap_selection.selected_candidate_count]
                if len(results) != recap_selection.selected_candidate_count:
                    msg = "RECAP-search result payload is shorter than its selected candidate count"
                    raise ValueError(msg)
                for candidate_index, result in enumerate(results, start=1):
                    candidate = run_recap_search_candidate_evaluation(
                        validation,
                        result=result,
                        candidate_index=candidate_index,
                        depends_on=(recap_selection.node_id,),
                    )
                    validation = validation.append(candidate)
                    validation = await self.run_search_candidate_validation(
                        validation,
                        candidate=candidate,
                        session=session,
                        state=search_state,
                    )
        return validation.append(run_search_citation_summary(validation))

    async def run_search_candidate_validation(
        self,
        validation: CitationValidation,
        *,
        candidate: CandidateEvaluationNode,
        session: MelleaSession | None,
        state: CandidateValidationState,
    ) -> CitationValidation:
        """Complete the reusable field-check subtree for one selected search candidate.

        The locator-not-found route already re-extracted citation-local parties
        before searching. A selected result therefore receives exact
        then semantic case-name checks, but never a second local re-extraction.

        Graph:
            selected search candidate evaluation
            ├── exact case-name check
            │   └── mismatch -> Mellea semantic case-name check
            ├── direct court check
            └── year check
                └── search candidate -> source-specific candidate assessment
        """
        exact_case_name_check = run_exact_case_name_check(validation, candidate=candidate)
        year_check = run_year_check(validation, candidate=candidate)
        court_check = run_court_check(validation, evidence=candidate)
        validation = validation.append(exact_case_name_check).append(year_check).append(court_check)
        state = _with_exact_case_name_result(state, exact_case_name_check)
        if exact_case_name_check.outcome is FieldCheckOutcome.MISMATCH:
            semantic = await run_mellea_case_name_check(
                validation,
                case_name_evidence=exact_case_name_check,
                session=session,
            )
            validation = validation.append(semantic)
            state = state.with_case_name_result(
                outcome=_aggregated_mellea_outcome(semantic.outcome),
                evidence="mellea",
                dependency_id=semantic.node_id,
            )
        if candidate.source is CandidateEvaluationSource.OPINION_SEARCH:
            return validation.append(
                run_opinion_search_candidate_assessment(
                    validation,
                    candidate=candidate,
                    state=state,
                )
            )
        if candidate.source is CandidateEvaluationSource.RECAP_SEARCH:
            return validation.append(
                run_recap_search_candidate_assessment(
                    validation,
                    candidate=candidate,
                    state=state,
                )
            )
        msg = "Search-candidate validation requires an opinion- or RECAP-search candidate"
        raise ValueError(msg)

    async def run_locator_ambiguous(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
        document_text: str,
        session: MelleaSession | None,
    ) -> CitationValidation:
        """Run the complete current graph rooted in an ambiguous locator.

        Graph:
            ambiguous locator
            └── candidate-selection guard
                ├── deferred over limit -> end
                └── candidate evaluation x selected candidate
                    └── ``run_locator_candidate_validation``
                        └── locator citation summary -> locator identity resolution
        """
        if lookup.outcome is not LocatorLookupOutcome.AMBIGUOUS:
            msg = "run_locator_ambiguous requires an ambiguous locator"
            raise ValueError(msg)
        selection = run_locator_candidate_selection(validation, lookup=lookup)
        validation = validation.append(selection)
        if not selection.selected_candidate_count:
            return validation.append(
                run_future_implementation_deferred_locator_identity_resolution(
                    validation,
                    depends_on=(selection.node_id,),
                    reason=(
                        f"{selection.total_candidate_count} exact-locator candidates meet or exceed the "
                        f"review limit of {selection.selection_limit}."
                    ),
                )
            )
        candidates = lookup.candidate_clusters[: selection.selected_candidate_count]
        if len(candidates) != selection.selected_candidate_count:
            msg = "Locator candidate payload is shorter than its selected candidate count"
            raise ValueError(msg)
        for candidate_index, cluster in enumerate(candidates, start=1):
            candidate = run_locator_candidate_evaluation(
                validation,
                cluster=cluster,
                candidate_index=candidate_index,
                depends_on=(selection.node_id,),
            )
            validation = validation.append(candidate)
            validation = await self.run_locator_candidate_validation(
                validation,
                lookup=lookup,
                candidate=candidate,
                document_text=document_text,
                session=session,
                state=CandidateValidationState(),
            )
        summary = run_locator_citation_summary(validation)
        validation = validation.append(summary)
        return await _resolve_locator_identity_from_summary(
            validation,
            summary=summary,
            document_text=document_text,
            session=session,
        )


async def _resolve_locator_identity_from_summary(
    validation: CitationValidation,
    *,
    summary: LocatorCitationSummaryNode,
    document_text: str,
    session: MelleaSession | None,
) -> CitationValidation:
    """Apply deterministic identity or one grounded selection to a summary."""
    if not requires_mellea_locator_candidate_choice(summary):
        return validation.append(run_locator_identity_resolution(validation, summary=summary))
    choice = await run_mellea_locator_candidate_choice(
        validation,
        summary=summary,
        document_text=document_text,
        session=session,
    )
    validation = validation.append(choice)
    return validation.append(run_locator_identity_resolution(validation, summary=summary, choice=choice))


def _with_exact_case_name_result(
    state: CandidateValidationState,
    exact: ExactCaseNameCheckNode,
) -> CandidateValidationState:
    """Record the deterministic case-name result before optional recovery."""
    return state.with_case_name_result(
        outcome=AggregatedFieldOutcome(exact.outcome.value),
        evidence="exact",
        dependency_id=exact.node_id,
    )


def _aggregated_mellea_outcome(
    outcome: MelleaCaseNameCheckOutcome,
) -> AggregatedFieldOutcome:
    """Project a semantic check into candidate-assessment vocabulary."""
    if outcome is MelleaCaseNameCheckOutcome.MATCH:
        return AggregatedFieldOutcome.MATCH
    if outcome is MelleaCaseNameCheckOutcome.MISMATCH:
        return AggregatedFieldOutcome.MISMATCH
    if outcome is MelleaCaseNameCheckOutcome.UNAVAILABLE:
        return AggregatedFieldOutcome.UNAVAILABLE
    return AggregatedFieldOutcome.FAILED
