"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.validation.aggregation import (
    run_locator_candidate_assessment,
    run_locator_citation_summary,
    run_opinion_search_candidate_assessment,
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
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameReextractionOutcome,
    OpinionSearchOutcome,
    RecapSearchOutcome,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


@dataclass(frozen=True, slots=True)
class CitationValidationRunner:
    """Run one citation with bound validation dependencies."""

    client: CourtListenerServiceClient

    async def run_validation(
        self,
        validation: CitationValidation,
        *,
        document_text: str,
        session: MelleaSession | None = None,
    ) -> CitationValidation:
        """Run the complete top-level locator-outcome graph.

        Graph:
            exact locator lookup
            ├── found -> ``run_locator_found``
            ├── not found -> ``run_locator_not_found``
            ├── ambiguous -> ``run_locator_ambiguous``
            └── unsupported, incomplete, or failed -> end
        """
        exact_locator_lookup_node = run_exact_locator_lookup(validation, client=self.client)
        validation = validation.append(exact_locator_lookup_node)

        if exact_locator_lookup_node.outcome is LocatorLookupOutcome.FOUND:
            return await self.run_locator_found(
                validation,
                lookup=exact_locator_lookup_node,
                document_text=document_text,
                session=session,
            )
        if exact_locator_lookup_node.outcome is LocatorLookupOutcome.NOT_FOUND:
            return await self.run_locator_not_found(
                validation,
                lookup=exact_locator_lookup_node,
                document_text=document_text,
                session=session,
            )
        if exact_locator_lookup_node.outcome is LocatorLookupOutcome.AMBIGUOUS:
            return await self.run_locator_ambiguous(validation, lookup=exact_locator_lookup_node)
        return validation

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
                │   ├── exact case-name mismatch ->
                │   │   ``run_locator_found_case_name_mismatch``
                │   └── match or unavailable -> locator candidate assessment
                ├── docket court retrieval -> court check
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
        if exact_case_name_check_node.outcome is FieldCheckOutcome.MISMATCH:
            validation = await self.run_locator_found_case_name_mismatch(
                validation,
                lookup=lookup,
                candidate=candidate,
                exact_case_name_check=exact_case_name_check_node,
                document_text=document_text,
                session=session,
            )
        assessment = run_locator_candidate_assessment(validation, candidate=candidate)
        validation = validation.append(assessment)
        return validation.append(run_locator_citation_summary(validation, assessment=assessment))

    async def run_locator_found_case_name_mismatch(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
        candidate: CandidateEvaluationNode,
        exact_case_name_check: ExactCaseNameCheckNode,
        document_text: str,
        session: MelleaSession | None,
    ) -> CitationValidation:
        """Run the complete case-name recovery graph after an exact mismatch.

        Graph:
            exact case-name mismatch
            └── Mellea semantic case-name check
                ├── match or failed -> end
                └── mismatch -> Mellea local party re-extraction
                    ├── complete -> Mellea re-extracted case-name check -> end
                    └── partial, not found, unavailable, or failed -> end
        """
        if exact_case_name_check.outcome is not FieldCheckOutcome.MISMATCH:
            return validation
        semantic = await run_mellea_case_name_check(
            validation,
            case_name_evidence=exact_case_name_check,
            session=session,
        )
        validation = validation.append(semantic)
        if semantic.outcome is not MelleaCaseNameCheckOutcome.MISMATCH:
            return validation
        reextraction = await run_mellea_case_name_reextraction(
            validation,
            trigger=semantic,
            locator_lookup=lookup,
            document_text=document_text,
            session=session,
        )
        validation = validation.append(reextraction)
        if reextraction.outcome is not MelleaCaseNameReextractionOutcome.COMPLETE:
            return validation
        return validation.append(
            await run_mellea_case_name_check(
                validation,
                case_name_evidence=reextraction,
                candidate=candidate,
                session=session,
            )
        )

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
                    )
        return validation

    async def run_search_candidate_validation(
        self,
        validation: CitationValidation,
        *,
        candidate: CandidateEvaluationNode,
        session: MelleaSession | None,
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
                └── opinion-search candidate -> candidate assessment
        """
        exact_case_name_check = run_exact_case_name_check(validation, candidate=candidate)
        year_check = run_year_check(validation, candidate=candidate)
        court_check = run_court_check(validation, evidence=candidate)
        validation = validation.append(exact_case_name_check).append(year_check).append(court_check)
        if exact_case_name_check.outcome is FieldCheckOutcome.MISMATCH:
            validation = validation.append(
                await run_mellea_case_name_check(
                    validation,
                    case_name_evidence=exact_case_name_check,
                    session=session,
                )
            )
        if candidate.source is CandidateEvaluationSource.OPINION_SEARCH:
            return validation.append(run_opinion_search_candidate_assessment(validation, candidate=candidate))
        return validation

    async def run_locator_ambiguous(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
    ) -> CitationValidation:
        """Run the complete current graph rooted in an ambiguous locator.

        Graph:
            ambiguous locator
            └── candidate-selection guard -> candidate evaluation x selected candidate

        A later ``run_locator_ambiguous_*`` decomposition will extend this
        route without changing the top-level progression selector.
        """
        if lookup.outcome is not LocatorLookupOutcome.AMBIGUOUS:
            msg = "run_locator_ambiguous requires an ambiguous locator"
            raise ValueError(msg)
        selection = run_locator_candidate_selection(validation, lookup=lookup)
        validation = validation.append(selection)
        if not selection.selected_candidate_count:
            return validation
        candidates = lookup.candidate_clusters[: selection.selected_candidate_count]
        if len(candidates) != selection.selected_candidate_count:
            msg = "Locator candidate payload is shorter than its selected candidate count"
            raise ValueError(msg)
        for candidate_index, cluster in enumerate(candidates, start=1):
            validation = validation.append(
                run_locator_candidate_evaluation(
                    validation,
                    cluster=cluster,
                    candidate_index=candidate_index,
                    depends_on=(selection.node_id,),
                )
            )
        return validation
