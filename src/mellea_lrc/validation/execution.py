"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameReextractionOutcome,
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
            return await self.run_locator_not_found(validation, lookup=exact_locator_lookup_node)
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
            ├── exact case-name check + year check + docket court retrieval
            │   ├── exact case-name mismatch ->
            │   │   ``run_locator_found_case_name_mismatch``
            │   └── match or unavailable -> end
            ├── docket court retrieval -> court check
            └── year and court results do not alter this progression yet
        """
        if lookup.outcome is not LocatorLookupOutcome.FOUND:
            msg = "run_locator_found requires a found locator"
            raise ValueError(msg)
        exact_case_name_check_node = run_exact_case_name_check(validation, lookup=lookup)
        year_check_node = run_year_check(validation, lookup=lookup)
        docket_court_retrieval_node = run_docket_court_retrieval(
            validation,
            lookup=lookup,
            client=self.client,
        )
        court_check_node = run_court_check(validation, retrieval=docket_court_retrieval_node)
        validation = (
            validation.append(exact_case_name_check_node)
            .append(year_check_node)
            .append(docket_court_retrieval_node)
            .append(court_check_node)
        )
        if exact_case_name_check_node.outcome is not FieldCheckOutcome.MISMATCH:
            return validation
        return await self.run_locator_found_case_name_mismatch(
            validation,
            exact_case_name_check=exact_case_name_check_node,
            document_text=document_text,
            session=session,
        )

    async def run_locator_found_case_name_mismatch(
        self,
        validation: CitationValidation,
        *,
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
        semantic = await run_mellea_case_name_check(validation, session=session)
        validation = validation.append(semantic)
        if semantic.outcome is not MelleaCaseNameCheckOutcome.MISMATCH:
            return validation
        reextraction = await run_mellea_case_name_reextraction(
            validation, document_text=document_text, session=session
        )
        validation = validation.append(reextraction)
        if reextraction.outcome is not MelleaCaseNameReextractionOutcome.COMPLETE:
            return validation
        return validation.append(await run_mellea_case_name_check(validation, session=session))

    async def run_locator_not_found(
        self,
        validation: CitationValidation,
        *,
        lookup: ExactLocatorLookupNode,
    ) -> CitationValidation:
        """Run the complete current graph rooted in a locator miss.

        Graph:
            locator not found
            └── end

        A later ``run_locator_not_found_*`` decomposition will extend this
        route without changing the top-level progression selector.
        """
        if lookup.outcome is not LocatorLookupOutcome.NOT_FOUND:
            msg = "run_locator_not_found requires a not-found locator"
            raise ValueError(msg)
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
            └── end

        A later ``run_locator_ambiguous_*`` decomposition will extend this
        route without changing the top-level progression selector.
        """
        if lookup.outcome is not LocatorLookupOutcome.AMBIGUOUS:
            msg = "run_locator_ambiguous requires an ambiguous locator"
            raise ValueError(msg)
        return validation
