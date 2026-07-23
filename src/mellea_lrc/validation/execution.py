"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup
from mellea_lrc.validation.field_checks import (
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
        """Run one citation through its explicit locator-outcome progression."""
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
        """Run the progression rooted in one uniquely resolved locator."""
        if lookup.outcome is not LocatorLookupOutcome.FOUND:
            msg = "run_locator_found requires a found locator"
            raise ValueError(msg)
        exact_case_name_check_node = run_exact_case_name_check(validation, lookup=lookup)
        year_check_node = run_year_check(validation, lookup=lookup)
        validation = validation.append(exact_case_name_check_node).append(year_check_node)
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
        """Recover an exact case-name mismatch through semantic evidence."""
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
        """Record a locator miss until its dedicated progression is introduced."""
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
        """Record ambiguous locator evidence until candidate handling is introduced."""
        if lookup.outcome is not LocatorLookupOutcome.AMBIGUOUS:
            msg = "run_locator_ambiguous requires an ambiguous locator"
            raise ValueError(msg)
        return validation
