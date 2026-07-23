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
    FieldCheckOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameReextractionOutcome,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


@dataclass(frozen=True, slots=True)
class CitationValidationRunner:
    """Run one citation with bound validation dependencies."""

    client: CourtListenerServiceClient

    def run(self, validation: CitationValidation) -> CitationValidation:
        """Run the explicitly supported locator-found progression."""
        exact_locator_lookup_node = run_exact_locator_lookup(validation, client=self.client)
        validation = validation.append(exact_locator_lookup_node)

        if exact_locator_lookup_node.outcome is LocatorLookupOutcome.FOUND:
            exact_case_name_check_node = run_exact_case_name_check(
                validation,
                lookup=exact_locator_lookup_node,
            )
            year_check_node = run_year_check(validation, lookup=exact_locator_lookup_node)
            validation = validation.append(exact_case_name_check_node).append(year_check_node)

        return validation

    async def run_with_mellea(
        self,
        validation: CitationValidation,
        *,
        document_text: str,
        session: object | None = None,
    ) -> CitationValidation:
        """Run the explicit semantic-mismatch recovery progression."""
        validation = self.run(validation)
        exact = next(node for node in reversed(validation.nodes) if isinstance(node, ExactCaseNameCheckNode))
        if exact.outcome is not FieldCheckOutcome.MISMATCH:
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
