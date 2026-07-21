"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup
from mellea_lrc.validation.field_checks import run_exact_case_name_check, run_year_check
from mellea_lrc.validation.types import LocatorLookupOutcome

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
