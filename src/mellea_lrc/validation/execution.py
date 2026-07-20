"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup
from mellea_lrc.validation.field_checks import run_case_name_check, run_year_check
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
        lookup = run_exact_locator_lookup(validation, client=self.client)
        validation = validation.append(lookup)

        if lookup.outcome is LocatorLookupOutcome.FOUND:
            case_name = run_case_name_check(validation, lookup=lookup)
            year = run_year_check(validation, lookup=lookup)
            validation = validation.append(case_name).append(year)

        return validation
