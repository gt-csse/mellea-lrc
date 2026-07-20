"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup
from mellea_lrc.validation.field_checks import run_case_name_check, run_year_check
from mellea_lrc.validation.field_checks.mellea_case_name import mellea_case_names_match
from mellea_lrc.validation.types import LocatorLookupOutcome

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


@dataclass(frozen=True, slots=True)
class CitationValidationRunner:
    """Run one citation with bound validation dependencies."""

    client: CourtListenerServiceClient
    mellea_session: MelleaSession | None = None

    async def run(self, validation: CitationValidation) -> CitationValidation:
        """Run the explicitly supported locator-found progression."""
        lookup = await run_exact_locator_lookup(validation, client=self.client)
        validation = validation.append(lookup)

        if lookup.outcome is LocatorLookupOutcome.FOUND:
            case_name = await run_case_name_check(
                validation,
                lookup=lookup,
                semantic_matcher=self._semantic_match,
            )
            year = await run_year_check(validation, lookup=lookup)
            validation = validation.append(case_name).append(year)

        return validation

    async def _semantic_match(self, extracted: str, retrieved: str) -> bool:
        return await mellea_case_names_match(
            extracted,
            retrieved,
            session=self.mellea_session,
        )
