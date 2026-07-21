"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


@dataclass(frozen=True, slots=True)
class CitationValidationRunner:
    """Run one citation with bound validation dependencies."""

    client: CourtListenerServiceClient

    def run(self, validation: CitationValidation) -> CitationValidation:
        """Append the only validation node implemented in this slice."""
        lookup = run_exact_locator_lookup(validation, client=self.client)
        return validation.append(lookup)
