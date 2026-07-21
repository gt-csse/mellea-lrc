"""Explicit per-citation validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup
from mellea_lrc.validation.field_checks import run_case_name_check, run_year_check
from mellea_lrc.validation.field_checks.case_name_check import run_rechecked_case_name
from mellea_lrc.validation.field_checks.mellea_case_name_check import mellea_case_names_match
from mellea_lrc.validation.field_checks.mellea_case_name_reextract import (
    run_mellea_case_name_reextract,
)
from mellea_lrc.validation.types import (
    CaseNameCheckNode,
    CaseNameCheckOutcome,
    CaseNameReextractionNode,
    CaseNameReextractionOutcome,
    CaseSearchNode,
    CaseSearchOutcome,
    CitationValidation,
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    RecheckedCaseNameNode,
    ValidationNode,
    ValidationNodeStatus,
    YearCheckNode,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient

NodeT = TypeVar("NodeT", bound=ValidationNode)


@dataclass(frozen=True, slots=True)
class CitationValidationRunner:
    """Run one citation while keeping runtime dependencies out of progression logic."""

    client: CourtListenerServiceClient
    document_text: str
    mellea_session: MelleaSession | None = None

    async def run(self, validation: CitationValidation) -> CitationValidation:
        """Run one citation through the explicitly supported progressions."""
        lookup = await self._exact_locator_lookup(validation)
        validation = validation.append(lookup)

        if lookup.outcome is LocatorLookupOutcome.FOUND:
            case_name = await self._case_name_check(validation)
            year = await self._year_check(validation)
            validation = validation.append(case_name).append(year)

            if case_name.outcome is CaseNameCheckOutcome.NOT_SEMANTIC_MATCH:
                reextraction = await self._case_name_reextract(validation)
                validation = validation.append(reextraction)

                if reextraction.outcome is CaseNameReextractionOutcome.ACCEPTED:
                    recheck = await self._case_name_recheck(validation)
                    validation = validation.append(recheck)

        elif lookup.outcome is LocatorLookupOutcome.NOT_FOUND:
            reextraction = await self._case_name_reextract(validation)
            validation = validation.append(reextraction)

            if reextraction.outcome is CaseNameReextractionOutcome.ACCEPTED:
                search = self._case_search_placeholder(validation)
                validation = validation.append(search)

        return validation

    async def _exact_locator_lookup(
        self,
        validation: CitationValidation,
    ) -> ExactLocatorLookupNode:
        return await run_exact_locator_lookup(validation, client=self.client)

    async def _case_name_check(
        self,
        validation: CitationValidation,
    ) -> CaseNameCheckNode:
        return await run_case_name_check(
            validation,
            lookup=_latest(validation, ExactLocatorLookupNode),
            semantic_matcher=self._semantic_match,
        )

    async def _year_check(self, validation: CitationValidation) -> YearCheckNode:
        return await run_year_check(
            validation,
            lookup=_latest(validation, ExactLocatorLookupNode),
        )

    async def _case_name_reextract(
        self,
        validation: CitationValidation,
    ) -> CaseNameReextractionNode:
        return await run_mellea_case_name_reextract(
            validation,
            document_text=self.document_text,
            session=self.mellea_session,
        )

    async def _case_name_recheck(
        self,
        validation: CitationValidation,
    ) -> RecheckedCaseNameNode:
        return await run_rechecked_case_name(
            validation,
            reextraction=_latest(validation, CaseNameReextractionNode),
            semantic_matcher=self._semantic_match,
        )

    def _case_search_placeholder(self, validation: CitationValidation) -> CaseSearchNode:
        lookup = _latest(validation, ExactLocatorLookupNode)
        reextraction = _latest(validation, CaseNameReextractionNode)
        return CaseSearchNode(
            node_id=f"{validation.citation_id}:case_search",
            status=ValidationNodeStatus.SKIPPED,
            outcome=CaseSearchOutcome.NOT_IMPLEMENTED,
            depends_on=(lookup.node_id, reextraction.node_id),
        )

    async def _semantic_match(self, extracted: str, retrieved: str) -> bool:
        return await mellea_case_names_match(
            extracted,
            retrieved,
            session=self.mellea_session,
        )


def _latest(
    validation: CitationValidation,
    node_type: type[NodeT],
) -> NodeT:
    try:
        return next(node for node in reversed(validation.nodes) if isinstance(node, node_type))
    except StopIteration as exc:
        msg = f"Validation requires prior {node_type.__name__}"
        raise RuntimeError(msg) from exc
