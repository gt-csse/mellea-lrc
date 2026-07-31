"""Branch-local state carried through explicit candidate validation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea_lrc.validation.types import (
        AggregatedFieldOutcome,
        MelleaCaseNameReextractionNode,
    )


@dataclass(frozen=True, slots=True)
class CaseNameValidationState:
    """Final case-name conclusion for the current candidate branch."""

    outcome: AggregatedFieldOutcome
    evidence: str
    dependency_id: str


@dataclass(frozen=True, slots=True)
class CandidateValidationState:
    """Explicit state carried through one candidate-validation branch.

    Currently this state contains only re-extracted case-name evidence. Its
    eventual scope needs deliberate design: it may belong to a whole citation
    or to each citation candidate, and other cross-node state may need to join
    it as validation progression grows.
    """

    reextracted_case_name: str | None = None
    case_name: CaseNameValidationState | None = None

    def with_reextraction(
        self,
        node: MelleaCaseNameReextractionNode,
    ) -> CandidateValidationState:
        """Record the locally grounded case name when re-extraction succeeds."""
        if node.plaintiff is None or node.defendant is None:
            return self
        return CandidateValidationState(
            reextracted_case_name=f"{node.plaintiff} v. {node.defendant}",
            case_name=self.case_name,
        )

    def with_case_name_result(
        self,
        *,
        outcome: AggregatedFieldOutcome,
        evidence: str,
        dependency_id: str,
    ) -> CandidateValidationState:
        """Record the completed case-name conclusion selected by execution."""
        return CandidateValidationState(
            reextracted_case_name=self.reextracted_case_name,
            case_name=CaseNameValidationState(
                outcome=outcome,
                evidence=evidence,
                dependency_id=dependency_id,
            ),
        )

    def require_case_name_result(self) -> CaseNameValidationState:
        """Return the case-name result once this branch completed its checks."""
        if self.case_name is None:
            msg = "Candidate validation state is missing its case-name result"
            raise ValueError(msg)
        return self.case_name

    def displayed_case_name(self, extracted_case_name: str | None) -> str | None:
        """Prefer locally re-extracted evidence in aggregation presentation."""
        return self.reextracted_case_name or extracted_case_name
