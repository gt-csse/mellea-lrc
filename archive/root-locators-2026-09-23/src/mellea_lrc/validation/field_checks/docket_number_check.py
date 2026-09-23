"""Grounded comparison of the docket number stated in a filing and retrieved record."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.llm.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citations import DocketCitation
from mellea_lrc.model.fuzziness import FuzzinessOption
from mellea_lrc.validation.types import (
    CandidateEvaluationSource,
    DocketNumberCheckNode,
    FieldCheckOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CandidateEvaluationNode, CitationValidation


_DOCKET_FUZZINESS = FuzzinessOption.whitespace_relaxation()
_BODY_SEARCH_SOURCES = frozenset(
    {
        CandidateEvaluationSource.COURTLISTENER_CLUSTER_BODY_SEARCH,
        CandidateEvaluationSource.COURTLISTENER_DOCKET_BODY_SEARCH,
        CandidateEvaluationSource.GOVINFO_BODY_SEARCH,
    }
)


def run_docket_number_check(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
) -> DocketNumberCheckNode:
    """Compare docket numbers without normalizing or modifying the filing text.

    The comparison accepts literal equality and whitespace-only layout damage.
    Punctuation, office prefixes, type codes, sequence numbers, and judge
    suffixes remain evidence: treating them as interchangeable would turn this
    identity check into a second extraction grammar.
    """
    citation = validation.citation.fields
    extracted = citation.docket_number if isinstance(citation, DocketCitation) else None
    retrieved = candidate.docket_number or _body_locator(candidate)
    if extracted is None or retrieved is None:
        status = ValidationNodeStatus.SKIPPED
        outcome = FieldCheckOutcome.UNAVAILABLE
        status_message = "Skipped docket-number comparison because required evidence is missing."
        outcome_message = "Docket-number comparison is unavailable because one docket number is missing."
    else:
        match = GroundingEvidence((EvidenceCandidate(text=retrieved, value=retrieved),)).resolve(
            extracted,
            _DOCKET_FUZZINESS,
        )
        status = ValidationNodeStatus.SUCCEEDED
        outcome = FieldCheckOutcome.MATCH if match is not None else FieldCheckOutcome.MISMATCH
        status_message = "Docket-number comparison completed."
        outcome_message = (
            "Stated and retrieved docket numbers match under whitespace relaxation."
            if match is not None
            else "Stated and retrieved docket numbers differ."
        )
    return DocketNumberCheckNode(
        node_id=f"{candidate.node_id}:docket_number_check",
        status=status,
        outcome=outcome,
        extracted_docket_number=extracted,
        retrieved_docket_number=retrieved,
        depends_on=(candidate.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
    )


def _body_locator(candidate: CandidateEvaluationNode) -> str | None:
    """Use the literal query as evidence only for a saved body-search hit.

    Opinion-cluster results do not always expose the docket in metadata.  The
    search node nevertheless records that this returned body was hit by the
    literal source locator.  A metadata result without ``docketNumber`` stays
    unavailable as before.
    """
    if candidate.source not in _BODY_SEARCH_SOURCES:
        return None
    value = candidate.record.get("body_locator")
    return value if isinstance(value, str) and value else None
