"""Materialize selected retrieved candidates for later validation subtrees."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationOutcome,
    CandidateEvaluationSource,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationRecord
    from mellea_lrc.validation.types import CitationValidation


def run_locator_candidate_evaluation(
    validation: CitationValidation,
    *,
    record: CourtListenerCitationRecord,
    candidate_index: int,
    depends_on: tuple[str, ...],
) -> CandidateEvaluationNode:
    """Materialize one locator result for a reusable validation subtree."""
    return CandidateEvaluationNode(
        node_id=f"{validation.citation_id}:locator_candidate_evaluation:{candidate_index}",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=CandidateEvaluationOutcome.READY,
        source=CandidateEvaluationSource.LOCATOR_LOOKUP,
        candidate_index=candidate_index,
        cluster_id=record.cluster_id,
        case_name=record.case_name,
        date_filed=record.date_filed,
        court_id=record.court_id,
        docket_id=record.docket_id,
        record=record,
        depends_on=depends_on,
        status_message="Locator candidate evaluation branch initialized.",
        outcome_message="Candidate is ready for independent validation checks.",
    )


def run_opinion_search_candidate_evaluation(
    validation: CitationValidation,
    *,
    result: Mapping[str, object],
    candidate_index: int,
    depends_on: tuple[str, ...],
) -> CandidateEvaluationNode:
    """Materialize one selected opinion-search result for later field checks."""
    return _search_candidate_evaluation(
        validation,
        result=result,
        candidate_index=candidate_index,
        depends_on=depends_on,
        source=CandidateEvaluationSource.OPINION_SEARCH,
    )


def run_recap_search_candidate_evaluation(
    validation: CitationValidation,
    *,
    result: Mapping[str, object],
    candidate_index: int,
    depends_on: tuple[str, ...],
) -> CandidateEvaluationNode:
    """Materialize one selected RECAP-search result for later field checks."""
    return _search_candidate_evaluation(
        validation,
        result=result,
        candidate_index=candidate_index,
        depends_on=depends_on,
        source=CandidateEvaluationSource.RECAP_SEARCH,
    )


def _search_candidate_evaluation(
    validation: CitationValidation,
    *,
    result: Mapping[str, object],
    candidate_index: int,
    depends_on: tuple[str, ...],
    source: CandidateEvaluationSource,
) -> CandidateEvaluationNode:
    """Materialize one result from a CourtListener search endpoint."""
    source_label = source.value.replace("_", " ").capitalize()
    return CandidateEvaluationNode(
        node_id=f"{validation.citation_id}:{source.value}_candidate_evaluation:{candidate_index}",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=CandidateEvaluationOutcome.READY,
        source=source,
        candidate_index=candidate_index,
        cluster_id=_optional_string(result.get("cluster_id")),
        case_name=_optional_string(result.get("caseName")),
        date_filed=_optional_string(result.get("dateFiled")),
        court_id=_optional_string(result.get("court_id")),
        docket_id=_optional_string(result.get("docket_id")),
        record=result,
        depends_on=depends_on,
        status_message=f"{source_label} candidate evaluation branch initialized.",
        outcome_message="Candidate is ready for independent validation checks.",
    )


def _optional_string(value: object) -> str | None:
    """Normalize scalar CourtListener values while preserving missing values."""
    return str(value) if isinstance(value, str | int) else None
