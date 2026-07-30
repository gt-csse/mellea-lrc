"""JSON round-trip support for validated documents."""

from __future__ import annotations

from collections.abc import Mapping

from mellea_lrc.courtlistener import CourtListenerOpinionCluster, CourtListenerSearchResult
from mellea_lrc.serialization._json import JsonValue, require_list, require_mapping, serialize_dataclass
from mellea_lrc.serialization.extracted_document import (
    SCHEMA_VERSION,
    _require_artifact,
    deserialize_extracted_document,
    serialize_extracted_document,
)
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationOutcome,
    CandidateEvaluationSource,
    CandidateSelectionNode,
    CandidateSelectionOutcome,
    CitationValidation,
    CourtCheckNode,
    DocketCourtRetrievalNode,
    DocketCourtRetrievalOutcome,
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorCandidateAssessmentNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorCitationSummaryOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameCheckNode,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameQueryPreparationNode,
    MelleaCaseNameQueryPreparationOutcome,
    MelleaCaseNameReextractionNode,
    MelleaCaseNameReextractionOutcome,
    MelleaReextractedCaseNameCheckNode,
    OpinionSearchCandidateAssessmentNode,
    OpinionSearchNode,
    OpinionSearchOutcome,
    RecapSearchNode,
    RecapSearchOutcome,
    SearchCandidateAssessmentOutcome,
    ValidatedDocument,
    ValidationNode,
    ValidationNodeStatus,
    YearCheckNode,
)

_ARTIFACT_TYPE = "validated_document"

_NODE_TYPES: dict[str, type[ValidationNode]] = {
    node_type.__name__: node_type
    for node_type in (
        ExactLocatorLookupNode,
        ExactCaseNameCheckNode,
        MelleaCaseNameCheckNode,
        MelleaCaseNameReextractionNode,
        MelleaCaseNameQueryPreparationNode,
        OpinionSearchNode,
        RecapSearchNode,
        CandidateSelectionNode,
        CandidateEvaluationNode,
        MelleaReextractedCaseNameCheckNode,
        DocketCourtRetrievalNode,
        CourtCheckNode,
        LocatorCandidateAssessmentNode,
        LocatorCitationSummaryNode,
        OpinionSearchCandidateAssessmentNode,
        YearCheckNode,
    )
}

_OUTCOME_TYPES = {
    ExactLocatorLookupNode: LocatorLookupOutcome,
    ExactCaseNameCheckNode: FieldCheckOutcome,
    MelleaCaseNameCheckNode: MelleaCaseNameCheckOutcome,
    MelleaCaseNameReextractionNode: MelleaCaseNameReextractionOutcome,
    MelleaCaseNameQueryPreparationNode: MelleaCaseNameQueryPreparationOutcome,
    OpinionSearchNode: OpinionSearchOutcome,
    RecapSearchNode: RecapSearchOutcome,
    CandidateSelectionNode: CandidateSelectionOutcome,
    CandidateEvaluationNode: CandidateEvaluationOutcome,
    MelleaReextractedCaseNameCheckNode: MelleaCaseNameCheckOutcome,
    DocketCourtRetrievalNode: DocketCourtRetrievalOutcome,
    CourtCheckNode: FieldCheckOutcome,
    LocatorCandidateAssessmentNode: LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode: LocatorCitationSummaryOutcome,
    OpinionSearchCandidateAssessmentNode: SearchCandidateAssessmentOutcome,
    YearCheckNode: FieldCheckOutcome,
}


def serialize_validated_document(document: ValidatedDocument) -> dict[str, JsonValue]:
    """Project one ``ValidatedDocument`` into a recoverable JSON artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "source": serialize_extracted_document(document.source),
        "citations": [
            {
                "citation_id": progression.citation_id,
                "nodes": [
                    {
                        "node_type": type(node).__name__,
                        **serialize_dataclass(node),
                    }
                    for node in progression.nodes
                ],
            }
            for progression in document.citations
        ],
    }


def deserialize_validated_document(payload: Mapping[str, object]) -> ValidatedDocument:
    """Recover one ``ValidatedDocument`` and its explicit node graph from JSON."""
    _require_artifact(payload, artifact_type=_ARTIFACT_TYPE)
    source = deserialize_extracted_document(require_mapping(payload.get("source"), name="source"))
    progressions = require_list(payload.get("citations"), name="citations")
    if len(progressions) != len(source.citations):
        msg = "Validation progressions must exactly match extracted citations"
        raise ValueError(msg)

    citations: list[CitationValidation] = []
    for source_citation, value in zip(source.citations, progressions, strict=True):
        progression = require_mapping(value, name="citation progression")
        citation_id = progression.get("citation_id")
        if citation_id != source_citation.citation_id:
            msg = "Validation progressions must preserve extracted citation order"
            raise ValueError(msg)
        validation = CitationValidation(citation=source_citation)
        for node_payload in require_list(progression.get("nodes"), name="citation progression.nodes"):
            validation = validation.append(_deserialize_node(node_payload))
        citations.append(validation)
    return ValidatedDocument(source=source, citations=tuple(citations))


def _deserialize_node(value: object) -> ValidationNode:
    payload = require_mapping(value, name="validation node")
    node_type_name = payload.get("node_type")
    if not isinstance(node_type_name, str) or node_type_name not in _NODE_TYPES:
        msg = f"Unknown validation node type: {node_type_name!r}"
        raise ValueError(msg)
    node_type = _NODE_TYPES[node_type_name]
    fields = {key: value for key, value in payload.items() if key != "node_type"}
    fields["status"] = ValidationNodeStatus(fields["status"])
    fields["outcome"] = _OUTCOME_TYPES[node_type](fields["outcome"])
    fields["depends_on"] = tuple(require_list(fields["depends_on"], name="node.depends_on"))

    if node_type is ExactLocatorLookupNode:
        fields["cluster"] = _deserialize_cluster(fields["cluster"]) if fields["cluster"] is not None else None
        fields["candidate_clusters"] = tuple(
            _deserialize_cluster(item)
            for item in require_list(fields["candidate_clusters"], name="node.candidate_clusters")
        )
    elif node_type in (OpinionSearchNode, RecapSearchNode):
        fields["results"] = _freeze_search_results(require_list(fields["results"], name="node.results"))
    elif node_type is CandidateEvaluationNode:
        fields["source"] = CandidateEvaluationSource(fields["source"])
        if fields["source"] is CandidateEvaluationSource.LOCATOR_LOOKUP:
            fields["record"] = _deserialize_cluster(fields["record"])
        else:
            fields["record"] = _freeze_search_results([fields["record"]])[0]
    elif node_type in (LocatorCandidateAssessmentNode, OpinionSearchCandidateAssessmentNode):
        for field_name in ("case_name_outcome", "year_outcome", "court_outcome"):
            fields[field_name] = AggregatedFieldOutcome(fields[field_name])
    return node_type(**fields)


def _deserialize_cluster(value: object) -> CourtListenerOpinionCluster:
    payload = require_mapping(value, name="opinion cluster")
    return CourtListenerOpinionCluster(
        cluster_id=_optional_string(payload.get("cluster_id"), name="cluster.cluster_id"),
        case_name=_optional_string(payload.get("case_name"), name="cluster.case_name"),
        date_filed=_optional_string(payload.get("date_filed"), name="cluster.date_filed"),
        court=_optional_string(payload.get("court"), name="cluster.court"),
        court_id=_optional_string(payload.get("court_id"), name="cluster.court_id"),
        docket_id=_optional_string(payload.get("docket_id"), name="cluster.docket_id"),
    )


def _freeze_search_results(values: list[object]) -> tuple[Mapping[str, object], ...]:
    records = [dict(require_mapping(value, name="search result")) for value in values]
    return CourtListenerSearchResult.from_payload(
        query="",
        search_type="",
        semantic=False,
        count=len(records),
        results=records,
        next_cursor=None,
        previous_cursor=None,
    ).results


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{name} must be a string or null"
        raise ValueError(msg)
    return value
