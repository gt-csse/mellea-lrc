"""JSON round-trip support for validated documents."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TypeVar

from mellea_lrc.courtlistener import (
    CourtListenerOpinionCluster,
    CourtListenerOpinionClusterCitation,
    CourtListenerSearchResult,
)
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.document import Document
from mellea_lrc.model.evidence import EvidenceQuoteMatchMethod
from mellea_lrc.model.spans import Span
from mellea_lrc.serialization._json import JsonValue, require_list, require_mapping, serialize_dataclass
from mellea_lrc.serialization.ivr import deserialize_ivr_run
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    BodySearchAttempt,
    CandidateEvaluationNode,
    CandidateEvaluationOutcome,
    CandidateEvaluationSource,
    CandidateProvenance,
    CandidateSelectionNode,
    CandidateSelectionOutcome,
    CitationSummaryAssessmentOutcome,
    CitationSummaryCandidate,
    CitationSummaryPinpoint,
    CitationValidation,
    CourtCheckNode,
    DocketCourtRetrievalNode,
    DocketCourtRetrievalOutcome,
    DocketMetadataShortlistCandidate,
    DocketMetadataShortlistNode,
    DocketMetadataShortlistOutcome,
    DocketNumberCheckNode,
    DocketRootSearchNode,
    DocketRootSearchOutcome,
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    FullReporterSearchNode,
    FullReporterSearchOutcome,
    GovInfoDocketSearchNode,
    GovInfoFullReporterSearchNode,
    LocatorCandidateAssessmentNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorCitationSummaryOutcome,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameCheckNode,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameQueryPreparationNode,
    MelleaCaseNameQueryPreparationOutcome,
    MelleaCaseNameReextractionNode,
    MelleaCaseNameReextractionOutcome,
    MelleaCaseNameReviewNode,
    MelleaCitingPropositionExtractionNode,
    MelleaCitingPropositionExtractionOutcome,
    MelleaDocketCitationReextractionNode,
    MelleaDocketCitationReextractionOutcome,
    MelleaDocketNumberEquivalenceNode,
    MelleaDocketNumberEquivalenceOutcome,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    MelleaPinpointCheckNode,
    MelleaPinpointCheckOutcome,
    MelleaReextractedCaseNameCheckNode,
    MetadataSearchAttempt,
    OpinionSearchCandidateAssessmentNode,
    OpinionSearchNode,
    OpinionSearchOutcome,
    RecapSearchCandidateAssessmentNode,
    RecapSearchNode,
    RecapSearchOutcome,
    ReporterPageEvidence,
    ReporterPageRetrievalNode,
    ReporterPageRetrievalOutcome,
    RootBodySearchNode,
    RootBodySearchOutcome,
    RootBodySearchSource,
    SearchCandidateAssessmentOutcome,
    SearchCitationSummaryNode,
    SearchCitationSummaryOutcome,
    ValidatedDocument,
    ValidationNode,
    ValidationNodeStatus,
    YearCheckNode,
)

EnumT = TypeVar("EnumT", bound=Enum)

_ARTIFACT_TYPE = "validated_document"
SCHEMA_VERSION = 21

_NODE_TYPES: dict[str, type[ValidationNode]] = {
    node_type.__name__: node_type
    for node_type in (
        ExactLocatorLookupNode,
        DocketRootSearchNode,
        FullReporterSearchNode,
        DocketMetadataShortlistNode,
        GovInfoDocketSearchNode,
        GovInfoFullReporterSearchNode,
        RootBodySearchNode,
        MelleaDocketCitationReextractionNode,
        MelleaDocketNumberEquivalenceNode,
        ExactCaseNameCheckNode,
        MelleaCaseNameCheckNode,
        MelleaCaseNameReviewNode,
        MelleaCaseNameReextractionNode,
        MelleaCaseNameQueryPreparationNode,
        OpinionSearchNode,
        RecapSearchNode,
        CandidateSelectionNode,
        CandidateEvaluationNode,
        MelleaReextractedCaseNameCheckNode,
        DocketCourtRetrievalNode,
        DocketNumberCheckNode,
        ReporterPageRetrievalNode,
        MelleaCitingPropositionExtractionNode,
        MelleaPinpointCheckNode,
        CourtCheckNode,
        LocatorCandidateAssessmentNode,
        LocatorCitationSummaryNode,
        MelleaLocatorCandidateChoiceNode,
        LocatorIdentityResolutionNode,
        OpinionSearchCandidateAssessmentNode,
        RecapSearchCandidateAssessmentNode,
        SearchCitationSummaryNode,
        YearCheckNode,
    )
}

_OUTCOME_TYPES = {
    ExactLocatorLookupNode: LocatorLookupOutcome,
    DocketRootSearchNode: DocketRootSearchOutcome,
    FullReporterSearchNode: FullReporterSearchOutcome,
    DocketMetadataShortlistNode: DocketMetadataShortlistOutcome,
    GovInfoDocketSearchNode: DocketRootSearchOutcome,
    GovInfoFullReporterSearchNode: FullReporterSearchOutcome,
    RootBodySearchNode: RootBodySearchOutcome,
    MelleaDocketCitationReextractionNode: MelleaDocketCitationReextractionOutcome,
    MelleaDocketNumberEquivalenceNode: MelleaDocketNumberEquivalenceOutcome,
    ExactCaseNameCheckNode: FieldCheckOutcome,
    MelleaCaseNameCheckNode: MelleaCaseNameCheckOutcome,
    MelleaCaseNameReviewNode: MelleaCaseNameCheckOutcome,
    MelleaCaseNameReextractionNode: MelleaCaseNameReextractionOutcome,
    MelleaCaseNameQueryPreparationNode: MelleaCaseNameQueryPreparationOutcome,
    OpinionSearchNode: OpinionSearchOutcome,
    RecapSearchNode: RecapSearchOutcome,
    CandidateSelectionNode: CandidateSelectionOutcome,
    CandidateEvaluationNode: CandidateEvaluationOutcome,
    MelleaReextractedCaseNameCheckNode: MelleaCaseNameCheckOutcome,
    DocketCourtRetrievalNode: DocketCourtRetrievalOutcome,
    DocketNumberCheckNode: FieldCheckOutcome,
    ReporterPageRetrievalNode: ReporterPageRetrievalOutcome,
    MelleaCitingPropositionExtractionNode: MelleaCitingPropositionExtractionOutcome,
    MelleaPinpointCheckNode: MelleaPinpointCheckOutcome,
    CourtCheckNode: FieldCheckOutcome,
    LocatorCandidateAssessmentNode: LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode: LocatorCitationSummaryOutcome,
    MelleaLocatorCandidateChoiceNode: MelleaLocatorCandidateChoiceOutcome,
    LocatorIdentityResolutionNode: LocatorIdentityResolutionOutcome,
    OpinionSearchCandidateAssessmentNode: SearchCandidateAssessmentOutcome,
    RecapSearchCandidateAssessmentNode: SearchCandidateAssessmentOutcome,
    SearchCitationSummaryNode: SearchCitationSummaryOutcome,
    YearCheckNode: FieldCheckOutcome,
}

_IVR_NODE_TYPES = frozenset(
    {
        MelleaCaseNameCheckNode,
        MelleaCaseNameReviewNode,
        MelleaCaseNameReextractionNode,
        MelleaCaseNameQueryPreparationNode,
        MelleaDocketCitationReextractionNode,
        MelleaDocketNumberEquivalenceNode,
        MelleaReextractedCaseNameCheckNode,
        MelleaLocatorCandidateChoiceNode,
    }
)


def serialize_validated_document(document: ValidatedDocument) -> dict[str, JsonValue]:
    """Project one ``ValidatedDocument`` into a recoverable JSON artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "source": document.source.model_dump(mode="json"),
        "citations": [
            {
                "citation_id": progression.citation_id,
                "aggregation": (
                    {
                        "node_type": type(progression.aggregation).__name__,
                        **serialize_dataclass(progression.aggregation),
                    }
                    if progression.aggregation is not None
                    else None
                ),
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
    _require_artifact(payload)
    source = Document.model_validate(require_mapping(payload.get("source"), name="source"))
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
            validation = validation.append(deserialize_validation_node(node_payload))
        citations.append(validation)
    return ValidatedDocument(source=source, citations=tuple(citations))


def _require_artifact(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        msg = f"Unsupported serialization schema version: {payload.get('schema_version')!r}"
        raise ValueError(msg)
    if payload.get("artifact_type") != _ARTIFACT_TYPE:
        msg = f"Expected artifact_type={_ARTIFACT_TYPE!r}"
        raise ValueError(msg)


def deserialize_validation_node(value: object) -> ValidationNode:
    """Recover one typed validation node from its JSON-ready representation.

    Document-native stages store the same representation in a stage-neutral
    trace node.  Exposing this decoder lets a later ``Document -> Document``
    stage resume from that evidence without repeating a lookup.
    """
    payload = require_mapping(value, name="validation node")
    node_type_name = payload.get("node_type")
    if not isinstance(node_type_name, str) or node_type_name not in _NODE_TYPES:
        msg = f"Unknown validation node type: {node_type_name!r}"
        raise ValueError(msg)
    node_type = _NODE_TYPES[node_type_name]
    fields = {key: value for key, value in payload.items() if key != "node_type"}
    fields["status"] = ValidationNodeStatus(fields["status"])
    fields["outcome"] = (
        None
        if node_type is YearCheckNode and fields["outcome"] is None
        else _OUTCOME_TYPES[node_type](fields["outcome"])
    )
    fields["depends_on"] = tuple(require_list(fields["depends_on"], name="node.depends_on"))
    if node_type in _IVR_NODE_TYPES:
        run = fields.get("run")
        fields["run"] = (
            deserialize_ivr_run(require_mapping(run, name="node.run")) if run is not None else None
        )
    if node_type is MelleaDocketCitationReextractionNode and fields.get("grounded_case_name") is not None:
        fields["grounded_case_name"] = _deserialize_case_name(
            fields["grounded_case_name"], name="node.grounded_case_name"
        )

    if node_type in (
        DocketRootSearchNode,
        GovInfoDocketSearchNode,
        FullReporterSearchNode,
        GovInfoFullReporterSearchNode,
    ):
        fields["candidates"] = _freeze_search_results(
            require_list(fields["candidates"], name="node.candidates")
        )
        fields["attempts"] = tuple(
            MetadataSearchAttempt(
                **{
                    **require_mapping(item, name="node.attempts[]"),
                    "status": ValidationNodeStatus(require_mapping(item, name="node.attempts[]")["status"]),
                    "candidates": _freeze_search_results(
                        require_list(
                            require_mapping(item, name="node.attempts[]")["candidates"],
                            name="node.attempts[].candidates",
                        )
                    ),
                }
            )
            for item in require_list(fields.get("attempts", []), name="node.attempts")
        )
    elif node_type is DocketMetadataShortlistNode:
        fields["candidates"] = tuple(
            DocketMetadataShortlistCandidate(
                **{
                    **require_mapping(item, name="node.candidates[]"),
                    "parties": tuple(
                        require_list(
                            require_mapping(item, name="node.candidates[]")["parties"],
                            name="node.candidates[].parties",
                        )
                    ),
                }
            )
            for item in require_list(fields["candidates"], name="node.candidates")
        )
    elif node_type is ExactLocatorLookupNode:
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
    elif node_type is RootBodySearchNode:
        fields["source"] = RootBodySearchSource(fields["source"])
        fields["candidates"] = _freeze_search_results(
            require_list(fields["candidates"], name="node.candidates")
        )
        fields["search_attempts"] = tuple(
            BodySearchAttempt(
                **{
                    **require_mapping(item, name="node.search_attempts[]"),
                    "status": ValidationNodeStatus(
                        require_mapping(item, name="node.search_attempts[]")["status"]
                    ),
                }
            )
            for item in require_list(fields.get("search_attempts", []), name="node.search_attempts")
        )
    elif node_type is ReporterPageRetrievalNode:
        fields["evidence"] = (
            ReporterPageEvidence(**require_mapping(fields["evidence"], name="node.evidence"))
            if fields["evidence"] is not None
            else None
        )
    elif node_type in (MelleaCaseNameReextractionNode, MelleaCaseNameReviewNode):
        name = fields.get("case_name")
        if name is not None:
            fields["case_name"] = _deserialize_case_name(name, name="node.case_name")
    elif node_type is MelleaCitingPropositionExtractionNode:
        fields["context_span"] = _deserialize_span(fields["context_span"], name="node.context_span")
        fields["proposition_span"] = _optional_span(
            fields["proposition_span"],
            name="node.proposition_span",
        )
        fields["proposition_match_method"] = _optional_enum(
            EvidenceQuoteMatchMethod,
            fields["proposition_match_method"],
        )
    elif node_type is MelleaPinpointCheckNode:
        fields["evidence_span"] = _optional_span(fields["evidence_span"], name="node.evidence_span")
        fields["evidence_match_method"] = _optional_enum(
            EvidenceQuoteMatchMethod,
            fields["evidence_match_method"],
        )
    elif node_type in (
        LocatorCandidateAssessmentNode,
        OpinionSearchCandidateAssessmentNode,
        RecapSearchCandidateAssessmentNode,
    ):
        for field_name in ("case_name_outcome", "year_outcome", "court_outcome"):
            fields[field_name] = (
                None
                if field_name == "year_outcome" and fields[field_name] is None
                else AggregatedFieldOutcome(fields[field_name])
            )
    elif node_type is MelleaLocatorCandidateChoiceNode:
        fields["candidate_indices"] = tuple(
            require_list(fields["candidate_indices"], name="node.candidate_indices")
        )
    elif node_type is LocatorIdentityResolutionNode:
        fields["matching_candidate_indices"] = tuple(
            require_list(fields["matching_candidate_indices"], name="node.matching_candidate_indices")
        )
    elif node_type in (LocatorCitationSummaryNode, SearchCitationSummaryNode):
        overall_outcome = fields["overall_outcome"]
        fields["overall_outcome"] = (
            CitationSummaryAssessmentOutcome(overall_outcome) if overall_outcome is not None else None
        )
        candidate_outcome_type = (
            LocatorCandidateAssessmentOutcome
            if node_type is LocatorCitationSummaryNode
            else SearchCandidateAssessmentOutcome
        )
        fields["candidates"] = tuple(
            _deserialize_summary_candidate(item, outcome_type=candidate_outcome_type)
            for item in require_list(fields["candidates"], name="node.candidates")
        )
    return node_type(**fields)


def _deserialize_summary_candidate(
    value: object,
    *,
    outcome_type: type[LocatorCandidateAssessmentOutcome] | type[SearchCandidateAssessmentOutcome],
) -> CitationSummaryCandidate:
    fields = dict(require_mapping(value, name="summary candidate"))
    fields["provenance"] = CandidateProvenance(fields["provenance"])
    fields["outcome"] = outcome_type(fields["outcome"])
    for field_name in ("case_name_outcome", "year_outcome", "court_outcome"):
        fields[field_name] = (
            None
            if field_name == "year_outcome" and fields[field_name] is None
            else AggregatedFieldOutcome(fields[field_name])
        )
    pinpoint = fields.get("pinpoint")
    fields["pinpoint"] = _deserialize_summary_pinpoint(pinpoint) if pinpoint is not None else None
    return CitationSummaryCandidate(**fields)


def _deserialize_summary_pinpoint(value: object) -> CitationSummaryPinpoint:
    fields = dict(require_mapping(value, name="summary pinpoint"))
    fields["status"] = ValidationNodeStatus(fields["status"])
    fields["outcome"] = MelleaPinpointCheckOutcome(fields["outcome"])
    fields["citing_context_span"] = _deserialize_span(
        fields["citing_context_span"],
        name="summary pinpoint.citing_context_span",
    )
    fields["citation_span"] = _deserialize_span(
        fields["citation_span"],
        name="summary pinpoint.citation_span",
    )
    fields["proposition_span"] = _optional_span(
        fields["proposition_span"],
        name="summary pinpoint.proposition_span",
    )
    fields["evidence_span"] = _optional_span(
        fields["evidence_span"],
        name="summary pinpoint.evidence_span",
    )
    fields["evidence_match_method"] = _optional_enum(
        EvidenceQuoteMatchMethod,
        fields["evidence_match_method"],
    )
    return CitationSummaryPinpoint(**fields)


def _deserialize_case_name(value: object, *, name: str) -> CaseName:
    payload = require_mapping(value, name=name)
    written = payload.get("text")
    if not isinstance(written, str):
        raise ValueError(f"{name}.text must be a string")
    return CaseName(
        span=_deserialize_span(payload.get("span"), name=f"{name}.span"),
        text=written,
        plaintiff=_optional_string(payload.get("plaintiff"), name=f"{name}.plaintiff"),
        defendant=_optional_string(payload.get("defendant"), name=f"{name}.defendant"),
    )


def _deserialize_span(value: object, *, name: str) -> Span:
    payload = require_mapping(value, name=name)
    return Span(
        start=_required_integer(payload.get("start"), name=f"{name}.start"),
        end=_required_integer(payload.get("end"), name=f"{name}.end"),
    )


def _optional_span(value: object, *, name: str) -> Span | None:
    return None if value is None else _deserialize_span(value, name=name)


def _optional_enum(enum_type: type[EnumT], value: object) -> EnumT | None:
    return None if value is None else enum_type(value)


def _deserialize_cluster(value: object) -> CourtListenerOpinionCluster:
    payload = require_mapping(value, name="opinion cluster")
    return CourtListenerOpinionCluster(
        cluster_id=_optional_string(payload.get("cluster_id"), name="cluster.cluster_id"),
        opinion_url=_optional_string(payload.get("opinion_url"), name="cluster.opinion_url"),
        case_name=_optional_string(payload.get("case_name"), name="cluster.case_name"),
        date_filed=_optional_string(payload.get("date_filed"), name="cluster.date_filed"),
        other_dates=_optional_string(payload.get("other_dates"), name="cluster.other_dates"),
        court=_optional_string(payload.get("court"), name="cluster.court"),
        court_id=_optional_string(payload.get("court_id"), name="cluster.court_id"),
        docket_id=_optional_string(payload.get("docket_id"), name="cluster.docket_id"),
        citations=tuple(
            CourtListenerOpinionClusterCitation(
                volume=_required_string(item.get("volume"), name="cluster.citations.volume"),
                reporter=_required_string(item.get("reporter"), name="cluster.citations.reporter"),
                page=_required_string(item.get("page"), name="cluster.citations.page"),
            )
            for value in require_list(payload.get("citations", []), name="cluster.citations")
            for item in (require_mapping(value, name="cluster citation"),)
        ),
        sub_opinion_ids=tuple(
            _required_string(value, name="cluster.sub_opinion_ids")
            for value in require_list(payload.get("sub_opinion_ids", []), name="cluster.sub_opinion_ids")
        ),
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


def _required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        msg = f"{name} must be a string"
        raise ValueError(msg)
    return value


def _required_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} must be an integer"
        raise ValueError(msg)
    return value
