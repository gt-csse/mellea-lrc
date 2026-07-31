"""Tests for validated-document serialization."""

import json

from mellea_lrc.core.citations import (
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
)
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerOpinionCluster, CourtListenerSearchResult
from mellea_lrc.extraction import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.serialization import (
    deserialize_extracted_document,
    deserialize_validated_document,
    serialize_extracted_document,
    serialize_validated_document,
)
from mellea_lrc.validation import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationOutcome,
    CandidateEvaluationSource,
    CandidateProvenance,
    CandidateSelectionNode,
    CandidateSelectionOutcome,
    CitationSummaryAssessmentOutcome,
    CitationSummaryCandidate,
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
    RecapSearchCandidateAssessmentNode,
    RecapSearchNode,
    RecapSearchOutcome,
    ReporterPageEvidence,
    ReporterPageRetrievalNode,
    ReporterPageRetrievalOutcome,
    SearchCandidateAssessmentOutcome,
    ValidationNodeStatus,
    YearCheckNode,
    initialize_validation,
)


def _document_with_one_citation() -> ExtractedDocument:
    """Build one extracted case citation for serializer tests."""
    text = "Brown v. Board of Education, 347 U.S. 483 (1954)."
    preprocessed = preprocess_plain_text_from_string(text)
    matched_text = "347 U.S. 483"
    start = text.index(matched_text)
    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(
            ExtractedCitation(
                citation_id="cite-0001",
                span=Span(0, len(text) - 1),
                locator_span=Span(start, start + len(matched_text)),
                matched_text=matched_text,
                citation=FullCaseCitation(
                    plaintiff="Brown",
                    defendant="Board of Education",
                    volume="347",
                    reporter="U.S.",
                    page="483",
                    year="1954",
                    court="scotus",
                ),
            ),
        ),
        extraction_metadata=ExtractionMetadata(),
    )


def test_extracted_document_round_trip_preserves_recoverable_fields() -> None:
    """Preserve source provenance, both citation spans, and canonical fields."""
    document = _document_with_one_citation()

    payload = serialize_extracted_document(document)

    assert payload["schema_version"] == 2
    assert payload["artifact_type"] == "extracted_document"
    assert payload["citations"][0]["span"] == {"start": 0, "end": len(document.text) - 1}
    assert payload["citations"][0]["locator_span"] == {"start": 29, "end": 41}
    assert deserialize_extracted_document(payload) == document
    assert json.loads(json.dumps(payload)) == payload


def test_extracted_document_round_trip_supports_every_canonical_citation_type() -> None:
    """Keep each canonical citation shape recoverable from an extracted artifact."""
    citations = (
        FullCaseCitation(plaintiff="A", defendant="B", volume="1", reporter="U.S.", page="2"),
        FullLawCitation(volume="1", reporter="U.S.C.", page="2"),
        FullJournalCitation(volume="1", reporter="Harv. L. Rev.", page="2"),
        ShortCaseCitation(volume="1", reporter="U.S.", page="2"),
        SupraCitation(pin_cite="2"),
        IdCitation(pin_cite="2"),
        ReferenceCitation(plaintiff="A", defendant="B"),
        UnknownCitation(),
    )
    source = preprocess_plain_text_from_string("x" * len(citations))
    document = ExtractedDocument(
        source_metadata=source.source_metadata,
        text=source.text,
        preprocessing_metadata=source.preprocessing_metadata,
        citations=tuple(
            ExtractedCitation(
                citation_id=f"cite-{index}",
                span=Span(index, index + 1),
                locator_span=Span(index, index + 1),
                matched_text="x",
                citation=citation,
            )
            for index, citation in enumerate(citations)
        ),
        extraction_metadata=ExtractionMetadata(),
    )

    assert deserialize_extracted_document(serialize_extracted_document(document)) == document


def test_serialize_validated_document_preserves_source_and_node_graph() -> None:
    """Emit one source citation and its explicit validation-node dependency."""
    text = "Brown v. Board of Education, 347 U.S. 483 (1954)."
    preprocessed = preprocess_plain_text_from_string(text)
    matched_text = "347 U.S. 483"
    start = text.index(matched_text)
    extracted = ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(
            ExtractedCitation(
                citation_id="cite-0001",
                span=Span(start, start + len(matched_text)),
                locator_span=Span(start, start + len(matched_text)),
                matched_text=matched_text,
                citation=FullCaseCitation(
                    plaintiff="Brown",
                    defendant="Board of Education",
                    volume="347",
                    reporter="U.S.",
                    page="483",
                    year="1954",
                    court="scotus",
                ),
            ),
        ),
        extraction_metadata=ExtractionMetadata(),
    )
    initialized = initialize_validation(extracted)
    node = ExactLocatorLookupNode(
        node_id="cite-0001:exact_locator_lookup",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorLookupOutcome.FOUND,
        locator=matched_text,
        cluster=CourtListenerOpinionCluster(case_name="Brown v. Board of Education"),
        candidate_count=1,
    )
    validated = type(initialized)(
        source=initialized.source,
        citations=(initialized.citations[0].append(node),),
    )

    payload = serialize_validated_document(validated)

    assert payload["schema_version"] == 2
    assert payload["artifact_type"] == "validated_document"
    assert payload["source"]["artifact_type"] == "extracted_document"
    assert payload["source"]["citations"][0]["citation"]["citation_type"] == "FullCaseCitation"
    assert payload["citations"] == [
        {
            "citation_id": "cite-0001",
            "aggregation": None,
            "nodes": [
                {
                    "node_type": "ExactLocatorLookupNode",
                    "node_id": "cite-0001:exact_locator_lookup",
                    "status": "succeeded",
                    "outcome": "found",
                    "locator": "347 U.S. 483",
                    "cluster": {
                        "cluster_id": None,
                        "case_name": "Brown v. Board of Education",
                        "date_filed": None,
                        "court": None,
                        "court_id": None,
                        "docket_id": None,
                        "citations": [],
                        "sub_opinion_ids": [],
                    },
                    "candidate_clusters": [],
                    "candidate_count": 1,
                    "status_message": None,
                    "outcome_message": None,
                    "error": None,
                    "depends_on": [],
                }
            ],
        }
    ]
    assert deserialize_validated_document(payload) == validated
    assert json.loads(json.dumps(payload)) == payload


def test_serialize_validated_document_preserves_frozen_opinion_search_results() -> None:
    """Expose immutable upstream opinion results without losing their fields."""
    document = _document_with_one_citation()
    initialized = initialize_validation(document)
    preparation = MelleaCaseNameQueryPreparationNode(
        node_id="cite-0001:mellea_case_name_query_preparation",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaCaseNameQueryPreparationOutcome.PREPARED,
        query='caseName:("Brown" AND "Board")',
        query_plaintiff="Brown",
        query_defendant="Board",
        court_id="scotus",
        depends_on=(),
    )
    response = CourtListenerSearchResult.from_payload(
        query=preparation.query,
        search_type="o",
        semantic=False,
        count=1,
        results=[{"cluster_id": 123, "caseName": "Brown v. Board", "meta": {"rank": 1}}],
        next_cursor=None,
        previous_cursor=None,
    )
    search = OpinionSearchNode(
        node_id="cite-0001:opinion_search",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=OpinionSearchOutcome.SEARCHED,
        query=response.query,
        result_count=response.count,
        results=response.results,
        next_cursor=response.next_cursor,
        depends_on=(preparation.node_id,),
    )
    validated = type(initialized)(
        source=initialized.source,
        citations=(initialized.citations[0].append(preparation).append(search),),
    )

    node = serialize_validated_document(validated)["citations"][0]["nodes"][1]

    assert node["results"] == [{"cluster_id": 123, "caseName": "Brown v. Board", "meta": {"rank": 1}}]
    assert json.loads(json.dumps(node)) == node
    assert deserialize_validated_document(serialize_validated_document(validated)) == validated


def test_validated_document_round_trip_supports_every_current_node_type() -> None:
    """Keep every explicit progression node recoverable as the graph grows."""
    initialized = initialize_validation(_document_with_one_citation())
    citation_id = "cite-0001"
    lookup_id = f"{citation_id}:exact_locator_lookup"
    exact_id = f"{citation_id}:exact_case_name_check"
    semantic_id = f"{citation_id}:mellea_case_name_check"
    reextraction_id = f"{citation_id}:mellea_case_name_reextraction"
    preparation_id = f"{citation_id}:mellea_case_name_query_preparation"
    opinion_search_id = f"{citation_id}:opinion_search"
    selection_id = f"{citation_id}:candidate_selection"
    candidate_id = f"{citation_id}:candidate_evaluation"
    cluster = CourtListenerOpinionCluster(
        cluster_id="123",
        case_name="Brown v. Board of Education",
        date_filed="1954-05-17",
        court="Supreme Court",
        court_id="scotus",
        docket_id="42",
    )
    nodes = (
        ExactLocatorLookupNode(
            node_id=lookup_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorLookupOutcome.FOUND,
            locator="347 U.S. 483",
            cluster=cluster,
            candidate_count=1,
        ),
        ExactCaseNameCheckNode(
            node_id=exact_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=FieldCheckOutcome.MISMATCH,
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name,
            depends_on=(lookup_id,),
        ),
        MelleaCaseNameCheckNode(
            node_id=semantic_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.MATCH,
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name or "",
            depends_on=(exact_id,),
        ),
        MelleaCaseNameReextractionNode(
            node_id=reextraction_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameReextractionOutcome.COMPLETE,
            plaintiff="Brown",
            defendant="Board of Education",
            depends_on=(semantic_id,),
        ),
        MelleaCaseNameQueryPreparationNode(
            node_id=preparation_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameQueryPreparationOutcome.PREPARED,
            query="Brown Board of Education",
            query_plaintiff="Brown",
            query_defendant="Board of Education",
            court_id="scotus",
            depends_on=(reextraction_id,),
        ),
        OpinionSearchNode(
            node_id=opinion_search_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=OpinionSearchOutcome.SEARCHED,
            query="Brown Board of Education",
            result_count=1,
            results=({"cluster_id": 123, "meta": {"rank": 1}},),
            next_cursor="next",
            depends_on=(preparation_id,),
        ),
        RecapSearchNode(
            node_id=f"{citation_id}:recap_search",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=RecapSearchOutcome.SEARCHED,
            query="Brown Board of Education",
            result_count=1,
            results=({"docket_id": 42},),
            next_cursor=None,
            depends_on=(preparation_id,),
        ),
        CandidateSelectionNode(
            node_id=selection_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=CandidateSelectionOutcome.ALL_SELECTED,
            total_candidate_count=1,
            selected_candidate_count=1,
            selection_limit=3,
            depends_on=(opinion_search_id,),
        ),
        CandidateEvaluationNode(
            node_id=candidate_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=CandidateEvaluationOutcome.READY,
            source=CandidateEvaluationSource.OPINION_SEARCH,
            candidate_index=0,
            cluster_id="123",
            case_name=cluster.case_name,
            date_filed=cluster.date_filed,
            court_id=cluster.court_id,
            docket_id=cluster.docket_id,
            record={"cluster_id": 123, "meta": {"rank": 1}},
            depends_on=(selection_id,),
        ),
        MelleaReextractedCaseNameCheckNode(
            node_id=f"{citation_id}:mellea_reextracted_case_name_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.MATCH,
            reextracted_case_name="Brown v. Board of Education",
            retrieved_case_name=cluster.case_name,
            depends_on=(reextraction_id,),
        ),
        DocketCourtRetrievalNode(
            node_id=f"{citation_id}:docket_court_retrieval",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=DocketCourtRetrievalOutcome.FOUND,
            docket_id="42",
            court_id="scotus",
            depends_on=(candidate_id,),
        ),
        CourtCheckNode(
            node_id=f"{citation_id}:court_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=FieldCheckOutcome.MATCH,
            extracted_court_id="scotus",
            retrieved_court_id="scotus",
            depends_on=(candidate_id,),
        ),
        ReporterPageRetrievalNode(
            node_id=f"{candidate_id}:reporter_page_retrieval",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=ReporterPageRetrievalOutcome.FOUND,
            cluster_id="123",
            reporter_citation="347 U.S. 483",
            pin_cite="483",
            citation_index=1,
            evidence=ReporterPageEvidence(
                opinion_id="456",
                opinion_type="020lead",
                text="The plaintiffs contend that segregated public schools are not equal.",
            ),
            depends_on=(candidate_id,),
        ),
        YearCheckNode(
            node_id=f"{citation_id}:year_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=FieldCheckOutcome.MATCH,
            extracted_year="1954",
            retrieved_year="1954",
            depends_on=(candidate_id,),
        ),
        OpinionSearchCandidateAssessmentNode(
            node_id=f"{candidate_id}:opinion_search_candidate_assessment",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=SearchCandidateAssessmentOutcome.POSSIBLE_MATCH,
            candidate_index=0,
            extracted_citation="347 U.S. 483",
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name,
            case_name_outcome=AggregatedFieldOutcome.MATCH,
            case_name_evidence="mellea",
            extracted_year="1954",
            retrieved_year="1954",
            year_outcome=AggregatedFieldOutcome.MATCH,
            extracted_court_id="scotus",
            retrieved_court_id="scotus",
            court_outcome=AggregatedFieldOutcome.MATCH,
            docket_id="42",
            depends_on=(
                semantic_id,
                f"{citation_id}:year_check",
                f"{citation_id}:court_check",
            ),
        ),
        RecapSearchCandidateAssessmentNode(
            node_id=f"{candidate_id}:recap_search_candidate_assessment",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=SearchCandidateAssessmentOutcome.POSSIBLE_MATCH,
            candidate_index=0,
            extracted_citation="347 U.S. 483",
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name,
            case_name_outcome=AggregatedFieldOutcome.MATCH,
            case_name_evidence="mellea",
            extracted_year="1954",
            retrieved_year="1954",
            year_outcome=AggregatedFieldOutcome.MATCH,
            extracted_court_id="scotus",
            retrieved_court_id="scotus",
            court_outcome=AggregatedFieldOutcome.MATCH,
            docket_id="42",
            depends_on=(
                semantic_id,
                f"{citation_id}:year_check",
                f"{citation_id}:court_check",
            ),
        ),
        LocatorCandidateAssessmentNode(
            node_id=f"{candidate_id}:locator_candidate_assessment",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorCandidateAssessmentOutcome.MATCH,
            candidate_index=0,
            extracted_citation="347 U.S. 483",
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name,
            case_name_outcome=AggregatedFieldOutcome.MATCH,
            case_name_evidence="mellea",
            extracted_year="1954",
            retrieved_year="1954",
            year_outcome=AggregatedFieldOutcome.MATCH,
            extracted_court_id="scotus",
            retrieved_court_id="scotus",
            court_outcome=AggregatedFieldOutcome.MATCH,
            docket_id="42",
            depends_on=(
                semantic_id,
                f"{citation_id}:year_check",
                f"{citation_id}:court_check",
            ),
        ),
        LocatorCitationSummaryNode(
            node_id=f"{citation_id}:locator_citation_summary",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorCitationSummaryOutcome.COMPLETE,
            overall_outcome=CitationSummaryAssessmentOutcome.MATCH,
            candidates=(
                CitationSummaryCandidate(
                    provenance=CandidateProvenance.OPINION,
                    candidate_index=0,
                    assessment_node_id=f"{candidate_id}:locator_candidate_assessment",
                    outcome=LocatorCandidateAssessmentOutcome.MATCH,
                    extracted_citation="347 U.S. 483",
                    extracted_case_name="Brown v. Board",
                    retrieved_case_name=cluster.case_name,
                    case_name_outcome=AggregatedFieldOutcome.MATCH,
                    case_name_evidence="mellea",
                    extracted_year="1954",
                    retrieved_year="1954",
                    year_outcome=AggregatedFieldOutcome.MATCH,
                    extracted_court_id="scotus",
                    retrieved_court_id="scotus",
                    court_outcome=AggregatedFieldOutcome.MATCH,
                    docket_id="42",
                ),
            ),
            depends_on=(f"{candidate_id}:locator_candidate_assessment",),
        ),
    )
    validation = initialized.citations[0]
    for node in nodes:
        validation = validation.append(node)
    document = type(initialized)(source=initialized.source, citations=(validation,))

    assert deserialize_validated_document(serialize_validated_document(document)) == document
