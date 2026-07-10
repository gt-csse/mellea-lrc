"""Tests for citation-level node data model."""

from dataclasses import FrozenInstanceError

import pytest

from mellea_lrc.citation_nodes import (
    CitationNode,
    CitationNodeDocument,
    CitationNodeInput,
    CitationNodeStatus,
    CitationStep,
    CitationStepStatus,
    citation_node_document_to_json,
    nodes_from_extracted_document,
    run_node_operation,
    run_operation_for_each_node,
)
from mellea_lrc.assessment import AssessmentSkipReason, SkippedCitationAssessment
from mellea_lrc.citation_nodes.projections import (
    with_assessment_steps,
    with_jurisdiction_steps,
    with_retrieval_steps,
)
from mellea_lrc.core.citations import FullCaseCitation, Reporter
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.jurisdiction_inference.types import (
    CourtInference,
    CourtInferenceStatus,
    Jurisdiction,
    ReporterInference,
    ReporterInferenceStatus,
)
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.retrieval.types import (
    CaseNamePreparationStatus,
    CaseNameSearchCandidate,
    CaseNameSearchCorpus,
    CaseNameSearchPreparation,
    CaseNameSearchProbe,
    CaseNameSearchStatus,
    CaseNameSearchTrace,
    CourtListenerRequestTrace,
    NotFoundCitationRetrieval,
    RetrievalMetadata,
    RetrievedDocument,
)


def _document() -> ExtractedDocument:
    preprocessed = preprocess_plain_text_from_string("Norton v. Shelby County, 118 U.S. 425.")
    locator_start = preprocessed.text.index("118 U.S. 425")
    citation = ExtractedCitation(
        citation_id="cite-0001",
        citation_span=Span(locator_start, locator_start + len("118 U.S. 425")),
        matched_locator_text="118 U.S. 425",
        matched_citation_text="118 U.S. 425",
        citation=FullCaseCitation(
            plaintiff="Norton",
            defendant="Shelby County",
            volume="118",
            page="425",
            reporter=Reporter(
                edition_short_name="U.S.",
                root_short_name="U.S.",
                name="United States Supreme Court Reports",
                cite_type="federal",
                is_scotus=True,
                source="reporters",
            ),
        ),
    )
    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(citation,),
        extraction_metadata=ExtractionMetadata(),
    )


def test_nodes_from_extracted_document_copies_inputs_without_mutating_extraction() -> None:
    """Existing extracted documents are inputs, not the execution trace store."""
    extracted = _document()

    node_document = nodes_from_extracted_document(extracted)

    assert node_document.text == extracted.text
    assert len(node_document.nodes) == len(extracted.citations)
    node = node_document.node_by_id("cite-0001")
    assert node.input.citation_id == extracted.citations[0].citation_id
    assert node.input.citation_span == extracted.citations[0].citation_span
    assert node.input.matched_locator_text == extracted.citations[0].matched_locator_text
    assert node.input.matched_citation_text == extracted.text[
        extracted.citations[0].citation_span.start : extracted.citations[0].citation_span.end
    ]
    assert node.input.citation == extracted.citations[0].citation
    assert node.status is CitationNodeStatus.READY
    assert node.steps == ()
    assert extracted.citations[0].resolves_to is None


def test_citation_node_append_step_returns_new_node_with_trace() -> None:
    """Node transitions append trace steps immutably."""
    node = nodes_from_extracted_document(_document()).node_by_id("cite-0001")
    step = CitationStep(
        operation="exact_lookup",
        status=CitationStepStatus.SUCCEEDED,
        summary="Exact citation lookup found one cluster.",
        data={"cluster_id": 123, "sources": ["courtlistener"]},
    )

    updated = node.append_step(step)

    assert node.steps == ()
    assert updated.steps == (step,)
    assert updated.status is CitationNodeStatus.READY
    assert updated.steps[0].data["cluster_id"] == 123
    assert updated.steps[0].data["sources"] == ("courtlistener",)
    with pytest.raises(TypeError):
        updated.steps[0].data["cluster_id"] = 456  # type: ignore[index]


def test_failed_step_marks_node_failed_by_default() -> None:
    """Operation status drives the coarse node status unless explicitly overridden."""
    node = nodes_from_extracted_document(_document()).node_by_id("cite-0001")

    updated = node.append_step(
        CitationStep(
            operation="exact_lookup",
            status=CitationStepStatus.FAILED,
            summary="CourtListener request failed.",
            error="timeout",
        )
    )

    assert updated.status is CitationNodeStatus.FAILED


def test_node_document_rejects_duplicate_node_ids() -> None:
    """Citation node ids remain document-local unique identifiers."""
    node = nodes_from_extracted_document(_document()).node_by_id("cite-0001")

    with pytest.raises(ValueError, match="unique"):
        CitationNodeDocument(text="118 U.S. 425", nodes=(node, node))


def test_node_document_validates_spans_against_document_text() -> None:
    """Node projections still defend against invalid source spans."""
    node = CitationNode(
        input=CitationNodeInput(
            citation_id="cite-1",
            citation_span=Span(0, 99),
            matched_locator_text="118 U.S. 425",
            matched_citation_text="118 U.S. 425",
            citation=FullCaseCitation(
                volume="118",
                page="425",
                reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S."),
            ),
        )
    )

    with pytest.raises(ValueError, match="citation_span exceeds"):
        CitationNodeDocument(text="short", nodes=(node,))


def test_node_input_is_frozen() -> None:
    """Citation inputs are stable snapshots of extraction output."""
    node = nodes_from_extracted_document(_document()).node_by_id("cite-0001")

    with pytest.raises(FrozenInstanceError):
        node.input.matched_locator_text = "changed"  # type: ignore[misc]


def test_run_node_operation_replaces_only_target_node() -> None:
    """The internal runner advances one citation without document-wide alignment work."""
    document = nodes_from_extracted_document(_document())

    class Operation:
        name = "mark"

        def run(self, node: CitationNode) -> CitationNode:
            return node.append_step(
                CitationStep(
                    operation=self.name,
                    status=CitationStepStatus.SUCCEEDED,
                    summary="Marked target node.",
                )
            )

    updated = run_node_operation(document, "cite-0001", Operation())

    assert document.node_by_id("cite-0001").steps == ()
    assert updated.node_by_id("cite-0001").steps[0].operation == "mark"


def test_run_node_operation_rejects_id_rewriting_operation() -> None:
    """Node operations cannot silently retarget their result to another citation."""
    document = nodes_from_extracted_document(_document())

    class BadOperation:
        name = "bad"

        def run(self, node: CitationNode) -> CitationNode:
            return CitationNode(
                input=CitationNodeInput(
                    citation_id="changed",
                    citation_span=node.input.citation_span,
                    matched_locator_text=node.input.matched_locator_text,
                    matched_citation_text=node.input.matched_citation_text,
                    citation=node.input.citation,
                )
            )

    with pytest.raises(ValueError, match="changed citation id"):
        run_node_operation(document, "cite-0001", BadOperation())


def test_run_operation_for_each_node_keeps_operation_independent() -> None:
    """Batch execution is just repeated independent per-citation execution."""
    document = nodes_from_extracted_document(_document())

    class Operation:
        name = "observe"

        def run(self, node: CitationNode) -> CitationNode:
            return node.append_step(
                CitationStep(
                    operation=self.name,
                    status=CitationStepStatus.SUCCEEDED,
                    summary=f"Observed {node.citation_id}.",
                )
            )

    updated = run_operation_for_each_node(document, Operation())

    assert [node.steps[0].summary for node in updated.nodes] == ["Observed cite-0001."]


def test_citation_node_document_to_json_projects_trace_shape() -> None:
    """Node traces have a stable JSON-compatible projection for snapshots/UI."""
    document = nodes_from_extracted_document(_document())
    node = document.node_by_id("cite-0001").append_step(
        CitationStep(
            operation="exact_lookup",
            status=CitationStepStatus.SUCCEEDED,
            summary="Exact lookup found one match.",
            data={"cluster_id": 10, "sources": ["courtlistener"]},
        )
    )
    projected = citation_node_document_to_json(
        CitationNodeDocument(text=document.text, nodes=(node,))
    )

    assert projected == {
        "schema_version": 19,
        "artifact_type": "citation_node_document",
        "text": "Norton v. Shelby County, 118 U.S. 425.",
        "nodes": [
            {
                "citation_id": "cite-0001",
                "status": "ready",
                "input": {
                    "citation_id": "cite-0001",
                    "citation_span": {"start": 25, "end": 37},
                    "matched_locator_text": "118 U.S. 425",
                    "matched_citation_text": "118 U.S. 425",
                    "citation": {
                        "type": "FullCaseCitation",
                        "plaintiff": "Norton",
                        "defendant": "Shelby County",
                        "volume": "118",
                        "page": "425",
                        "pin_cite": None,
                        "extra": None,
                        "year": None,
                        "court": None,
                        "parenthetical": None,
                        "reporter": {
                            "edition_short_name": "U.S.",
                            "root_short_name": "U.S.",
                            "name": "United States Supreme Court Reports",
                            "cite_type": "federal",
                            "is_scotus": True,
                            "source": "reporters",
                        },
                    },
                    "resolves_to": None,
                },
                "steps": [
                    {
                        "step_id": None,
                        "operation": "exact_lookup",
                        "status": "succeeded",
                        "depends_on": [],
                        "lane": None,
                        "summary": "Exact lookup found one match.",
                        "data": {
                            "cluster_id": 10,
                            "sources": ["courtlistener"],
                        },
                        "error": None,
                    }
                ],
            }
        ],
    }


def test_not_found_retrieval_projects_granular_case_name_search_chain() -> None:
    """Candidate fallback traces retain preparation, query, probe, and result evidence."""
    extracted = _document()
    node_document = nodes_from_extracted_document(extracted)
    retrieval = RetrievedDocument(
        source_metadata=extracted.source_metadata,
        text=extracted.text,
        preprocessing_metadata=extracted.preprocessing_metadata,
        citations=extracted.citations,
        extraction_metadata=extracted.extraction_metadata,
        jurisdictions=(
            Jurisdiction(
                reporter_inference=ReporterInference(
                    reporter=extracted.citations[0].citation.reporter,
                    status=ReporterInferenceStatus.RECOGNIZED,
                    mlz_jurisdictions=("us",),
                ),
                court_inference=CourtInference(
                    extracted_court=None,
                    status=CourtInferenceStatus.MISSING_COURT,
                    courts_db_classification=None,
                ),
            ),
        ),
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
        retrievals=(
            NotFoundCitationRetrieval(
                citation_id="cite-0001",
                locator="118 U.S. 425",
                source="courtlistener",
                request_trace=CourtListenerRequestTrace(http_status=404),
                    candidate_search=CaseNameSearchTrace(
                        status=CaseNameSearchStatus.PARTIAL,
                        query="caseName:(Norton AND Shelby) AND court_id:scotus",
                        preparation=CaseNameSearchPreparation(
                            status=CaseNamePreparationStatus.ACCEPTED,
                            original_case_name="Norton v. Shelby County",
                            plaintiff="Norton",
                            defendant="Shelby County",
                            prepared_case_name="Norton v. Shelby County",
                            court="scotus",
                            locator="118 U.S. 425",
                            source="llm",
                            llm_classification="complete_case_name",
                            llm_reason="Parties appear before the locator.",
                        ),
                        probes=(
                        CaseNameSearchProbe(
                            corpus=CaseNameSearchCorpus.OPINIONS,
                            status=CaseNameSearchStatus.SEARCHED,
                            request_trace=CourtListenerRequestTrace(
                                http_status=200,
                                cache="miss",
                                key="opinion-search",
                            ),
                            case_count=2,
                            candidates=(
                                CaseNameSearchCandidate(
                                    case_name="Norton v. Shelby County",
                                    court_id="scotus",
                                    date_filed="1886-05-10",
                                    cluster_id="100",
                                ),
                            ),
                        ),
                        CaseNameSearchProbe(
                            corpus=CaseNameSearchCorpus.RECAP,
                            status=CaseNameSearchStatus.SEARCH_FAILED,
                            request_trace=CourtListenerRequestTrace(
                                http_status=503,
                                error_message="RECAP unavailable",
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    projected = with_retrieval_steps(node_document, retrieval)
    node = projected.node_by_id("cite-0001")

    assert [step.operation for step in node.steps] == [
        "retrieval.exact_lookup",
        "retrieval.fallback_decision",
        "retrieval.case_name_preparation",
        "retrieval.candidate_query",
        "retrieval.corpus_probe",
        "retrieval.corpus_probe",
        "retrieval.candidate_results",
    ]
    preparation = node.steps[2]
    assert preparation.data["original_case_name"] == "Norton v. Shelby County"
    assert preparation.data["llm_status"] == "accepted"
    assert preparation.data["llm_classification"] == "complete_case_name"
    assert preparation.data["prepared_case_name"] == "Norton v. Shelby County"
    assert preparation.depends_on == ("cite-0001:retrieval:fallback_decision",)
    assert node.steps[4].lane == "o"
    assert node.steps[4].data["candidates"][0]["case_name"] == "Norton v. Shelby County"
    assert node.steps[5].lane == "r"
    assert node.steps[5].status is CitationStepStatus.FAILED
    assert node.steps[6].depends_on == (
        "cite-0001:retrieval:corpus_probe:o",
        "cite-0001:retrieval:corpus_probe:r",
    )


def test_projected_node_dependencies_keep_jurisdiction_as_terminal_side_branch() -> None:
    """Assessment consumes retrieval, not always-run jurisdiction inference."""
    extracted = _document()
    node_document = nodes_from_extracted_document(extracted)
    inferred = RetrievedDocument(
        source_metadata=extracted.source_metadata,
        text=extracted.text,
        preprocessing_metadata=extracted.preprocessing_metadata,
        citations=extracted.citations,
        extraction_metadata=extracted.extraction_metadata,
        jurisdictions=(
            Jurisdiction(
                reporter_inference=ReporterInference(
                    reporter=extracted.citations[0].citation.reporter,
                    status=ReporterInferenceStatus.RECOGNIZED,
                    mlz_jurisdictions=("us",),
                ),
                court_inference=CourtInference(
                    extracted_court=None,
                    status=CourtInferenceStatus.MISSING_COURT,
                    courts_db_classification=None,
                ),
            ),
        ),
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="test"),
        retrievals=(
            NotFoundCitationRetrieval(
                citation_id="cite-0001",
                locator="118 U.S. 425",
                source="courtlistener",
                request_trace=CourtListenerRequestTrace(http_status=404),
                candidate_search=CaseNameSearchTrace(
                    status=CaseNameSearchStatus.SKIPPED_NO_CASE_NAME,
                    query=None,
                    preparation=CaseNameSearchPreparation(
                        status=CaseNamePreparationStatus.EMPTY,
                        original_case_name=None,
                        locator="118 U.S. 425",
                        source="llm",
                    ),
                    probes=(),
                ),
            ),
        ),
    )

    projected = with_jurisdiction_steps(node_document, inferred)
    projected = with_retrieval_steps(projected, inferred)
    projected = with_assessment_steps(
        projected,
        (
            SkippedCitationAssessment(
                citation_id="cite-0001",
                reason=AssessmentSkipReason.RETRIEVAL_NOT_ELIGIBLE,
                message="not found",
            ),
        ),
    )

    node = projected.node_by_id("cite-0001")
    jurisdiction = next(step for step in node.steps if step.operation == "jurisdiction.inference")
    assessment = next(step for step in node.steps if step.operation == "assessment.field_check")
    consumed_dependencies = {dependency for step in node.steps for dependency in step.depends_on}

    assert jurisdiction.step_id not in consumed_dependencies
    assert assessment.depends_on == ("cite-0001:retrieval:candidate_results",)
