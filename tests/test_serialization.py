"""Tests for validated-document serialization."""

import json

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerCitationRecord, CourtListenerSearchResult
from mellea_lrc.extraction import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.serialization import serialize_validated_document
from mellea_lrc.validation import (
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    MelleaCaseNameQueryPreparationNode,
    MelleaCaseNameQueryPreparationOutcome,
    OpinionSearchNode,
    OpinionSearchOutcome,
    ValidationNodeStatus,
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
        record=CourtListenerCitationRecord(case_name="Brown v. Board of Education"),
        candidate_count=1,
    )
    validated = type(initialized)(
        source=initialized.source,
        citations=(initialized.citations[0].append(node),),
    )

    payload = serialize_validated_document(validated)

    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == "validated_document"
    assert payload["source"]["citations"][0]["citation"]["citation_type"] == "FullCaseCitation"
    assert payload["citations"] == [
        {
            "citation_id": "cite-0001",
            "nodes": [
                {
                    "node_type": "ExactLocatorLookupNode",
                    "node_id": "cite-0001:exact_locator_lookup",
                    "status": "succeeded",
                    "outcome": "found",
                    "locator": "347 U.S. 483",
                    "record": {
                        "case_name": "Brown v. Board of Education",
                        "date_filed": None,
                        "court": None,
                        "court_id": None,
                        "docket_id": None,
                    },
                    "candidate_count": 1,
                    "error": None,
                    "depends_on": [],
                }
            ],
        }
    ]
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
