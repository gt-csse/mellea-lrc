"""Tests for validated-document serialization."""

import json

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerCitationRecord
from mellea_lrc.extraction import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.serialization import serialize_validated_document
from mellea_lrc.validation import (
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    ValidationNodeStatus,
    initialize_validation,
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
