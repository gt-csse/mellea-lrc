"""Tests for neutral and Label Studio-specific serialization."""

from mellea_lrc.extraction import extract_citations
from mellea_lrc.serialization import serialize_document_extraction
from scripts.label_studio.label_studio import to_label_studio_prediction
from scripts.label_studio.pre_annotate import build_task_payload

SAMPLE_TEXT = "Under Norton v. Shelby County, 118 U.S. 425, 442 (1886), an act confers no rights."
RECAP_TEXT = (
    "Case: Example\n"
    "Recovered file description: sample\n\n"
    "--- Plain text ---\n"
    "The Court cites Oconner v. Agilant Sols., Inc., 444 F. Supp. 3d 593.\n"
)


def test_document_extraction_serializes_without_ui_assumptions() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    artifact = serialize_document_extraction(extraction)

    assert artifact["source_path"] is None
    assert artifact["text"] == SAMPLE_TEXT
    assert artifact["citations"]
    assert artifact["counts"]["total"] == len(extraction.citations)
    assert artifact["counts"]["full"] == len(extraction.full_citations)

    citation = artifact["citations"][0]
    assert citation["citation_id"] == extraction.citations[0].citation_id
    assert citation["span"]["start"] == extraction.citations[0].span.start
    assert citation["citation"]["type"] == "FullCaseCitation"
    assert "from_name" not in citation


def test_label_studio_prediction_shape() -> None:
    extraction = extract_citations(SAMPLE_TEXT)
    prediction = to_label_studio_prediction(extraction)

    assert prediction["model_version"] == "eyecite-pre-annotation"
    assert prediction["score"] == 1.0
    assert prediction["result"]

    label_results = [item for item in prediction["result"] if item["type"] == "labels"]
    assert label_results
    assert label_results[0]["value"]["labels"] == ["FullCaseCitation"]

    textarea_results = [item for item in prediction["result"] if item["type"] == "textarea"]
    field_names = {item["from_name"] for item in textarea_results}
    assert "plaintiff" in field_names
    assert "volume" in field_names


def test_label_studio_task_text_matches_prediction_spans() -> None:
    task = build_task_payload(RECAP_TEXT, source_path="sample.txt")
    text = task["data"]["text"]
    prediction = task["predictions"][0]

    assert text.startswith("The Court cites")
    assert task["data"]["source_path"] == "sample.txt"
    assert "source_header" in task["data"]

    label_results = [item for item in prediction["result"] if item["type"] == "labels"]
    assert label_results
    for result in label_results:
        value = result["value"]
        assert text[value["start"] : value["end"]] == value["text"]


def test_label_studio_uses_full_span_text_not_matched_text() -> None:
    extraction = extract_citations(RECAP_TEXT)
    prediction = to_label_studio_prediction(extraction)

    label_result = next(item for item in prediction["result"] if item["type"] == "labels")
    span_text = extraction.text[
        extraction.citations[0].span.start : extraction.citations[0].span.end
    ]

    assert extraction.citations[0].matched_text == "444 F. Supp. 3d 593"
    assert label_result["value"]["text"] == span_text
    assert label_result["value"]["text"] != extraction.citations[0].matched_text
