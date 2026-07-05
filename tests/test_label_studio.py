"""Tests for Label Studio pre-annotation payloads."""

from scripts.label_studio.pre_annotate import build_task_payload


def test_build_task_payload_includes_extraction_prediction() -> None:
    text = "Norton v. Shelby County, 118 U.S. 425, 442 (1886)."
    matched_citation = "Norton v. Shelby County, 118 U.S. 425, 442 (1886)"

    task = build_task_payload(text, source_path="sample.txt")

    assert task["data"]["text"] == text
    assert task["data"]["source_path"] == "sample.txt"

    prediction = task["predictions"][0]
    assert prediction["model_version"] == "eyecite-pre-annotation"
    assert prediction["score"] == 1.0

    label_result = next(result for result in prediction["result"] if result["type"] == "labels")
    assert label_result["from_name"] == "label"
    assert label_result["to_name"] == "text"
    assert label_result["value"]["text"] == matched_citation
    assert label_result["value"]["labels"] == ["FullCaseCitation"]

    field_results = {
        result["from_name"]: result["value"]["text"][0]
        for result in prediction["result"]
        if result["type"] == "textarea"
    }
    assert field_results["defendant"] == "Shelby County"
    assert field_results["volume"] == "118"
    assert field_results["reporter"]["edition"] == "U.S."
    assert field_results["page"] == "425"
