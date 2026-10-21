"""ML predictions remain native, node-bound, append-only citation records."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.ml_field_reading import MLFieldPrediction, MLFieldReading
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span

SOURCE = "Smith v. Jones, 347 U.S. 483, 495 (9th Cir. 1954)."


def _span(quote: str) -> Span:
    start = SOURCE.index(quote)
    return Span(start, start + len(quote))


def _created() -> Document:
    citation = FullReporterCitation.from_locator(
        citation_id="reporter", substage="create", source=SOURCE, span=_span("347 U.S. 483")
    )
    return Document.from_source(SOURCE).add_citation(citation).complete_substage("create")


def _prediction() -> MLFieldPrediction:
    return MLFieldPrediction(
        model_id="test/model",
        revision="a" * 40,
        variant="bert",
        context_span=Span(0, len(SOURCE)),
        case_name=_span("Smith v. Jones"),
        court=_span("9th Cir."),
        date=_span("1954"),
        pin_cite=_span("495"),
    )


def _reading(citation, prediction: MLFieldPrediction | None = None) -> MLFieldReading:
    return MLFieldReading(
        node_id=citation.nodes[-1].id,
        prediction=_prediction() if prediction is None else prediction,
    )


def _recorded() -> Document:
    document = _created()
    citation = document.citations[0].record("predict_ml_fields")
    citation = citation.with_ml_field_reading(_reading(citation))
    return document.replace_citation(citation).complete_substage("predict_ml_fields")


def test_ml_prediction_roundtrips_and_rewinds_without_reading_normalized_fields() -> None:
    created = _created()
    recorded = _recorded()
    restored = Document.model_validate_json(recorded.model_dump_json())
    citation = restored.citations[0]

    assert restored == recorded
    assert restored.model_dump_json() == recorded.model_dump_json()
    assert restored.get_substage("create") == created
    assert restored.get_substage("predict_ml_fields") == recorded
    assert citation.get_ml_field_prediction() == _prediction()
    assert citation.case_name == ()
    assert citation.court == ()
    assert citation.date == ()
    assert citation.pin_cite is None
    assert citation.locator == created.citations[0].locator
    assert json.loads(recorded.model_dump_json())["citations"][0]["ml_field_reading"] == {
        "node_id": citation.nodes[-1].id,
        "prediction": _prediction().model_dump(mode="json"),
    }


def test_ml_prediction_requires_a_new_decision_and_cannot_be_reassigned() -> None:
    citation = _created().citations[0]
    with pytest.raises(ValueError, match="unavailable"):
        citation.get_ml_field_prediction()
    with pytest.raises(ValueError, match="decision node"):
        citation.with_ml_field_reading(_reading(citation))

    citation = citation.record("predict_ml_fields")
    with pytest.raises(ValueError, match="current decision node"):
        citation.with_ml_field_reading(
            _reading(citation).model_copy(update={"node_id": citation.nodes[0].id})
        )
    recorded = citation.with_ml_field_reading(_reading(citation))
    with pytest.raises(ValueError, match="already recorded"):
        recorded.with_ml_field_reading(_reading(recorded))
    later = recorded.record("read_case_name")
    with pytest.raises(ValueError, match="already recorded"):
        later.with_ml_field_reading(_reading(later))


@pytest.mark.parametrize("clear", [False, True])
def test_document_rejects_removing_or_replacing_saved_ml_prediction(clear: bool) -> None:
    document = _recorded()
    citation = document.citations[0].record("change_ml_prediction")
    replacement = None if clear else _reading(citation)
    citation = citation._with_log(ml_field_reading=replacement)
    with pytest.raises(ValueError, match="ml_field_reading must be append-only"):
        document.replace_citation(citation)


@pytest.mark.parametrize(
    "prediction,error",
    [
        (
            MLFieldPrediction(
                model_id="test/model",
                revision="a" * 40,
                variant="bert",
                context_span=Span(0, len(SOURCE) + 1),
            ),
            "outside the source",
        ),
        (
            MLFieldPrediction(
                model_id="test/model", revision="a" * 40, variant="bert", context_span=Span(0, 10)
            ),
            "contain the target locator",
        ),
        (
            MLFieldPrediction(
                model_id="test/model",
                revision="a" * 40,
                variant="bert",
                context_span=Span(0, len(SOURCE)),
                date=Span(5, 6),
            ),
            "cannot be blank",
        ),
    ],
)
def test_ml_prediction_checks_source_and_target_bounds(prediction, error) -> None:
    document = _created()
    citation = document.citations[0].record("predict_ml_fields")
    citation = citation.with_ml_field_reading(_reading(citation, prediction))
    with pytest.raises(ValueError, match=error):
        document.replace_citation(citation)


def test_native_json_rejects_tampered_ml_prediction_bounds() -> None:
    payload = json.loads(_recorded().model_dump_json())
    payload["citations"][0]["ml_field_reading"]["prediction"]["context_span"]["end"] = len(SOURCE) + 1
    with pytest.raises(ValueError, match="outside the source"):
        Document.model_validate(payload)
