"""ML predictions remain source-grounded, node-bound citation records."""

import json

import pytest

from mellea_lrc.model import Document, PreprocessedDocument, PreprocessingMetadata, SourceMetadata, Span
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.ml_field_reading import MLFieldPrediction, MLFieldReading

SOURCE = "Smith v. Jones, 347 U.S. 483, 495 (9th Cir. 1954)."


def _span(quote: str) -> Span:
    start = SOURCE.index(quote)
    return Span(start, start + len(quote))


def _created() -> Document:
    document = Document.from_preprocessed(
        PreprocessedDocument(
            text=SOURCE,
            source_metadata=SourceMetadata(path="brief.txt"),
            preprocessing_metadata=PreprocessingMetadata(),
        )
    )
    citation = FullReporterCitation.from_locator(
        citation_id="reporter", substage="create", source=SOURCE, span=_span("347 U.S. 483")
    )
    return document.add_citation(citation).complete_substage("create")


def _prediction(**overrides) -> MLFieldPrediction:
    values = {
        "model_id": "test/model",
        "revision": "a" * 40,
        "variant": "bert",
        "context_span": Span(0, len(SOURCE)),
        "case_name": _span("Smith v. Jones"),
        "court": _span("9th Cir."),
        "date": _span("1954"),
        "pin_cite": _span("495"),
    }
    return MLFieldPrediction.model_validate({**values, **overrides})


def _reading(citation, prediction: MLFieldPrediction | None = None) -> MLFieldReading:
    return MLFieldReading(
        node_id=citation.nodes[-1].id,
        prediction=_prediction() if prediction is None else prediction,
    )


def _recorded() -> Document:
    document = _created()
    citation = document.citations[0].record("predict_ml_fields")
    return document.replace_citation(citation.with_ml_field_reading(_reading(citation))).complete_substage(
        "predict_ml_fields"
    )


def test_ml_prediction_roundtrips_and_rewinds_without_reading_normalized_fields():
    created = _created()
    recorded = _recorded()
    restored = Document.model_validate_json(recorded.model_dump_json())
    citation = restored.citations[0]

    assert restored == recorded
    assert restored.get_substage("create") == created
    assert restored.get_substage("predict_ml_fields") == recorded
    assert citation.get_ml_field_prediction() == _prediction()
    assert citation.case_name == citation.court == citation.date == ()
    assert citation.pin_cite is None
    assert citation.locator == created.citations[0].locator
    assert json.loads(restored.model_dump_json())["citations"][0]["ml_field_reading"] == {
        "node_id": citation.nodes[-1].id,
        "prediction": _prediction().model_dump(mode="json"),
    }


def test_atomic_document_without_ml_history_loads():
    created = _created()
    payload = created.model_dump(mode="json")
    del payload["citations"][0]["ml_field_reading"]

    restored = Document.model_validate_json(json.dumps(payload))
    assert restored == created
    assert restored.get_substage("create") == restored
    assert restored.citations[0].ml_field_reading is None


def test_ml_prediction_requires_its_own_decision_node():
    citation = _created().citations[0]
    with pytest.raises(ValueError, match="unavailable"):
        citation.get_ml_field_prediction()
    with pytest.raises(ValueError, match="decision node"):
        citation.with_ml_field_reading(_reading(citation))

    recorded = citation.record("predict_ml_fields")
    with pytest.raises(ValueError, match="current decision node"):
        recorded.with_ml_field_reading(_reading(citation))
    saved = recorded.with_ml_field_reading(_reading(recorded))
    later = saved.record("read_case_name")
    with pytest.raises(ValueError, match="already recorded"):
        later.with_ml_field_reading(_reading(later))


@pytest.mark.parametrize("clear", [False, True])
def test_document_rejects_removing_or_replacing_saved_ml_prediction(clear):
    document = _recorded()
    citation = document.citations[0].record("change_ml_prediction")
    replacement = None if clear else _reading(citation)
    altered = citation._with_log(ml_field_reading=replacement)

    with pytest.raises(ValueError, match="ml_field_reading must be append-only"):
        document.replace_citation(altered)


@pytest.mark.parametrize("field", ["model_id", "revision", "variant"])
def test_ml_prediction_rejects_blank_provenance(field):
    with pytest.raises(ValueError, match="model, revision, and variant"):
        _prediction(**{field: " "})


@pytest.mark.parametrize("invalid_date", [Span(0, 0), Span(len(SOURCE), len(SOURCE) + 1)])
def test_ml_prediction_rejects_empty_or_out_of_context_fields(invalid_date):
    with pytest.raises(ValueError, match="nonempty and inside the prediction context"):
        _prediction(date=invalid_date)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"context_span": Span(0, len(SOURCE) + 1)}, "context is outside the source"),
        (
            {"context_span": Span(0, 16), "court": None, "date": None, "pin_cite": None},
            "context must contain the target locator",
        ),
        ({"date": Span(5, 6)}, "field cannot be blank"),
    ],
)
def test_ml_prediction_checks_owning_source_and_locator(overrides, error):
    document = _created()
    citation = document.citations[0].record("predict_ml_fields")
    attempted = citation.with_ml_field_reading(_reading(citation, _prediction(**overrides)))
    with pytest.raises(ValueError, match=error):
        document.replace_citation(attempted)


def test_native_json_rejects_a_prediction_with_a_forged_node():
    payload = _recorded().model_dump(mode="json")
    payload["citations"][0]["ml_field_reading"]["node_id"] = "another:node:1"

    with pytest.raises(ValueError, match="missing node"):
        Document.model_validate_json(json.dumps(payload))
