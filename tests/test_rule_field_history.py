"""Raw rule spans remain native, node-bound, append-only citation records."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mellea_lrc.field_extraction.models import FieldPrediction
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.field_extraction.models import MLFieldPrediction
from mellea_lrc.model.citations.ml_field_reading import MLFieldReading
from mellea_lrc.model.citations.rule_field_reading import RuleFieldReading
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


def _prediction() -> FieldPrediction:
    return FieldPrediction(
        context_span=Span(0, len(SOURCE)),
        case_name=_span("Smith v. Jones"),
        court=_span("9th Cir."),
        date=_span("1954"),
        pin_cite=_span("495"),
    )


def _reading(citation, prediction: FieldPrediction | None = None) -> RuleFieldReading:
    return RuleFieldReading(
        node_id=citation.nodes[-1].id,
        prediction=_prediction() if prediction is None else prediction,
        case_name_window=160,
        post_locator_window=120,
    )


def _recorded() -> Document:
    document = _created()
    citation = document.citations[0].record("predict_rule_fields")
    citation = citation.with_rule_field_reading(_reading(citation))
    return document.replace_citation(citation).complete_substage("predict_rule_fields")


def test_rule_prediction_roundtrips_and_rewinds_without_reading_normalized_fields() -> None:
    created = _created()
    recorded = _recorded()
    restored = Document.model_validate_json(recorded.model_dump_json())
    citation = restored.citations[0]

    assert restored == recorded
    assert restored.model_dump_json() == recorded.model_dump_json()
    assert restored.get_substage("create") == created
    assert restored.get_substage("predict_rule_fields") == recorded
    assert citation.get_rule_field_prediction() == _prediction()
    assert citation.case_name == ()
    assert citation.court == ()
    assert citation.date == ()
    assert citation.pin_cite is None
    assert citation.locator == created.citations[0].locator
    assert json.loads(recorded.model_dump_json())["citations"][0]["rule_field_reading"] == {
        "node_id": citation.nodes[-1].id,
        "prediction": _prediction().model_dump(mode="json"),
        "case_name_window": 160,
        "post_locator_window": 120,
    }


def test_rule_prediction_requires_a_new_decision_and_cannot_be_reassigned() -> None:
    citation = _created().citations[0]
    with pytest.raises(ValueError, match="unavailable"):
        citation.get_rule_field_prediction()
    with pytest.raises(ValueError, match="decision node"):
        citation.with_rule_field_reading(_reading(citation))

    citation = citation.record("predict_rule_fields")
    with pytest.raises(ValueError, match="current decision node"):
        citation.with_rule_field_reading(
            _reading(citation).model_copy(update={"node_id": citation.nodes[0].id})
        )
    recorded = citation.with_rule_field_reading(_reading(citation))
    with pytest.raises(ValueError, match="already recorded"):
        recorded.with_rule_field_reading(_reading(recorded))
    later = recorded.record("read_case_name")
    with pytest.raises(ValueError, match="already recorded"):
        later.with_rule_field_reading(_reading(later))


@pytest.mark.parametrize("clear", [False, True])
def test_document_rejects_removing_or_replacing_saved_rule_prediction(clear: bool) -> None:
    document = _recorded()
    citation = document.citations[0].record("change_rule_prediction")
    replacement = None if clear else _reading(citation)
    citation = citation._with_log(rule_field_reading=replacement)
    with pytest.raises(ValueError, match="rule_field_reading must be append-only"):
        document.replace_citation(citation)


def test_rule_and_ml_predictions_coexist_at_independent_nodes() -> None:
    rules = _recorded()
    citation = rules.citations[0].record("predict_ml_fields")
    ml_prediction = MLFieldPrediction(
        **_prediction().model_dump(mode="python"), model_id="test/model", revision="a" * 40, variant="bert"
    )
    citation = citation.with_ml_field_reading(
        MLFieldReading(node_id=citation.nodes[-1].id, prediction=ml_prediction)
    )
    both = rules.replace_citation(citation).complete_substage("predict_ml_fields")
    restored = Document.model_validate_json(both.model_dump_json())

    assert restored == both
    assert restored.citations[0].get_rule_field_prediction() == _prediction()
    assert restored.citations[0].get_ml_field_prediction() == ml_prediction
    assert restored.get_substage("predict_rule_fields") == rules
    assert restored.get_substage("create") == _created()
    assert restored.get_substage("predict_rule_fields").citations[0].ml_field_reading is None


@pytest.mark.parametrize(
    "prediction,error",
    [
        (FieldPrediction(context_span=Span(0, len(SOURCE) + 1)), "outside the source"),
        (FieldPrediction(context_span=Span(0, 10)), "contain the target locator"),
        (FieldPrediction(context_span=Span(0, len(SOURCE)), date=Span(5, 6)), "cannot be blank"),
    ],
)
def test_rule_prediction_checks_source_and_target_bounds(prediction, error) -> None:
    document = _created()
    citation = document.citations[0].record("predict_rule_fields")
    citation = citation.with_rule_field_reading(_reading(citation, prediction))
    with pytest.raises(ValueError, match=error):
        document.replace_citation(citation)


def test_native_json_rejects_tampered_rule_prediction_bounds() -> None:
    payload = json.loads(_recorded().model_dump_json())
    payload["citations"][0]["rule_field_reading"]["prediction"]["context_span"]["end"] = len(SOURCE) + 1
    with pytest.raises(ValueError, match="outside the source"):
        Document.model_validate(payload)


@pytest.mark.parametrize("field", ["case_name_window", "post_locator_window"])
def test_rule_prediction_requires_positive_explicit_window_provenance(field: str) -> None:
    payload = _reading(_created().citations[0]).model_dump(mode="python")
    del payload[field]
    with pytest.raises(ValidationError, match="Field required"):
        RuleFieldReading.model_validate(payload)
    payload[field] = 0
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        RuleFieldReading.model_validate(payload)
